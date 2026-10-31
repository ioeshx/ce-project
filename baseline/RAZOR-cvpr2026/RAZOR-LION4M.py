# RAZOR Unlearning for CLIP on CelebA ( identities) — with LAION fallback


#!pip -q install "torch>=2.2" "transformers>=4.44" "datasets>=2.20" "torchvision>=0.17" "accelerate>=0.30" "numpy>=1.24" "tqdm>=4.66" "Pillow>=9.5" requests

import os, math, copy, random, json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# --------------------
# Config
# --------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# Fast-mode knobs for Colab (reduce wall-clock without changing the pipeline)
FAST_MODE = True
SAMPLE_BATCHES_DROP = 2      # used in prediction-drop ablation (was 3)
BATCHES_FOR_SIG     = 2      # batches to estimate gradient significance
BATCHES_FOR_FORGET  = 2      # forget batches per round (was 3)
BATCHES_FOR_RETAIN  = 2      # retain batches per round (was 3)

# Choose identities / filelists for CelebA (preferred)
FORGET_ID_NAME  = "Megan Fox"
RETAIN_ID_NAME  = "Taylor Swift"
# If you know the CelebA identity integer IDs, set them here; otherwise leave None and use filename lists or LAION fallback
CELEBA_FORGET_ID = None   # e.g., 1234
CELEBA_RETAIN_ID = None   # e.g., 5678
# Optionally, provide explicit filename allow-lists (relative names like "000001.jpg")
CELEBA_FORGET_FILES = []  # e.g., ["000123.jpg", "000456.jpg", ...]
CELEBA_RETAIN_FILES = []

# LAION fallback caption queries (used only if CelebA IDs/files are not provided)
FORGET_QUERY  = "Megan Fox"
RETAIN_QUERY  = "Taylor Swift"

N_PER_CLASS = 1000        #  1000 samples for each of forget and retain
BATCH_SIZE  = 16

# LSU knobs
K_INIT     = 1      # initial top-k by prediction drop
K_PRIME    = 1      # layers to add per round by significance
MAX_ROUNDS = 2 if FAST_MODE else 3
STEP_SIZE  = 1e-3   # gradient step on selected layers
ETA_FORGET   = 1.0
ETA_MISMATCH = 0.5
ETA_RETAIN   = 1.0
FORGET_TAU = 0.10   # stop when ForgetAcc ≤ τ
RETAIN_TOL = 0.05   # allow ≤5% absolute drop in retain utility

LOGIT_SCALE = 1/0.07
EPS = 1e-8

# --------------------
# Utils
# --------------------

def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
set_seed(SEED)

@dataclass
class Batch:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: List[int]  # 0 for forget-class, 1 for retain-class (binary here)
    captions: List[str]

# --------------------
# Data: Prefer CelebA by identity / filenames; otherwise fall back to LAION by caption
# --------------------
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

from PIL import Image
import io, requests as _requests

forget_samples: List[Dict] = []
retain_samples: List[Dict] = []

# ---------- Helpers ----------

def _resolve_image_from_url(url: str):
    if not url:
        return None
    try:
        r = _requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

# Collate for CLIP

def collate_clip(batch):
    images  = [b["image"] for b in batch]
    captions = [b["caption"] for b in batch]
    labels   = [b["label"] for b in batch]
    proc = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True)
    return Batch(proc["pixel_values"], proc["input_ids"], proc["attention_mask"], labels, captions)

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict]): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

# ---------- Try CelebA first ----------
use_laion_fallback = False
try:
    celeba = load_dataset("celeba", split="train", streaming=True)
    # celeba columns typically include 'image' and 'identity' (int); filenames may be under 'image_id' or similar
    got_forget = 0; got_retain = 0
    for ex in celeba:
        img = ex.get("image")
        if img is None:
            continue
        ident = ex.get("identity")
        fname = ex.get("image_id") or ex.get("file_name") or getattr(img, "filename", None)
        # Match by explicit filelists first if provided
        if CELEBA_FORGET_FILES and isinstance(fname, str) and fname in CELEBA_FORGET_FILES and got_forget < N_PER_CLASS:
            forget_samples.append({"image": img, "label": 0, "caption": FORGET_ID_NAME}); got_forget += 1; continue
        if CELEBA_RETAIN_FILES and isinstance(fname, str) and fname in CELEBA_RETAIN_FILES and got_retain < N_PER_CLASS:
            retain_samples.append({"image": img, "label": 1, "caption": RETAIN_ID_NAME}); got_retain += 1; continue
        # Otherwise, match by numeric identity IDs if provided
        if CELEBA_FORGET_ID is not None and ident == CELEBA_FORGET_ID and got_forget < N_PER_CLASS:
            forget_samples.append({"image": img, "label": 0, "caption": FORGET_ID_NAME}); got_forget += 1
        elif CELEBA_RETAIN_ID is not None and ident == CELEBA_RETAIN_ID and got_retain < N_PER_CLASS:
            retain_samples.append({"image": img, "label": 1, "caption": RETAIN_ID_NAME}); got_retain += 1
        if got_forget >= N_PER_CLASS and got_retain >= N_PER_CLASS:
            break
    if len(forget_samples) < N_PER_CLASS or len(retain_samples) < N_PER_CLASS:
        use_laion_fallback = True
except Exception:
    use_laion_fallback = True

# ---------- If CelebA insufficient, fall back to LAION by captions ----------
if use_laion_fallback:
    stream = load_dataset("jp1924/Laion400m-1", split="train", streaming=True)
    for ex in stream:
        caption = (ex.get("text") or ex.get("caption") or "").strip()
        if not caption:
            continue
        cap_low = caption.lower()
        if FORGET_QUERY.lower() in cap_low and len(forget_samples) < N_PER_CLASS:
            img = ex.get("image") or _resolve_image_from_url(ex.get("url") or ex.get("URL"))
            if img is not None:
                forget_samples.append({"image": img, "label": 0, "caption": caption})
        elif RETAIN_QUERY.lower() in cap_low and len(retain_samples) < N_PER_CLASS:
            img = ex.get("image") or _resolve_image_from_url(ex.get("url") or ex.get("URL"))
            if img is not None:
                retain_samples.append({"image": img, "label": 1, "caption": caption})
        if len(forget_samples) >= N_PER_CLASS and len(retain_samples) >= N_PER_CLASS:
            break

assert len(forget_samples) == N_PER_CLASS and len(retain_samples) == N_PER_CLASS, "Could not assemble 100 samples for each identity. Provide CELEBA_* IDs or filelists, or adjust queries."

util_samples = retain_samples[:]

dloader_kw = dict(batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_clip, pin_memory=(DEVICE=="cuda"), num_workers=2)
forget_loader = DataLoader(ListDataset(forget_samples), **dloader_kw)
retain_loader = DataLoader(ListDataset(retain_samples), **dloader_kw)
util_loader   = DataLoader(ListDataset(util_samples),   **dloader_kw)

# --------------------
# Embedding helpers & metrics
# --------------------
@torch.no_grad()
def text_embed(texts: List[str]) -> torch.Tensor:
    # Add truncation=True here as well for consistency
    tok = processor.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    tfeat = model.get_text_features(**tok)
    return F.normalize(tfeat, dim=-1)

# Two-class prompts for simple zero-shot classification from queries
TEXT_2CLS = text_embed([f"a photo of a {FORGET_QUERY}", f"a photo of a {RETAIN_QUERY}"])  # [2, D]
TEXT_FORGET = TEXT_2CLS[0:1]

@torch.no_grad()
def compute_two_class_acc(loader: DataLoader, text_embeds: torch.Tensor) -> float:
    tot=0; cor=0
    for batch in loader:
        imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
        labels = torch.tensor(batch.labels, device=DEVICE)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
            logits = LOGIT_SCALE * img @ text_embeds.t()
        preds = logits.argmax(dim=-1)
        cor += (preds==labels).sum().item(); tot += labels.numel()
    return cor/max(1,tot)

@torch.no_grad()
def forget_as_forget_rate(loader: DataLoader, text_embeds: torch.Tensor) -> float:
    tot=0; pred_forget=0
    for batch in loader:
        imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
            logits = LOGIT_SCALE * img @ text_embeds.t()
        preds = logits.argmax(dim=-1)
        pred_forget += (preds==0).sum().item(); tot += preds.numel()
    return pred_forget/max(1,tot)

@torch.no_grad()
def mean_cos_sim(loader: DataLoader, txt: torch.Tensor) -> float:
    vals=[]
    for batch in loader:
        imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
            sim = (img*txt).sum(dim=-1)
        vals.append(sim.mean().item())
    return float(sum(vals)/max(1,len(vals)))

@torch.no_grad()
def batch_caption_retrieval_acc(loader: DataLoader) -> float:
    tot=0; cor=0
    for batch in loader:
        imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
        ids  = batch.input_ids.to(DEVICE, non_blocking=True)
        attn = batch.attention_mask.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
            img = F.normalize(out.image_embeds, dim=-1)
            txt = F.normalize(out.text_embeds,  dim=-1)
            logits = LOGIT_SCALE * img @ txt.t()
        preds = logits.argmax(dim=-1)
        labels = torch.arange(img.size(0), device=img.device)
        cor += (preds==labels).sum().item(); tot += labels.numel()
    return cor/max(1,tot)

@torch.no_grad()
def compute_M4(util_before: float, util_after: float) -> float:
    return 1.0 - abs(util_after - util_before)

# --------------------
# Baselines and mismatch snapshots
# --------------------
BASE_UTIL   = batch_caption_retrieval_acc(util_loader)    # utility proxy on retain class captions
BASE_RETAIN = compute_two_class_acc(retain_loader, TEXT_2CLS)
BASE_FACC   = forget_as_forget_rate(forget_loader, TEXT_2CLS)  # M1 baseline
BASE_FSIM   = mean_cos_sim(forget_loader, TEXT_FORGET)         # M2 baseline
print({
    "forget_query": FORGET_QUERY,
    "retain_query": RETAIN_QUERY,
    "baseline_util": BASE_UTIL,
    "baseline_retain_acc": BASE_RETAIN,
    "baseline_forget_as_forget": BASE_FACC,
    "baseline_forget_sim": BASE_FSIM
})

BASE_BATCH_SIMS = []  # for M3 mismatch-term snapshot
with torch.no_grad():
    for batch in forget_loader:
        imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
        ids  = batch.input_ids.to(DEVICE, non_blocking=True)
        attn = batch.attention_mask.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
            img = F.normalize(out.image_embeds, dim=-1)
            txt = F.normalize(out.text_embeds,  dim=-1)
            base_sim = (img*txt).sum(dim=-1).detach()
        BASE_BATCH_SIMS.append(base_sim)

# --------------------
# LSU Step 1: Prediction Drop Δ_ℓ via block ablation on forget set
# --------------------

def vision_blocks(m: CLIPModel):
    return list(m.vision_model.encoder.layers)

V_BLOCKS = vision_blocks(model)

# Hook to zero only hidden states from a transformer block

def block_zero_output_hook(module, inp, out):
    if isinstance(out, tuple):
        hs = out[0]
        z = torch.zeros_like(hs)
        return (z,) + tuple(out[1:])
    else:
        return torch.zeros_like(out)

@torch.no_grad()
def compute_prediction_drop_forget(block_idx: int, sample_batches: int = SAMPLE_BATCHES_DROP) -> float:
    # Measure average sim(img, class_forget_text) drop when zeroing a block
    sims_base = []
    count = 0
    for batch in forget_loader:
        imgs = batch.pixel_values.to(DEVICE)
        img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
        sims_base.append((img*TEXT_FORGET).sum(dim=-1).mean().item())
        count += 1
        if count>=sample_batches: break
    base = sum(sims_base)/max(1,len(sims_base))

    handle = V_BLOCKS[block_idx].register_forward_hook(block_zero_output_hook)
    sims_ab = []
    count = 0
    for batch in forget_loader:
        imgs = batch.pixel_values.to(DEVICE)
        img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
        sims_ab.append((img*TEXT_FORGET).sum(dim=-1).mean().item())
        count += 1
        if count>=sample_batches: break
    handle.remove()

    ab = sum(sims_ab)/max(1,len(sims_ab))
    return max(0.0, base - ab)

PRED_DROP = [compute_prediction_drop_forget(i) for i in range(len(V_BLOCKS))]
rank_by_drop = sorted(range(len(V_BLOCKS)), key=lambda i: PRED_DROP[i], reverse=True)

SELECTED = rank_by_drop[:K_INIT]
print("Initial selected blocks by Δ:", SELECTED, [f"{PRED_DROP[i]:.4f}" for i in SELECTED])

# --------------------
# LSU Step 2: Gradient-based significance Sig(ℓ)
# --------------------
class ClipContrastiveLoss(nn.Module):
    def __init__(self, logit_scale: float = LOGIT_SCALE):
        super().__init__(); self.logit_scale = logit_scale
    def forward(self, image_embeds, text_embeds):
        img = F.normalize(image_embeds, dim=-1)
        txt = F.normalize(text_embeds, dim=-1)
        logits_per_image = self.logit_scale * img @ txt.t()
        logits_per_text  = logits_per_image.t()
        labels = torch.arange(img.size(0), device=img.device)
        return 0.5*(F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels))

class ClipForgetCosineLoss(nn.Module):
    def __init__(self, margin: float = 0.0):
        super().__init__(); self.cosine = nn.CosineEmbeddingLoss(margin=margin)
    def forward(self, image_embeds, text_embeds):
        img = F.normalize(image_embeds, dim=-1)
        txt = F.normalize(text_embeds, dim=-1)
        targets = -torch.ones(img.size(0), device=img.device)
        return self.cosine(img, txt, targets)

class ClipMismatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, curr_sim: torch.Tensor, base_sim: torch.Tensor):
        return (curr_sim**2).mean() + F.relu(curr_sim - base_sim).pow(2).mean()

contrastive_loss = ClipContrastiveLoss()
forget_loss_fn  = ClipForgetCosineLoss()
mismatch_loss   = ClipMismatchLoss()


def zero_model_grads(m: nn.Module):
    for p in m.parameters():
        if p.grad is not None:
            p.grad.zero_()

def gather_block_grad_norms(blocks):
    norms = []
    for blk in blocks:
        vec = []
        for p in blk.parameters():
            if p.grad is not None:
                vec.append(p.grad.detach().flatten())
        norms.append(torch.norm(torch.cat(vec)) if len(vec)>0 else torch.tensor(0.0, device=DEVICE))
    return norms

# grads for L_forget at θ^(0)
model.train(); zero_model_grads(model)
fb = 0
for batch in forget_loader:
    imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
    ids  = batch.input_ids.to(DEVICE, non_blocking=True)
    attn = batch.attention_mask.to(DEVICE, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
        lf   = forget_loss_fn(out.image_embeds, out.text_embeds)
    lf.backward()
    fb += 1
    if fb >= BATCHES_FOR_SIG: break
G_FORGET = gather_block_grad_norms(V_BLOCKS)

# grads for L_retain at θ^(0)
zero_model_grads(model)
rb = 0
for batch in retain_loader:
    imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
    ids  = batch.input_ids.to(DEVICE, non_blocking=True)
    attn = batch.attention_mask.to(DEVICE, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
        lr   = contrastive_loss(out.image_embeds, out.text_embeds)
    lr.backward()
    rb += 1
    if rb >= BATCHES_FOR_SIG: break
G_RETAIN = gather_block_grad_norms(V_BLOCKS)
model.eval()

SIG = []
for i in range(len(V_BLOCKS)):
    I  = float(G_FORGET[i].item())
    Ir = float(G_RETAIN[i].item())
    SIG.append(I / (Ir + EPS))

# --------------------
# LSU Step 3: Iterative expansion + targeted updates
# --------------------
ORIG_STATES = {i: copy.deepcopy(V_BLOCKS[i].state_dict()) for i in range(len(V_BLOCKS))}

@torch.no_grad()
def apply_selected_step(step_size: float):
    for bi in SELECTED:
        for p in V_BLOCKS[bi].parameters():
            if p.grad is not None:
                p.add_(-step_size * p.grad)

@torch.no_grad()
def eval_all():
    util   = batch_caption_retrieval_acc(util_loader)           # utility proxy on retain class
    retain = compute_two_class_acc(retain_loader, TEXT_2CLS)    # two-class accuracy on retain class
    f_as_f = forget_as_forget_rate(forget_loader, TEXT_2CLS)    # M1
    f_sim  = mean_cos_sim(forget_loader, TEXT_FORGET)           # M2
    return {
        "util_retrieval": util,
        "retain_acc": retain,
        "M1_forget_as_forget": f_as_f,
        "M2_forget_sim": f_sim,
        "M4": compute_M4(BASE_UTIL, util)
    }

best_state = {i: copy.deepcopy(V_BLOCKS[i].state_dict()) for i in range(len(V_BLOCKS))}
best_metrics = {"util_retrieval": -1, "retain_acc": -1, "M1_forget_as_forget": 1e9, "M2_forget_sim": 1e9, "M3_priv_leak": 1e9, "M4": -1}

# Precompute baseline per-batch sims for M3
BASE_FLAT = torch.cat([b.to(DEVICE) for b in BASE_BATCH_SIMS]) if len(BASE_BATCH_SIMS)>0 else torch.zeros(1, device=DEVICE)

def set_requires_grad_for_blocks(blocks, keep_idxs):
    keep = set(keep_idxs)
    for i, blk in enumerate(blocks):
        req = (i in keep)
        for p in blk.parameters():
            p.requires_grad = req

for round_id in range(MAX_ROUNDS):
    model.train(); zero_model_grads(model)
    set_requires_grad_for_blocks(V_BLOCKS, SELECTED)

    # Forget + mismatch on forget batches
    fb = 0
    m3_terms = []
    for batch in forget_loader:
        imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
        ids  = batch.input_ids.to(DEVICE, non_blocking=True)
        attn = batch.attention_mask.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
            img_n = F.normalize(out.image_embeds, dim=-1)
            txt_n = F.normalize(out.text_embeds,  dim=-1)
            curr_sim = (img_n*txt_n).sum(dim=-1)
            base_sim = BASE_BATCH_SIMS[fb].to(DEVICE)
            lf = forget_loss_fn(out.image_embeds, out.text_embeds)
            lm = mismatch_loss(curr_sim, base_sim)
            loss = ETA_FORGET*lf + ETA_MISMATCH*lm
        loss.backward()
        m3_terms.append((curr_sim - base_sim).pow(2).mean().detach())
        fb += 1
        if fb >= BATCHES_FOR_FORGET: break

    # Retain (subtract gradient)
    rb = 0
    for batch in retain_loader:
        imgs = batch.pixel_values.to(DEVICE, non_blocking=True)
        ids  = batch.input_ids.to(DEVICE, non_blocking=True)
        attn = batch.attention_mask.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
            lr = contrastive_loss(out.image_embeds, out.text_embeds)
        (-ETA_RETAIN * lr).backward()
        rb += 1
        if rb >= BATCHES_FOR_RETAIN: break

    model.eval(); apply_selected_step(STEP_SIZE)

    metrics = eval_all()
    # M3 (PrivLeak → 0): mean squared change of forget sims vs baseline (aggregated from the last forward passes)
    metrics["M3_priv_leak"] = float(torch.stack(m3_terms).mean().item()) if len(m3_terms)>0 else 0.0
    print(f"Round {round_id}: S={SELECTED} -> {metrics}")

    retain_ok = metrics["retain_acc"] >= (BASE_RETAIN - RETAIN_TOL)
    score = (1.0 - metrics["M1_forget_as_forget"]) + metrics["M4"] + (0.5 if retain_ok else 0.0)
    best_score = (1.0 - best_metrics["M1_forget_as_forget"]) + best_metrics["M4"] + (0.5 if (best_metrics["retain_acc"]>=0 and best_metrics["retain_acc"] >= (BASE_RETAIN - RETAIN_TOL)) else 0.0)
    if best_metrics["util_retrieval"] < 0 or score > best_score:
        best_metrics = metrics
        best_state = {i: copy.deepcopy(V_BLOCKS[i].state_dict()) for i in range(len(V_BLOCKS))}

    if (metrics["M1_forget_as_forget"] <= FORGET_TAU) and retain_ok:
        print("Stopping: forgetting threshold reached with retain preserved.")
        break

    # Expand by top k' remaining by SIG
    remaining = [i for i in range(len(V_BLOCKS)) if i not in SELECTED]
    if len(remaining) == 0:
        print("No more layers to add.")
        break
    rem_sorted = sorted(remaining, key=lambda i: SIG[i], reverse=True)
    to_add = rem_sorted[:min(K_PRIME, len(rem_sorted))]
    SELECTED.extend(to_add)
    print("Expanded S with:", to_add, "; new S=", SELECTED)

# Restore best state and report
for i in range(len(V_BLOCKS)):
    V_BLOCKS[i].load_state_dict(best_state[i])

final = eval_all()
final["M3_priv_leak"] = best_metrics.get("M3_priv_leak", 0.0)

summary = {
    "queries": {"forget": FORGET_QUERY, "retain": RETAIN_QUERY},
    "k_init": K_INIT,
    "k_prime": K_PRIME,
    "rounds_ran": round_id+1,
    "selected_layers_final": SELECTED,
    "baseline": {
        "util_retrieval": BASE_UTIL,
        "retain_acc": BASE_RETAIN,
        "M1_forget_as_forget": BASE_FACC,
        "M2_forget_sim": BASE_FSIM
    },
    "final": final,
    "best_metrics_seen": best_metrics
}
print("==== LSU-CLIP LAION (two caption categories) Summary ====")
print(json.dumps(summary, indent=2))

# Optional: save edited state
# torch.save(model.state_dict(), "/content/clip_lsu_laion_two_cats.pt")