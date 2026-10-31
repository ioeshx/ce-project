# LSU-Style Layer-Selective Unlearning (LSU) for CLIP — Full Script (All-in-One)
# ---------------------------------------------------------------------------------
# This single cell implements LSU-style iterative layer selection + targeted unlearning
# for CLIP (openai/clip-vit-base-patch32) using small CIFAR-10 splits (≤100 items).
#
# Pipeline:
# 1) Data: build forget (Df) and retain (Dr) splits from CIFAR-10.
# 2) Baseline eval & snapshot sims for mismatch loss.
# 3) LSU Layer Selection:
#    (a) Prediction-Drop Δ_ℓ via block ablation on Df (init top-k).
#    (b) Gradient-Based Significance Sig(ℓ) = ||∇_ℓ L_forget|| / (||∇_ℓ L_retain|| + ε).
#    (c) Iterative expansion: add k′ layers by Sig(ℓ), update only selected layers.
# 4) Unlearning loss on selected layers only:  L = ηF * L_forget + ηM * L_mismatch − ηR * L_retain.
# 5) Evaluate after each round: ForgetAcc (↓), RetainAcc (≈), Utility M4 (↑), ΔSim(↓).
#
# Tweak at top: K_INIT, K_PRIME, MAX_ROUNDS, STEP_SIZE, ETA_* , FORGET_TAU, RETAIN_TOL.

# !pip -q install "torch>=2.2" "transformers>=4.44" "datasets>=2.20" "torchvision>=0.17" "accelerate>=0.30" "numpy>=1.24" "tqdm>=4.66"

import os, math, copy, random, json
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as tvds, transforms as T
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# --------------------
# Config
# --------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
FORGET_CLASS   = "cat"
RETAIN_CLASSES = [c for c in CIFAR10_CLASSES if c != FORGET_CLASS]

N_FORGET   = 80
N_RETAIN   = 100
BATCH_SIZE = 16

# LSU knobs
K_INIT     = 1      # initial top-k by prediction drop
K_PRIME    = 1      # layers to add per round by significance
MAX_ROUNDS = 3
STEP_SIZE  = 1e-3   # gradient step on selected layers
ETA_FORGET   = 1.0
ETA_MISMATCH = 0.5
ETA_RETAIN   = 1.0
FORGET_TAU = 0.10   # stop when ForgetAcc ≤ τ
RETAIN_TOL = 0.02   # allow ≤2% absolute drop in retain accuracy

LOGIT_SCALE = 1/0.07
EPS = 1e-8

# --------------------
# Utils & Data
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
    labels: List[int]


def build_cifar10_splits(forget_class: str, n_forget: int, n_retain: int):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    ds = tvds.CIFAR10(root="/content/data", train=True, download=True, transform=transform)
    class_to_idx = {name:i for i,name in enumerate(CIFAR10_CLASSES)}
    fidx = class_to_idx[forget_class]

    f_indices = [i for i, (_, y) in enumerate(ds) if y == fidx][:n_forget]

    r_indices = []
    per = max(1, n_retain // len(RETAIN_CLASSES))
    for cname in RETAIN_CLASSES:
        cid = class_to_idx[cname]
        cls = [i for i, (_, y) in enumerate(ds) if y == cid][:per]
        r_indices.extend(cls)
    r_indices = r_indices[:n_retain]

    return Subset(ds, f_indices), Subset(ds, r_indices), ds


def make_text_prompts(processor: CLIPProcessor, labels: List[int]):
    texts = [f"a photo of a {CIFAR10_CLASSES[y]}" for y in labels]
    tok = processor.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return tok["input_ids"], tok["attention_mask"]


def collate_clip(processor: CLIPProcessor, batch):
    imgs, labels = zip(*batch)
    pixel_values = torch.stack(list(imgs), dim=0)
    input_ids, attention_mask = make_text_prompts(processor, list(labels))
    return Batch(pixel_values, input_ids, attention_mask, list(labels))

# Losses
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
    # Penalize current forget-similarity being close to prior baseline similarity
    def __init__(self):
        super().__init__()
    def forward(self, curr_sim: torch.Tensor, base_sim: torch.Tensor):
        return (curr_sim**2).mean() + F.relu(curr_sim - base_sim).pow(2).mean()

# --------------------
# Model & DataLoaders
# --------------------
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

forget_ds, retain_ds, full_ds = build_cifar10_splits(FORGET_CLASS, N_FORGET, N_RETAIN)
forget_loader = DataLoader(forget_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_clip(processor, b))
retain_loader = DataLoader(retain_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_clip(processor, b))
util_ds = Subset(full_ds, list(range(300, 300+200)))
util_loader = DataLoader(util_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_clip(processor, b))

# --------------------
# Embedding helpers & metrics
# --------------------
@torch.no_grad()
def get_text_embeddings(class_names: List[str]):
    texts = [f"a photo of a {c}" for c in class_names]
    tok = processor.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    tfeat = model.get_text_features(**tok)
    return F.normalize(tfeat, dim=-1)

TEXT_ALL = get_text_embeddings(CIFAR10_CLASSES)
TEXT_FORGET = get_text_embeddings([FORGET_CLASS])[0:1]

@torch.no_grad()
def compute_zero_shot_acc(loader: DataLoader, text_embeds: torch.Tensor) -> float:
    tot=0; cor=0
    for batch in loader:
        imgs = batch.pixel_values.to(DEVICE)
        labels = torch.tensor(batch.labels, device=DEVICE)
        img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
        logits = LOGIT_SCALE * img @ text_embeds.t()
        preds = logits.argmax(dim=-1)
        cor += (preds==labels).sum().item(); tot += labels.numel()
    return cor/max(1,tot)

@torch.no_grad()
def forget_accuracy(loader: DataLoader, forget_class: str, text_all: torch.Tensor) -> float:
    class_to_idx = {c:i for i,c in enumerate(CIFAR10_CLASSES)}
    fidx = class_to_idx[forget_class]
    tot=0; f_as_pred=0
    for batch in loader:
        imgs = batch.pixel_values.to(DEVICE)
        labels = torch.tensor(batch.labels, device=DEVICE)
        img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
        logits = LOGIT_SCALE * img @ text_all.t()
        preds = logits.argmax(dim=-1)
        f_as_pred += (preds==fidx).sum().item(); tot += labels.numel()
    return f_as_pred/max(1,tot)

@torch.no_grad()
def mean_cos_sim(loader: DataLoader, txt: torch.Tensor) -> float:
    vals=[]
    for batch in loader:
        imgs = batch.pixel_values.to(DEVICE)
        img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
        sim = (img*txt).sum(dim=-1)
        vals.append(sim.mean().item())
    return float(sum(vals)/max(1,len(vals)))

@torch.no_grad()
def compute_M4(util_before: float, util_after: float) -> float:
    return 1.0 - abs(util_after - util_before)

# --------------------
# Baseline metrics + snapshot sims for mismatch
# --------------------
BASE_UTIL   = compute_zero_shot_acc(util_loader, TEXT_ALL)
BASE_RETAIN = compute_zero_shot_acc(retain_loader, TEXT_ALL)
BASE_FACC   = forget_accuracy(forget_loader, FORGET_CLASS, TEXT_ALL)
BASE_FSIM   = mean_cos_sim(forget_loader, TEXT_FORGET)
print({"baseline_util":BASE_UTIL, "baseline_retain":BASE_RETAIN, "baseline_forget_acc":BASE_FACC, "baseline_forget_sim":BASE_FSIM})

BASE_BATCH_SIMS = []
with torch.no_grad():
    for batch in forget_loader:
        imgs = batch.pixel_values.to(DEVICE)
        img = F.normalize(model.get_image_features(pixel_values=imgs), dim=-1)
        base_sim = (img*TEXT_FORGET).sum(dim=-1).detach()
        BASE_BATCH_SIMS.append(base_sim)

# --------------------
# LSU Step 1: Prediction Drop Δ_ℓ via block ablation on forget set
# --------------------

def vision_blocks(m: CLIPModel):
    return list(m.vision_model.encoder.layers)

V_BLOCKS = vision_blocks(model)

def block_zero_output_hook(module, inp, out):
    # Some transformer blocks return tuples (hidden_states, ...).
    # Ensure we zero ONLY the hidden_states while preserving structure.
    if isinstance(out, tuple):
        hs = out[0]
        z = torch.zeros_like(hs)
        return (z,) + tuple(out[1:])
    else:
        return torch.zeros_like(out)

@torch.no_grad()
def compute_prediction_drop_forget(block_idx: int, sample_batches: int = 3) -> float:
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
model.train()
zero_model_grads(model)
for batch in forget_loader:
    imgs = batch.pixel_values.to(DEVICE)
    ids  = batch.input_ids.to(DEVICE)
    attn = batch.attention_mask.to(DEVICE)
    out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
    lf   = forget_loss_fn(out.image_embeds, out.text_embeds)
    lf.backward()
G_FORGET = gather_block_grad_norms(V_BLOCKS)

# grads for L_retain at θ^(0)
zero_model_grads(model)
for batch in retain_loader:
    imgs = batch.pixel_values.to(DEVICE)
    ids  = batch.input_ids.to(DEVICE)
    attn = batch.attention_mask.to(DEVICE)
    out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
    lr   = contrastive_loss(out.image_embeds, out.text_embeds)
    lr.backward()
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
    util   = compute_zero_shot_acc(util_loader, TEXT_ALL)
    retain = compute_zero_shot_acc(retain_loader, TEXT_ALL)
    f_acc  = forget_accuracy(forget_loader, FORGET_CLASS, TEXT_ALL)
    f_sim  = mean_cos_sim(forget_loader, TEXT_FORGET)
    return {
        "util_acc": util,
        "retain_acc": retain,
        "forget_acc": f_acc,
        "forget_sim": f_sim,
        "M4": compute_M4(BASE_UTIL, util)
    }

best_state = {i: copy.deepcopy(V_BLOCKS[i].state_dict()) for i in range(len(V_BLOCKS))}
best_metrics = {"util_acc": -1, "retain_acc": -1, "forget_acc": 1e9, "forget_sim": 1e9, "M4": -1}

for round_id in range(MAX_ROUNDS):
    # Build combined loss on small subsets; update only selected layers
    model.train()
    zero_model_grads(model)

    # Forget + mismatch on a few forget batches
    fb = 0
    for batch in forget_loader:
        imgs = batch.pixel_values.to(DEVICE)
        ids  = batch.input_ids.to(DEVICE)
        attn = batch.attention_mask.to(DEVICE)
        out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
        curr_sim = (F.normalize(out.image_embeds, dim=-1) * F.normalize(out.text_embeds, dim=-1)).sum(dim=-1)
        base_sim = BASE_BATCH_SIMS[fb].to(DEVICE)
        lf = forget_loss_fn(out.image_embeds, out.text_embeds)
        lm = mismatch_loss(curr_sim, base_sim)
        (ETA_FORGET*lf + ETA_MISMATCH*lm).backward()
        fb += 1
        if fb >= 3: break

    # zero grads for unselected
    for bi in range(len(V_BLOCKS)):
        if bi not in SELECTED:
            for p in V_BLOCKS[bi].parameters():
                if p.grad is not None:
                    p.grad.zero_()

    # Retain (subtract gradient)
    rb = 0
    for batch in retain_loader:
        imgs = batch.pixel_values.to(DEVICE)
        ids  = batch.input_ids.to(DEVICE)
        attn = batch.attention_mask.to(DEVICE)
        out  = model(pixel_values=imgs, input_ids=ids, attention_mask=attn)
        lr = contrastive_loss(out.image_embeds, out.text_embeds)
        (-ETA_RETAIN * lr).backward()
        rb += 1
        if rb >= 3: break

    # zero grads for unselected again
    for bi in range(len(V_BLOCKS)):
        if bi not in SELECTED:
            for p in V_BLOCKS[bi].parameters():
                if p.grad is not None:
                    p.grad.zero_()

    model.eval()
    apply_selected_step(STEP_SIZE)

    metrics = eval_all()
    print(f"Round {round_id}: S={SELECTED} -> {metrics}")

    retain_ok = metrics["retain_acc"] >= (BASE_RETAIN - RETAIN_TOL)
    score = (1.0 - metrics["forget_acc"]) + metrics["M4"] + (0.5 if retain_ok else 0.0)
    best_score = (1.0 - best_metrics["forget_acc"]) + best_metrics["M4"] + (0.5 if (best_metrics["retain_acc"]>=0 and best_metrics["retain_acc"] >= (BASE_RETAIN - RETAIN_TOL)) else 0.0)
    if best_metrics["util_acc"] < 0 or score > best_score:
        best_metrics = metrics
        best_state = {i: copy.deepcopy(V_BLOCKS[i].state_dict()) for i in range(len(V_BLOCKS))}

    if (metrics["forget_acc"] <= FORGET_TAU) and retain_ok:
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

summary = {
    "k_init": K_INIT,
    "k_prime": K_PRIME,
    "rounds_ran": round_id+1,
    "selected_layers_final": SELECTED,
    "baseline": {"util": BASE_UTIL, "retain": BASE_RETAIN, "forget": BASE_FACC, "fsim": BASE_FSIM},
    "final": final,
    "best_metrics_seen": best_metrics
}
print("==== LSU-CLIP Summary ====")
print(json.dumps(summary, indent=2))

# Optional: save edited state
# torch.save(model.state_dict(), "/content/clip_lsu_unlearned.pt")