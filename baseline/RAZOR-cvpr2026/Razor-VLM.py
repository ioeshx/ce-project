
# RAZOR for Llava 1.6 
# ===== 0) Install and Imports =====
!pip -q uninstall -y pyarrow || true
!pip -q install "pyarrow>=14,<20" "numpy<2"
!pip -q install --upgrade --upgrade-strategy eager transformers==4.44.2 accelerate==0.33.0 bitsandbytes datasets==2.19.1 peft timm sentencepiece torchvision pillow safetensors

try:
    import xformers  # noqa: F401
except Exception:
    try:
        !pip -q install xformers==0.0.27.post2
    except Exception as e:
        print("[WARN] xformers install failed — continuing without it.")

import os, math, random, json, gc, io, sys, time
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== 1) Basic setup =====
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
MODEL_ID = "liuhaotian/llava-v1.5-7b"

# ===== 2) Control Params =====
FORGET_NAME = "Elon Musk"
MAX_FORGET, MAX_RETAIN, VAL_FORGET, VAL_RETAIN = 32, 64, 24, 48
BATCH, GRAD_ACCUM, MAX_ITERS, LR, STEPS_PER_ITER = 1, 1, 6, 1e-4, 25
TARGET_ANSWER = "unknown person"
W_FORGET, W_MISMATCH, W_RETAIN = 1.0, 0.3, 0.7
TARGET_FA, MAX_RETAIN_PPL_DEGR = 0.10, 0.15

# ===== 3) Load Model =====
print("Loading LLaVA-v1.5-7B in 4-bit…")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE)
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", quantization_config=bnb_config, torch_dtype=DTYPE)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ===== 4) Identify Vision Layers =====
VISION_MODULE_NAMES = []
for n, m in model.named_modules():
    if any(k in n for k in ["vision_tower", "multi_modal_projector", "visual", "vision_model", "mm_projector"]):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            VISION_MODULE_NAMES.append(n)
if not VISION_MODULE_NAMES:
    for n, m in model.named_modules():
        if "vision" in n and isinstance(m, nn.Linear):
            VISION_MODULE_NAMES.append(n)
print(f"[Vision Layers] {len(VISION_MODULE_NAMES)} found, sample: {VISION_MODULE_NAMES[:5]}")

# ===== 5) Load Dataset =====
print("Loading LAION subset…")
try:
    ds = load_dataset("jp1924/Laion400m-1", split="train[:4000]")
except Exception:
    ds = load_dataset("hf-internal-testing/fixtures_image_utils", split="train")

import requests
requests.packages.urllib3.disable_warnings()

def fetch_image(url):
    try:
        r = requests.get(url, timeout=5, verify=False)
        if r.status_code == 2000:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

forget_records, retain_records = [], []
for ex in ds:
    cap = ex.get("TEXT") or ex.get("caption") or ""
    url = ex.get("URL") or ex.get("url") or None
    if not cap or not url: continue
    img = fetch_image(url)
    if img is None: continue
    if FORGET_NAME.lower() in cap.lower():
        if len(forget_records) < MAX_FORGET + VAL_FORGET:
            forget_records.append({"image": img, "caption": cap})
    else:
        if len(retain_records) < MAX_RETAIN + VAL_RETAIN:
            retain_records.append({"image": img, "caption": cap})
    if len(forget_records) >= MAX_FORGET + VAL_FORGET and len(retain_records) >= MAX_RETAIN + VAL_RETAIN:
        break

forget_train, forget_val = forget_records[:MAX_FORGET], forget_records[MAX_FORGET:]
retain_train, retain_val = retain_records[:MAX_RETAIN], retain_records[MAX_RETAIN:]

# ===== 6) Dataset Wrappers =====
QUESTION, CAPTION_PROMPT = "What's the name of the person in this image?", "Describe this image briefly."
@dataclass
class Sample: image: Image.Image; prompt: str; target: Optional[str]
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, recs, mode): self.recs, self.mode = recs, mode
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]; img = r["image"]
        if self.mode == "forget": return Sample(img, QUESTION, TARGET_ANSWER)
        if self.mode == "forget_eval": return Sample(img, QUESTION, FORGET_NAME)
        if self.mode == "retain": return Sample(img, CAPTION_PROMPT, None)
        return Sample(img, CAPTION_PROMPT, None)

def collate(batch):
    imgs, txts = [b.image for b in batch], [b.prompt for b in batch]
    proc = processor(images=imgs, text=txts, return_tensors="pt").to(DEVICE, dtype=DTYPE)
    return proc, [b.target for b in batch]

forget_ds, retain_ds = VQADataset(forget_train, "forget"), VQADataset(retain_train, "retain")
forget_val_ds, retain_val_ds = VQADataset(forget_val, "forget_eval"), VQADataset(retain_val, "retain_eval")

# ===== 7) Loss and Metric Helpers =====
@torch.no_grad()
def generate_answer(imgs, q):
    inp = processor(images=imgs, text=[q]*len(imgs), return_tensors="pt").to(DEVICE, dtype=DTYPE)
    out = model.generate(**inp, max_new_tokens=32)
    text = processor.batch_decode(out, skip_special_tokens=True)
    return [t.split(q)[-1].strip() for t in text]

def lm_loss(imgs, prm, tgt=None):
    if tgt is None:
        with torch.no_grad(): tgt = [generate_answer([i], p)[0] for i, p in zip(imgs, prm)]
    texts = [f"{p}\n{t}" for p, t in zip(prm, tgt)]
    b = processor(images=imgs, text=texts, return_tensors="pt").to(DEVICE, dtype=DTYPE)
    out = model(**b, labels=b["input_ids"])
    return out.loss

def mismatch_loss(imgs, txts):
    b = processor(images=imgs, text=txts, return_tensors="pt").to(DEVICE, dtype=DTYPE)
    out = model(**b, output_hidden_states=True)
    h = out.hidden_states[-1]
    imgf, txtf = h[:, :h.shape[1]//2].mean(1), h[:, h.shape[1]//2:].mean(1)
    imgf, txtf = F.normalize(imgf, dim=-1), F.normalize(txtf, dim=-1)
    return (1 - (imgf*txtf).sum(-1)).mean()

# ===== 8) Gradient Ratio Selection =====
def get_grad_norms(data, prompt, target=None):
    model.eval()
    for n, p in model.named_parameters():
        if p.grad is not None: p.grad.zero_()
    imgs = [d["image"] for d in data[:BATCH]]
    loss = lm_loss(imgs, [prompt]*len(imgs), [target]*len(imgs) if target else None)
    loss.backward()
    norms = {}
    for n, m in model.named_modules():
        if n in VISION_MODULE_NAMES:
            g2 = sum((p.grad.detach().float()**2).sum().item() for p in m.parameters() if p.grad is not None)
            norms[n] = math.sqrt(g2) if g2>0 else 0
    return norms

forget_norms = get_grad_norms(forget_train, QUESTION, TARGET_ANSWER)
retain_norms = get_grad_norms(retain_train, CAPTION_PROMPT)
ratios = {k: forget_norms.get(k,0)/(retain_norms.get(k,1e-8)) for k in VISION_MODULE_NAMES}
vals = torch.tensor(list(ratios.values()))
th = 0.5*(vals.mean()+vals.std())
selected = [k for k,v in ratios.items() if v>=th]
print(f"[RAZOR] Selected {len(selected)} layers above threshold {th:.4f}")

# ===== 9) Attach LoRA =====
lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=list(set([n.split(".")[-1] for n in selected])), task_type="CAUSAL_LM")
model = get_peft_model(model, lora_cfg)
optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

# ===== 10) Training Loop =====
def forget_acc(ds, name):
    c=0
    for i in range(0,len(ds),4):
        ims=[r["image"] for r in ds.recs[i:i+4]]; outs=generate_answer(ims,QUESTION)
        c+=sum(name.lower() in o.lower() for o in outs)
    return c/len(ds)

def retain_ppl(ds):
    L=[lm_loss([r["image"]],[CAPTION_PROMPT]) for r in ds.recs[:8]]
    return math.exp(sum(float(l.detach().cpu()) for l in L)/len(L))

base_FA, base_PPL = forget_acc(forget_val_ds, FORGET_NAME), retain_ppl(retain_val_ds)

for it in range(MAX_ITERS):
    print(f"\n[Iter {it+1}/{MAX_ITERS}] Training {len(selected)} layers…")
    model.train()
    for step in range(STEPS_PER_ITER):
        imgs=[forget_train[random.randrange(len(forget_train))]["image"]]
        Lf=lm_loss(imgs,[QUESTION],[TARGET_ANSWER])
        Lm=mismatch_loss(imgs,[FORGET_NAME])
        imgs_r=[retain_train[random.randrange(len(retain_train))]["image"]]
        Lr=lm_loss(imgs_r,[CAPTION_PROMPT])
        loss=W_FORGET*Lf+W_MISMATCH*Lm+W_RETAIN*Lr
        optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step()
        if step%5==0: print(f" step{step}: Lf={Lf:.3f} Lm={Lm:.3f} Lr={Lr:.3f}")
    FA, PPL = forget_acc(forget_val_ds, FORGET_NAME), retain_ppl(retain_val_ds)
    print(f"[Eval] FA={FA:.3f}, Retain-PPL={PPL:.2f}")
    if FA<=TARGET_FA and PPL<=base_PPL*(1+MAX_RETAIN_PPL_DEGR): break

# ===== 11) Summary =====
print(f"\nFinal Forget Accuracy: {FA:.3f}, Retain PPL: {PPL:.2f}")
SAVE_DIR="/content/razor_llava15_lora"
os.makedirs(SAVE_DIR,exist_ok=True)
model.save_pretrained(SAVE_DIR)
print(f"Saved LoRA adapters to {SAVE_DIR}")