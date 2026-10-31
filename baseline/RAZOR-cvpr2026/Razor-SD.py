# RAZOR for Stable Diffusion 3 Medium — **STYLE UNLEARNING** 

# =====================
# 0) Installs & Imports
# =====================
# !pip -q install --upgrade diffusers transformers accelerate datasets ftfy open_clip_torch sentencepiece einops safetensors torchvision

import os, io, math, json, time, random, functools, tempfile, itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np

import open_clip
from open_clip import create_model_and_transforms, get_tokenizer as get_clip_tokenizer

from diffusers import StableDiffusion3Pipeline
from diffusers.utils import make_image_grid

# Reproducibility
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =====================
# 1) Config — STYLE targets
# =====================
@dataclass
class CFG:
    # Model
    sd3_id: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    hf_token_env: str = "HF_TOKEN"

    # Style Unlearning targets
    content_object: str = "truck"          # neutral content to render styles on
    forget_style: str = "pixel art"        # style to forget
    retain_style: str = "oil painting"     # style to retain
    neutral_style: str = "photorealistic"  # for CRA proxy

    # RAZOR selection/training
    max_selected_layers: int = 4
    importance_threshold_rule: str = "avg_mean_std/2"
    grad_chunk: int = 2              # tiny batch for one-time grads (prompts per style)

    # Loss weights
    w_GA: float = 1.0  # push apart (forget)
    w_GD: float = 1.0  # pull together (retain)
    w_MM: float = 0.5  # mismatch

    # Generation
    gen_steps: int = 8
    cfg_scale: float = 3.0
    images_per_prompt: int = 1
    eval_per_style_prompts: int = 2
    negative_prompt: str = ""

    # CLIP verifier
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # Iteration target (SINGLE TARGET)
    TARGET_UA: float = 0.70

    # Misc
    outdir: str = "./razor_sd3_out"

cfg = CFG()
os.makedirs(cfg.outdir, exist_ok=True)

# =====================
# 2) Prompt banks (style-conditioned)
# =====================
_STYLE_TEMPLATES = [
    "a {obj} in {style} style",
    "a detailed {style} illustration of a {obj}",
    "{style} depiction of a {obj}",
    "a {obj}, {style} aesthetic",
    "{style} artwork of a {obj}",
    "studio shot, {style} of a {obj}",
    "cinematic {style} image of a {obj}",
]

_DEFENSE_TEMPLATES = [  # neutral/alt prompts for CRA
    "a {obj}, {style}",
    "high quality {style} photo of a {obj}",
]

def make_style_prompts(obj: str, style: str, k: int, neutral: bool=False) -> List[str]:
    bank = list(_DEFENSE_TEMPLATES if neutral else _STYLE_TEMPLATES)
    random.shuffle(bank)
    return [t.format(obj=obj, style=style) for t in bank[:k]]

# =====================
# 3) OpenCLIP verifier (float32 throughout)
# =====================
clip_model, _, clip_preproc = create_model_and_transforms(cfg.clip_model_name, pretrained=cfg.clip_pretrained, device=DEVICE)
clip_model.eval()
clip_tokenizer = get_clip_tokenizer(cfg.clip_model_name)
OPENCLIP_DIM = clip_model.text_projection.shape[1] if hasattr(clip_model, 'text_projection') else 512

@torch.no_grad()
def clip_text_emb(texts: List[str]) -> torch.Tensor:
    toks = clip_tokenizer(texts).to(DEVICE)
    z = clip_model.encode_text(toks).float()
    return z / z.norm(dim=-1, keepdim=True)

@torch.no_grad()
def clip_image_emb(images: List[Image.Image]) -> torch.Tensor:
    batch = torch.stack([clip_preproc(im).to(DEVICE) for im in images])
    z = clip_model.encode_image(batch).float()
    return z / z.norm(dim=-1, keepdim=True)

@torch.no_grad()
def clip_presence_score(images: List[Image.Image], label_texts: List[str]) -> float:
    if len(images) == 0:
        return 0.0
    t = clip_text_emb(label_texts)
    v = clip_image_emb(images)
    logits = v @ t.T
    top = logits.argmax(dim=1).tolist()
    hits = sum(1 for i in top if (len(label_texts)==1 and i==0) or (0 <= i < len(label_texts)))
    return hits / len(images)

# =====================
# 4) SD3-Medium pipeline
# =====================
print("[Model] Loading SD3-Medium pipeline…")
hf_token = os.environ.get(cfg.hf_token_env, None)
pipe = StableDiffusion3Pipeline.from_pretrained(
    cfg.sd3_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True,
    token=hf_token,
)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)

# xFormers fallback
try:
    if torch.cuda.is_available():
        pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe.enable_model_cpu_offload()
except Exception:
    print("[Info] xFormers not available — continuing without it.")
    if not torch.cuda.is_available():
        pipe.enable_model_cpu_offload()

# Prefer CLIP text encoder for edits
clip_enc = getattr(pipe, "text_encoder", None) or getattr(pipe, "text_encoder_2", None)
clip_tok = getattr(pipe, "tokenizer", None) or getattr(pipe, "tokenizer_2", None)
if clip_enc is None or clip_tok is None:
    print("[Warn] No CLIP text encoder/tokenizer found; edits fall back to OpenCLIP-only embedding deltas.")

# Candidate layers in the text encoder
razor_layers: List[Tuple[str, nn.Module]] = []
if clip_enc is not None:
    for name, module in clip_enc.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            razor_layers.append((name, module))
if len(razor_layers) == 0:
    print("[Warn] No suitable layers found in CLIP text encoder; selection will be empty.")

# =====================
# 5) RAZOR helpers — dim adapter (e.g., 768→512), cosine loss
# =====================
TEXT_TO_IMG_ADAPTER: Optional[nn.Linear] = None

def ensure_txt_adapter(txt_dim: int, img_dim: int) -> nn.Linear:
    global TEXT_TO_IMG_ADAPTER
    if (TEXT_TO_IMG_ADAPTER is None) or (TEXT_TO_IMG_ADAPTER.in_features != txt_dim) or (TEXT_TO_IMG_ADAPTER.out_features != img_dim):
        layer = nn.Linear(txt_dim, img_dim, bias=False).to(DEVICE)
        with torch.no_grad():
            nn.init.xavier_uniform_(layer.weight)
        for p in layer.parameters():
            p.requires_grad_(False)
        TEXT_TO_IMG_ADAPTER = layer
    return TEXT_TO_IMG_ADAPTER


def encode_text_emb(prompts: List[str]) -> torch.Tensor:
    if clip_tok is None or clip_enc is None:
        return clip_text_emb(prompts)  # already 512-d
    inputs = clip_tok(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    out = clip_enc(**inputs)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        z = out.pooler_output
    elif hasattr(out, "last_hidden_state"):
        z = out.last_hidden_state.mean(dim=1)
    else:
        raise RuntimeError("Unexpected text encoder output structure.")
    z = z.float()
    # project if needed
    img_dim = OPENCLIP_DIM
    if z.size(-1) != img_dim:
        adapter = ensure_txt_adapter(z.size(-1), img_dim)
        z = adapter(z)
    return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

@torch.no_grad()
def encode_image_emb(imgs: List[Image.Image]) -> torch.Tensor:
    return clip_image_emb(imgs)  # 512-d normalized


def cosine_sim(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a*b).sum(dim=-1)

class RazorTextLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, txt_emb: torch.Tensor, img_emb: torch.Tensor, mode: str):
        s = cosine_sim(txt_emb, img_emb)
        if mode == "GA":     # push apart
            return (+1.0) * s.mean() * (-1.0)
        if mode == "GD":     # pull together
            return (-1.0) * s.mean() * (-1.0)
        if mode == "MM":     # mismatch shuffle
            B = img_emb.size(0)
            perm = torch.randperm(B, device=img_emb.device)
            s_mis = cosine_sim(txt_emb, img_emb[perm])
            return (+1.0) * s_mis.mean() * (-1.0)
        raise ValueError(mode)

razor_loss = RazorTextLoss()

def _to_scalar(x: torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.item())
        return float(x.detach().float().mean().item())
    return float(x)

# =====================
# 6) One-time grads for layer scoring (use generated style images)
# =====================

def generate_images_for_prompts(prompts: List[str]) -> List[Image.Image]:
    ims = []
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)
    for ptxt in prompts:
        out = pipe(
            ptxt,
            num_inference_steps=cfg.gen_steps,
            guidance_scale=cfg.cfg_scale,
            negative_prompt=cfg.negative_prompt,
            num_images_per_prompt=cfg.images_per_prompt,
            generator=generator,
        )
        ims.extend(out.images)
    return ims

# Prepare small forget/retain prompt batches for gradient scoring
forget_prompts = make_style_prompts(cfg.content_object, cfg.forget_style, cfg.grad_chunk)
retain_prompts = make_style_prompts(cfg.content_object, cfg.retain_style, cfg.grad_chunk)

# Generate tiny image pools and cache their embeddings (for speed)
with torch.no_grad():
    forget_imgs_pool = generate_images_for_prompts(forget_prompts)
    retain_imgs_pool = generate_images_for_prompts(retain_prompts)
    forget_img_emb = encode_image_emb(forget_imgs_pool)
    retain_img_emb = encode_image_emb(retain_imgs_pool)

layer_importance: Dict[str, float] = {}
layer_align: Dict[str, float] = {}
layer_grads_f: Dict[str, torch.Tensor] = {}
layer_grads_r: Dict[str, torch.Tensor] = {}

if len(razor_layers) > 0:
    print("[RAZOR] Computing one-time grads on text encoder layers (style)…")
    clip_enc.train()
    for p in clip_enc.parameters():
        p.requires_grad = True
    clip_enc.zero_grad(set_to_none=True)

    # Forget grads (GA + MM) — text prompts vs cached image embeddings
    txt_f = encode_text_emb(forget_prompts).requires_grad_(True)
    img_f = forget_img_emb.detach()
    # ensure batch sizes match (prompts equal generated images count)
    if txt_f.size(0) != img_f.size(0):
        # Repeat/trim text to match images
        if img_f.size(0) > txt_f.size(0):
            txt_f = txt_f.repeat(int(math.ceil(img_f.size(0)/txt_f.size(0))), 1)[:img_f.size(0)]
        else:
            img_f = img_f[:txt_f.size(0)]
    loss_f = cfg.w_GA * razor_loss(txt_f, img_f, mode="GA") + cfg.w_MM * razor_loss(txt_f, img_f, mode="MM")
    loss_f.backward(retain_graph=True)

    for name, module in razor_layers:
        g = None
        if hasattr(module, 'weight') and module.weight.grad is not None:
            g = module.weight.grad.detach().clone()
        elif any(p.grad is not None for p in module.parameters()):
            grads = []
            for p in module.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().flatten())
            if len(grads):
                g = torch.cat(grads)
        if g is not None:
            layer_grads_f[name] = g

    # Retain grads (GD)
    clip_enc.zero_grad(set_to_none=True)
    txt_r = encode_text_emb(retain_prompts).requires_grad_(True)
    img_r = retain_img_emb.detach()
    if txt_r.size(0) != img_r.size(0):
        if img_r.size(0) > txt_r.size(0):
            txt_r = txt_r.repeat(int(math.ceil(img_r.size(0)/txt_r.size(0))), 1)[:img_r.size(0)]
        else:
            img_r = img_r[:txt_r.size(0)]
    loss_r = cfg.w_GD * razor_loss(txt_r, img_r, mode="GD")
    loss_r.backward()

    for name, module in razor_layers:
        g = None
        if hasattr(module, 'weight') and module.weight.grad is not None:
            g = module.weight.grad.detach().clone()
        elif any(p.grad is not None for p in module.parameters()):
            grads = []
            for p in module.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().flatten())
            if len(grads):
                g = torch.cat(grads)
        if g is not None:
            layer_grads_r[name] = g

    # Importance & alignment + scoring
    layer_scores = []
    for name, module in razor_layers:
        if name in layer_grads_f and name in layer_grads_r:
            gf = layer_grads_f[name].float()
            gr = layer_grads_r[name].float()
            if hasattr(module, 'weight') and module.weight is not None:
                W = module.weight.detach().flatten().float()
            else:
                ps = [p.detach().flatten().float() for p in module.parameters()]
                W = torch.cat(ps) if len(ps) else torch.ones_like(gf)
            importance = gf.norm() / (W.norm() + 1e-8)
            imp_f = _to_scalar(importance)
            layer_importance[name] = imp_f
            a = F.cosine_similarity(gf, gr, dim=0).clamp(-1, 1)
            ali_f = _to_scalar(a)
            layer_align[name] = ali_f
            score = imp_f / (1.0 + max(0.0, ali_f))
            layer_scores.append((name, imp_f, ali_f, float(score)))

    layer_scores.sort(key=lambda x: x[3], reverse=True)

    print("Top-8 layers by RAZOR score (name, importance, alignment, score):")
    for row in layer_scores[:8]:
        print("   ", row)

    if len(layer_scores) > 0:
        imps = torch.tensor([x[1] for x in layer_scores], dtype=torch.float32)
        tau = 0.5 * (imps.mean() + imps.std()) if imps.numel() > 0 else torch.tensor(0.0)
        selected = [n for (n, imp, ali, s) in layer_scores if imp >= _to_scalar(tau)]
        if len(selected) == 0:
            selected = [layer_scores[0][0]]
        if len(selected) > cfg.max_selected_layers:
            selected = selected[:cfg.max_selected_layers]
        print(f"[RAZOR] Threshold τ={_to_scalar(tau):.4g}; selected {len(selected)} layers: {selected}")
    else:
        selected = []
        print("[RAZOR] No layers scored — skipping parameter updates.")
else:
    selected = []

# =====================
# 7) Iterative unlearning (single target: forget-style UA)
# =====================
@torch.no_grad()
def generate_images_style(style: str, k_prompts: int, neutral: bool=False) -> List[Image.Image]:
    prompts = make_style_prompts(cfg.content_object, style, k_prompts, neutral=neutral)
    return generate_images_for_prompts(prompts)

@torch.no_grad()
def style_unlearn_metrics() -> Tuple[float, float, float]:
    ims_f = generate_images_style(cfg.forget_style, cfg.eval_per_style_prompts)
    UA = 1.0 - clip_presence_score(ims_f, [f"{cfg.forget_style} style"])  # lower presence ⇒ higher UA

    ims_r = generate_images_style(cfg.retain_style, cfg.eval_per_style_prompts)
    IRA = clip_presence_score(ims_r, [f"{cfg.retain_style} style"])  # retain presence

    ims_o = generate_images_style(cfg.neutral_style, cfg.eval_per_style_prompts, neutral=True)
    CRA = clip_presence_score(ims_o, [f"{cfg.neutral_style} style"])  # content/style robustness proxy
    return UA, IRA, CRA

# Freeze everything except selected layers
train_params = []
if len(selected) > 0:
    for name, module in razor_layers:
        req = (name in selected)
        for p in module.parameters(recurse=False):
            p.requires_grad_(req)
            if req:
                train_params.append(p)
optimizer = torch.optim.AdamW(train_params, lr=1e-4) if len(train_params) else None

print("[Eval] Pre-unlearning (style) proxy metrics…")
UA0, IRA0, CRA0 = style_unlearn_metrics()
print(f"   UA={UA0:.3f}  IRA={IRA0:.3f}  CRA={CRA0:.3f}")

TARGET_UA = cfg.TARGET_UA
active_layers = selected[:1] if len(selected)>0 else []
step_budget = min(cfg.max_selected_layers, len(selected))

print("[RAZOR] Iterative layer widening (single target: UA for style)…")
UA, IRA, CRA = UA0, IRA0, CRA0
for it in range(max(1, step_budget)):
    if it>0 and it < len(selected):
        # add next-highest layer
        for cand in selected:
            if cand not in active_layers:
                active_layers.append(cand)
                break
        # toggle grads
        train_params = []
        for name, module in razor_layers:
            req = (name in active_layers)
            for p in module.parameters(recurse=False):
                p.requires_grad_(req)
                if req:
                    train_params.append(p)
        optimizer = torch.optim.AdamW(train_params, lr=1e-4) if len(train_params) else None

    # Cache small image pools per-iter for speed (forget/retain)
    with torch.no_grad():
        f_prompts = make_style_prompts(cfg.content_object, cfg.forget_style, cfg.grad_chunk)
        r_prompts = make_style_prompts(cfg.content_object, cfg.retain_style, cfg.grad_chunk)
        f_imgs_pool = generate_images_for_prompts(f_prompts)
        r_imgs_pool = generate_images_for_prompts(r_prompts)
        f_img = encode_image_emb(f_imgs_pool)
        r_img = encode_image_emb(r_imgs_pool)

    inner_steps = 4  # light
    for s in range(inner_steps):
        if not train_params:
            break
        clip_enc.train()
        optimizer.zero_grad(set_to_none=True)

        f_txt = encode_text_emb(f_prompts)
        r_txt = encode_text_emb(r_prompts)
        # batch-match
        if f_txt.size(0) != f_img.size(0):
            if f_img.size(0) > f_txt.size(0):
                f_txt = f_txt.repeat(int(math.ceil(f_img.size(0)/f_txt.size(0))), 1)[:f_img.size(0)]
            else:
                f_img = f_img[:f_txt.size(0)]
        if r_txt.size(0) != r_img.size(0):
            if r_img.size(0) > r_txt.size(0):
                r_txt = r_txt.repeat(int(math.ceil(r_img.size(0)/r_txt.size(0))), 1)[:r_img.size(0)]
            else:
                r_img = r_img[:r_txt.size(0)]

        loss = (
            cfg.w_GA * razor_loss(f_txt, f_img, mode="GA") +
            cfg.w_MM * razor_loss(f_txt, f_img, mode="MM") +
            cfg.w_GD * razor_loss(r_txt, r_img, mode="GD")
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()

    UA, IRA, CRA = style_unlearn_metrics()
    print(f"  Iter {it+1}/{max(1, step_budget)}  layers={active_layers}  => UA={UA:.3f} IRA={IRA:.3f} CRA={CRA:.3f}")

    if UA >= TARGET_UA:
        print("  ✔ UA target met — stop widening.")
        break

print("[Final] (style) Proxy scores:")
print(json.dumps({"UA":float(UA), "IRA":float(IRA), "CRA":float(CRA), "UA0":float(UA0), "IRA0":float(IRA0), "CRA0":float(CRA0)}, indent=2))
