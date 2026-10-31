# RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning

This README consolidates all details discussed across previous chats into one comprehensive, reproducible document for RAZOR's implementation across **CLIP**, **Stable Diffusion**, and **Vision-Language Models (LLaVA)**. It includes conceptual summaries, method descriptions, datasets, metrics, implementation notes, and reproducibility guidance.

---

## 📌 Overview

**RAZOR** (Ratio-Aware Zero/One-step Optimized Retentive Unlearning) is a *multi-layer, ratio-aware machine unlearning framework* designed for transformer-based models.

It addresses the challenge of removing targeted information (identities, styles, objects, concepts) **without retraining** and **without harming utility**, and works across:

* **CLIP** (contrastive encoder)
* **Stable Diffusion** (generative diffusion models)
* **Vision-Language Models (VLMs)** such as **LLaVA**

The method is grounded in the core principles defined in *RAZOR.pdf* and compared to baselines such as **SLUG**, **SalUn**, **SSD**, and **LoTUS**.

---

## 🧠 Key Ideas of RAZOR

### 1. **Ratio-Aware Layer Scoring**

RAZOR computes one-shot gradients for each layer/head:

* Forget gradient: ( g_f^l = \nabla_{\theta_l} \mathcal{L}_{forget} )
* Retain gradient: ( g_r^l = \nabla_{\theta_l} \mathcal{L}_{retain} )

The importance score is:
[
\phi(l) = \frac{\lVert g_f^l \rVert^2}{\lVert \theta_l \rVert^2 + \epsilon} (1 - \cos(g_f^l, g_r^l))^\alpha
]
Layers with high (\phi(l)) are edited.

### 2. **Three-Loss Framework**

RAZOR optimizes:

* **Retain loss** (utility preservation)
* **Forget loss** (erase targeted info)
* **Mismatch loss** (stability / drift control)

### 3. **Iterative Refinement**

If forgetting is not sufficient, new layers are added progressively (max 6 iterations).

### 4. **Binary Search Step Size**

Per-layer step sizes are selected using a lightweight binary search.

---

## 📚 Files Included

```
RAZOR-CLIP.py       # Targeted unlearning on CLIP
Razor-SD.py         # Unlearning for Stable Diffusion
Razor-VLM.py        # Unlearning for LLaVA (vision-language model)
Razor.pdf            # Main RAZOR paper (CVPR submission)
SLUG ()             # Baseline unlearning approach
Readme.md            # This file
```

These scripts are **Google Colab–ready** with full dependency installation.

---

## ⚙️ Reproducibility Setup

### Install Dependencies

All scripts automatically install required packages:

```
pip uninstall -y pyarrow
pip install "pyarrow>=14,<20" numpy<2
pip install transformers accelerate bitsandbytes datasets peft timm sentencepiece torchvision pillow safetensors
pip install xformers==0.0.27.post2
```

### Supported Models

* **CLIP**: OpenCLIP ViT-B/32, ViT-L/14
* **Stable Diffusion**: SD-v1.5, SD-v3
* **VLM**: LLaVA v1.5–7B, v1.6–7B/8B

### Dataset Sources

* **LAION-400M** (primary forget/retain construction)
* CIFAR-10
* ImageNet-1K
* CelebA
* UnlearnCanvas (for SD style/object unlearning)

---

## 📊 Evaluation Metrics

### For CLIP (M1–M5)

* **M1 ↓ Forget Accuracy**
* **M2 ↓ Forget Cosine Similarity**
* **M3 → 0 Privacy Leakage**
* **M4 ↑ Utility Preservation**
* **M5 ↑ Retrieval Stability**

### For Stable Diffusion

* **UA ↑ Unlearned Accuracy**
* **IRA ↑ In-retain Alignment**
* **CRA ↑ Content Robustness Alignment**

### For VLMs

* **FA ↓ Forget Accuracy**
* **MME (Cognition/Perception)**
* **GQA**
* **MMBench (%)**

---

## 📌 Summary of RAZOR Strengths

* Works across model families (CLIP, SD, VLM) with identical principles.
* Multi-layer editing prevents under-forgetting (unlike SLUG).
* High retention accuracy and stability.
* Robust under 8-bit and 4-bit quantization.
* Strong performance across benchmarks.

---

## 🧪 Running Experiments

### CLIP Unlearning

```
python RAZOR-CLIP.py
```

### Stable Diffusion Unlearning

```
python Razor-SD.py
```

### LLaVA/VLM Unlearning

```
python Razor-VLM.py
```

Each script:

* Computes layer scores
* Selects high-ϕ layers
* Applies RAZOR loss
* Iteratively expands edited set
* Reports metrics
* Saves LoRA adapters

---

## 📁 Outputs

* Edited layers
* Forget/retain metrics
* LoRA adapters saved to:

  * `razor_clip_lora/`
  * `razor_sd_lora/`
  * `razor_llava_lora/`

---

## 🔬 Notes on Quantization Stability

RAZOR remains highly stable under:

* FP32
* 8-bit
* 4-bit (nf4 quantization)

This addresses concerns raised by *quantization-induced unlearning failure*.

---



---

## 

This README consolidates the entire discussion, all code references, the conceptual framework from the attached **RAZOR.pdf**, and implementation strategies used throughout the project.

This document is fully self-contained and designed for direct use in replication studies, academic submissions, or project documentation.
