# 💎 GEM: Geometric Erasure by Contrastive Velocity Matching in Rectified Flows (ICML 2026)

<p align="left-aligned">
  <a href="https://arxiv.org/abs/2606.00140"><img src="https://img.shields.io/badge/Paper-arXiv-005393?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper on arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-4ec9ff?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="MIT License"></a>
<a href="https://jonasgrebe.github.io/research/projects/gem/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page"></a>
</p>

<p align="center">
  <strong>ICML 2026 (Spotlight)</strong> &nbsp;·&nbsp; 🇰🇷 Seoul &nbsp;·&nbsp; Official PyTorch implementation<br>
  <em>"GEM: Geometric Erasure by Contrastive Velocity Matching in Rectified Flows"</em>
</p>

![Teaser](assets/teaser.png)

<p align="center"><em><strong>GEM</strong> removes targeted concepts from text-to-image rectified flow transformers, including nudity, gory content, and rights-protected concepts, while preserving overall model utility.</em></p>

While the rapid adoption of multimodal generative models offers immense potential, it has also increased the risks of harmful content synthesis, deepfakes, and copyright infringements. To address these challenges, concept erasure has emerged as a prospective safeguard. However, as the field gradually transitions from U-Net-based diffusion models to Rectified Flow Transformers, erasure research has struggled to keep pace. In this work, we introduce GEM, a simple but highly effective erasure framework for Rectified Flow models. As part of our contribution, we establish a principled bridge between trajectory-based unlearning grounded in Generative Flow Networks and classic teacher-guided erasure: we translate trajectory-based signals into a teacher-guided flow-matching setup that unifies the strengths of both paradigms. Concretely, a teacher provides complementary attraction and repulsion signals that we combine into a single geometric guidance objective, yielding targeted suppression of unwanted concepts while preserving benign generation.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Concept Erasure (Training)](#1-concept-erasure-training)
  - [Inference](#2-inference)
- [Evaluation](#evaluation)
- [Citation](#citation)

---

## Installation

GEM uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

```bash
pip install uv
uv sync
```

---

## Usage

The codebase exposes two primary entrypoints:

| Entrypoint | Purpose |
|---|---|
| `gem.train` | Fine-tune a model to erase a target concept |
| `gem.inference` | Run text-to-image inference on original or erased models |

### 1. Concept Erasure (Training)

Erasure is performed by pairing a **target concept** with an **anchor concept**:

```bash
# Nudity erasure
uv run python -m gem.train --model_type flux --method gem --targets "nudity" --anchors "fully dressed"

# Bloody gore erasure
uv run python -m gem.train --model_type flux --method gem --targets "bloody gore" --anchors "safe and clean" --max_step_budget 500
```

**Options**

- **Checkpoints** — Saved automatically to `models/adhoc_runs/run_gem_<random_id>`. Use `--run_id` to set a custom path.
- **Logging** — Pass `--use_wandb` to enable Weights & Biases logging.
- **Hyperparameters** — All configurable options (e.g. `--eta`) are documented in `gem.operators.erasure.erasure_gem`.

---

### 2. Inference

#### Pre-trained Checkpoints

Pre-trained checkpoints for nudity and bloody gore erasure are hosted on the [Multimodal AI Lab HuggingFace page](https://huggingface.co/MAI-Lab):

| Concept | Checkpoint |
|---|---|
| Nudity | [`MAI-Lab/GEM-Flux-Nudity`](https://huggingface.co/MAI-Lab/GEM-Flux-Nudity) |
| Bloody Gore | [`MAI-Lab/GEM-Flux-Bloody-Gore`](https://huggingface.co/MAI-Lab/GEM-Flux-Bloody-Gore) |

#### Custom Prompts

Compare outputs from the original Flux model against a GEM-erased checkpoint:

```bash
# Original Flux.1[dev]
uv run python -m gem.inference --model_type flux --prompts "a bloody gore scene"

# GEM-erased checkpoint
uv run python -m gem.inference --model_type flux --prompts "a bloody gore scene" --model_name_or_path "MAI-Lab/GEM-Flux-Bloody-Gore"
```

Multiple prompts can be passed as a space-separated list after `--prompts`.

#### Dataset Evaluation

Dataset wrappers for standard benchmarks (RAB, P4D, and basic-template datasets) are available under `gem.datasets.prompt_datasets`.

> **Note:** Source files for I2P, T2I-RP, MJHQ, and COCO are not included, but can be added as CSV files. See `gem.datasets.get_prompt_dataset(...)` for details.

```bash
# Evaluate on original Flux.1[dev]
uv run python -m gem.inference --model_type flux --dataset_name "basic_subject<naked person>"
uv run python -m gem.inference --model_type flux --dataset_name "rab_nudity"

# Evaluate on a GEM checkpoint
uv run python -m gem.inference --model_type flux --model_name_or_path "MAI-Lab/GEM-Flux-Nudity" --dataset_name "basic_subject<naked person>"
uv run python -m gem.inference --model_type flux --model_name_or_path "MAI-Lab/GEM-Flux-Nudity" --dataset_name "rab_nudity"
```

---

## Evaluation

Evaluation classes for all metrics reported in the paper are provided under `gem.evaluation`. To evaluate a folder of generated images:

```python
from gem.evaluation.nudity_evaluator import NudityEvaluator
from dotenv import load_dotenv

load_dotenv()

evaluator = NudityEvaluator()
evaluator.evaluate(
    "outputs/images/...",
    log_path="nudity_eval_results.csv"
)
```

Results are logged automatically to `outputs/metrics/nudity_eval_results.csv`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{grebe2026gem,
    title     = {{GEM}: Geometric Erasure by Contrastive Velocity Matching in Rectified Flows},
    author    = {Grebe, Jonas Henry and Braun, Tobias and Rohrbach, Anna and Rohrbach, Marcus},
    booktitle = {Forty-third International Conference on Machine Learning},
    year      = {2026},
    url       = {https://openreview.net/forum?id=NBMCwxTRSA}
}
```
