"""CLIP and standard 2048-D FID kernels extracted from casu-2/metrics.py.

Kept independent of CASU sampling, checkpoints, and DiT modules.
"""
from pathlib import Path
import hashlib
import json
from zipfile import BadZipFile

class CLIPScorer:
    def __init__(self, model_name, device, local_files_only=False, long_prompt="error"):
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer, CLIPImageProcessor, CLIPModel, CLIPProcessor

        self.device = device
        self.long_prompt = long_prompt
        image_options = {"backend": "pil"} if hasattr(CLIPImageProcessor, "backend") else {"use_fast": False}
        # Select the image backend separately, without changing the tokenizer backend.
        self.processor = CLIPProcessor(
            image_processor=AutoImageProcessor.from_pretrained(
                model_name, local_files_only=local_files_only, **image_options),
            tokenizer=AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only, use_fast=False))
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=local_files_only, torch_dtype=torch.float32,
        ).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def score(self, paths, prompts):
        import torch
        from PIL import Image

        length = self.model.config.text_config.max_position_embeddings
        tokenized = self.processor.tokenizer(prompts, truncation=False)["input_ids"]
        if self.long_prompt == "error" and any(len(x) > length for x in tokenized):
            raise ValueError(f"A CLIP prompt exceeds {length} tokens; shorten it instead of silently truncating.")
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        tokenizer_opts = {"truncation": True, "max_length": length} if self.long_prompt == "truncate" else {}
        inputs = self.processor(text=prompts, images=images, padding=True,
                                **tokenizer_opts, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
            image = torch.nn.functional.normalize(output.image_embeds.float(), dim=-1)
            text = torch.nn.functional.normalize(output.text_embeds.float(), dim=-1)
            # Match util/clip_score_cal.py: 100*cosine, without zero clipping.
            scores = 100 * (image * text).sum(-1)
        return scores.cpu().numpy()


class FIDExtractor:
    def __init__(self, device, local_files_only=False):
        import torch
        from torch_fidelity.feature_extractor_inceptionv3 import URL_INCEPTION_V3, FeatureExtractorInceptionV3

        weight = None
        if local_files_only:
            weight = Path(torch.hub.get_dir()) / "checkpoints" / URL_INCEPTION_V3.rsplit("/", 1)[-1]
            if not weight.is_file():
                raise ValueError(f"Offline FID needs the standard torch-fidelity weights at {weight}.")
        self.device = device
        self.model = FeatureExtractorInceptionV3(
            "inception-v3-compat", ["2048"],
            feature_extractor_weights_path=str(weight) if weight else None,
            verbose=False,
        ).to(device).eval()

    def extract(self, paths, batch_size, workers):
        import torch
        from torch_fidelity.datasets import ImagesPathDataset
        from torch.utils.data import DataLoader
        import numpy as np

        loader = DataLoader(ImagesPathDataset(paths), batch_size=batch_size, shuffle=False,
                            num_workers=workers, drop_last=False)
        values = []
        with torch.inference_mode():
            for index, images in enumerate(loader):
                # Same uint8 input and TensorFlow-compatible resize as the old
                # torch_fidelity.calculate_metrics call; do not pre-resize.
                values.append(self.model(images.to(self.device))[0].float().cpu().numpy())
                print(f"[FID features] {min((index + 1) * batch_size, len(paths))}/{len(paths)}", flush=True)
        return np.concatenate(values)


def fid_from_features(original, edited, method="auto"):
    """Standard sample-Gaussian FID, optionally using an equivalent thin factor.

    For centered row features A and B, the covariance square-root trace equals
    ||A @ B.T||_* / sqrt((N-1)(M-1)). This avoids a 2048-D matrix square root
    when there are only 300/800 images per concept. Features remain 2048-D.
    """
    import numpy as np

    a, b = np.asarray(original, dtype=np.float64), np.asarray(edited, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or min(len(a), len(b)) < 2:
        raise ValueError("FID needs at least two images per distribution and matching feature dimensions.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite Inception features.")
    if method == "auto":
        method = "lowrank" if max(len(a), len(b)) < a.shape[1] else "covariance"
    am, bm = a.mean(0), b.mean(0)
    if method == "lowrank":
        ac, bc = a - am, b - bm
        trace_a = (ac * ac).sum() / (len(a) - 1)
        trace_b = (bc * bc).sum() / (len(b) - 1)
        cross = (ac @ bc.T) / ((len(a) - 1) * (len(b) - 1)) ** 0.5
        root_trace = np.linalg.svd(cross, compute_uv=False).sum()
        value = float(((am - bm) ** 2).sum() + trace_a + trace_b - 2 * root_trace)
    elif method == "covariance":
        from torch_fidelity.metric_fid import fid_statistics_to_metric
        value = float(fid_statistics_to_metric(
            {"mu": am, "sigma": np.atleast_2d(np.cov(a, rowvar=False))},
            {"mu": bm, "sigma": np.atleast_2d(np.cov(b, rowvar=False))}, False,
        )["frechet_inception_distance"])
    else:
        raise ValueError("FID method must be auto, lowrank or covariance.")
    if not np.isfinite(value) or value < -1e-5:
        raise ValueError(f"Invalid numerical FID: {value}")
    return max(0.0, value), method


def feature_fingerprint(paths, *, portable=True, stat_paths=None):
    from .paths import artifact_root, relative_path
    root = artifact_root(paths[0]) if paths else None
    entries = []
    for p, actual in zip(paths, stat_paths if stat_paths is not None else paths):
        stat = actual.stat()
        entries.append((relative_path(p, root) if portable else str(p), stat.st_size, stat.st_mtime_ns))
    import importlib.metadata
    return hashlib.sha256(json.dumps(["torch-fidelity-inception-v3-compat-2048",
                                     importlib.metadata.version("torch-fidelity"), entries]).encode()).hexdigest()


def cached_features(paths, path, extract, *, use_cache=True, refresh=False):
    import numpy as np

    fingerprint = feature_fingerprint(paths)
    if use_cache and not refresh and path.is_file():
        try:
            with np.load(path, allow_pickle=False) as saved:
                values = saved["features"]
                if (str(saved["fingerprint"].item()) in {fingerprint, feature_fingerprint(paths, portable=False)}
                        and values.shape == (len(paths), 2048) and np.isfinite(values).all()):
                    print(f"[FID cache] {path.name}", flush=True)
                    return values
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
    values = np.asarray(extract(paths), dtype=np.float32)
    if values.shape != (len(paths), 2048) or not np.isfinite(values).all():
        raise ValueError("Expected finite 2048-D standard Inception features for every image.")
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        import os
        import tempfile
        fd, temporary = tempfile.mkstemp(suffix=".npz", prefix=".features-", dir=path.parent)
        os.close(fd)
        try:
            np.savez_compressed(temporary, features=values, fingerprint=fingerprint)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return values
