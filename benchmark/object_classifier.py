"""Fixed ImageNet-1K V2 ResNet-50 classification; never re-rank ten logits."""
from __future__ import annotations

from pathlib import Path

from .common import file_digest
from .execution import runtime_device

IMAGENET10 = {"church": 497, "golf ball": 574, "english springer": 217,
              "goose": 99, "chain saw": 491, "cassette player": 482,
              "french horn": 566, "garbage truck": 569, "banana": 954, "parachute": 701}


def weight_path(opts):
    import torch
    from torchvision.models import ResNet50_Weights
    if opts.get("resnet_weights", "IMAGENET1K_V2") != "IMAGENET1K_V2":
        raise ValueError("Object protocol requires ResNet50_Weights.IMAGENET1K_V2.")
    return Path(opts.get("resnet_weights_path") or
                Path(torch.hub.get_dir()) / "checkpoints" / Path(ResNet50_Weights.IMAGENET1K_V2.url).name)


def download_resnet(destination=None):
    import torch
    from torchvision.models import ResNet50_Weights
    path = Path(destination) if destination else weight_path({})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(ResNet50_Weights.IMAGENET1K_V2.url, str(path), hash_prefix="11ad3fa6")
    return path


class ResNetClassifier:
    def __init__(self, opts):
        import torch
        from torchvision.models import resnet50, ResNet50_Weights
        path = weight_path(opts)
        weights = ResNet50_Weights.IMAGENET1K_V2
        if opts["local_files_only"] and not path.is_file():
            raise FileNotFoundError("Offline ResNet-50 V2 weights missing; run python -m benchmark download --resource resnet50.")
        if opts.get("resnet_weights_path"):
            if not path.is_file():
                download_resnet(path)
            self.model = resnet50(weights=None)
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        else:
            # Torchvision reuses its standard cache and downloads missing weights.
            self.model = resnet50(weights=weights)
        checksum = file_digest(path)
        if not checksum.startswith("11ad3fa6"):
            raise ValueError("ResNet weights do not match the fixed official ImageNet-1K V2 release.")
        self.device = runtime_device(opts["device"])
        self.model.to(self.device).eval()
        self.preprocess = weights.transforms()
        if any(weights.meta["categories"][index].casefold() != name for name, index in IMAGENET10.items()):
            raise ValueError("Installed torchvision ImageNet category mapping differs from this protocol.")
        self.provenance = {"weights": "IMAGENET1K_V2", "weights_sha256": checksum,
                           "preprocessing": repr(self.preprocess), "ranking": "full ImageNet-1K, 1000 classes"}

    def predict(self, paths):
        import torch
        from PIL import Image
        batch = []
        for path in paths:
            with Image.open(path) as im:
                batch.append(self.preprocess(im.convert("RGB")))
        with torch.inference_mode():
            logits = self.model(torch.stack(batch).to(self.device)).float()
            if logits.shape != (len(paths), 1000) or not torch.isfinite(logits).all():
                raise ValueError("ResNet must return 1000 finite logits per image.")
            probabilities, indices = logits.softmax(-1).topk(3, dim=-1)
        return [{"prediction_index": int(i[0]), "top3_indices": i.tolist(), "top3_probabilities": p.tolist()}
                for i, p in zip(indices.cpu(), probabilities.cpu())]
