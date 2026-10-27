"""Apply native UCE partial K/V weights, then use benchmark sampling verbatim."""
from pathlib import Path

from benchmark.adapters import DiffusersAdapter
from benchmark.common import concept_key, file_digest, read_json


def apply_weights(unet, weights):
    import torch
    expected = {name + ".weight" for name, _ in unet.named_modules()
                if "attn2" in name and name.endswith(("to_k", "to_v"))}
    params = dict(unet.named_parameters())
    if not expected or set(weights) != expected:
        raise ValueError("UCE checkpoint must contain exactly all cross-attention K/V weights.")
    for name, value in weights.items():
        if value.shape != params[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid UCE parameter: {name}")
    with torch.no_grad():
        for name, value in weights.items():
            params[name].copy_(value.to(params[name]))


class UCEAdapter(DiffusersAdapter):
    def __init__(self, config):
        from safetensors.torch import load_file
        checkpoint = Path(config["adapter"]["checkpoint"])
        metadata = read_json(checkpoint / "metadata.json")
        if (metadata.get("format") != "uce-sd14-partial-kv-v1"
                or metadata["base_model"] != config["base_model"]
                or metadata["task"] != config["task"]
                or metadata["settings"]["model_revision"] != config["sampling"].get("model_revision")
                or {concept_key(t) for t in metadata["targets"]} != {concept_key(t) for t in config["targets"]}
                or metadata["weights_sha256"] != file_digest(checkpoint / "uce.safetensors")):
            raise ValueError("UCE checkpoint and benchmark model/task/targets/revision do not match.")
        super().__init__(config)
        apply_weights(self.pipe.unet, load_file(str(checkpoint / "uce.safetensors"), device="cpu"))


def create(config):
    return UCEAdapter(config)
