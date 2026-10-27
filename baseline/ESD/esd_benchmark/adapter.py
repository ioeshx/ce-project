"""Native ESD selected weights with benchmark's unmodified image sampler."""
from pathlib import Path

from benchmark.adapters import DiffusersAdapter
from benchmark.common import concept_key, file_digest, read_json


def apply_weights(unet, weights, train_method):
    import torch
    from utils.esd_trainer import StableDiffusionESDAdapter
    expected = set(StableDiffusionESDAdapter().select_parameter_names(unet, train_method))
    params = dict(unet.named_parameters())
    if not expected or set(weights) != expected:
        raise ValueError("ESD checkpoint must contain exactly the native selected parameter set.")
    for key, value in weights.items():
        if value.shape != params[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid ESD parameter: {key}")
    with torch.no_grad():
        for key, value in weights.items():
            params[key].copy_(value.to(params[key]))


class ESDAdapter(DiffusersAdapter):
    def __init__(self, config):
        from safetensors.torch import load_file
        checkpoint = Path(config["adapter"]["checkpoint"])
        info = read_json(checkpoint / "metadata.json")
        if (info.get("format") != "esd-sd14-benchmark-v1" or info["base_model"] != config["base_model"]
                or info["task"] != config["task"]
                or info["settings"]["model_revision"] != config["sampling"].get("model_revision")
                or {concept_key(t) for t in info["targets"]} != {concept_key(t) for t in config["targets"]}
                or info["weights_sha256"] != file_digest(checkpoint / "esd.safetensors")):
            raise ValueError("ESD checkpoint does not match model/task/targets/revision/weights.")
        super().__init__(config)
        weights = load_file(str(checkpoint / "esd.safetensors"), device="cpu")
        if set(weights) != set(info["trainable_parameters"]):
            raise ValueError("ESD metadata and stored parameter keys disagree.")
        apply_weights(self.pipe.unet, weights, info["settings"]["train_method"])


def create(config):
    return ESDAdapter(config)
