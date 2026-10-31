"""Attention Replacement: native DiT sampling, offline AV means, online hard replacement."""
from __future__ import annotations

import copy
from functools import partial
import importlib.metadata
import math
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from benchmark.common import ROOT, digest, file_digest, read_json, write_json
from benchmark.dit_adapter import DiTAdapter
from benchmark.sampling_profile import initial_noise

DIRECTORY = Path(__file__).resolve().parent
SUPPORTED_MODELS = ("sd35_medium", "flux1_dev")
DEFAULTS = dict(layer_start=0, layer_end=None, step_start=0, step_end=None,
                step_fraction=0.3, tau=0.0, tau_by_concept={},
                offline_seeds=list(range(10000, 10016)), intervention="replace", trace=True,
                trace_activations=False, trace_root=None,
                cache_root=str(DIRECTORY / "cache"), prototype_dir=None)


def resolve_settings(config, overrides=None):
    """Runner entry: resolve concept pairs and validate method parameters once."""
    settings = {**copy.deepcopy(DEFAULTS),
                **config["adapter"].get("attention_replacement", {}), **(overrides or {})}
    anchors = read_json(ROOT / "configs/dit_target_anchor.json")
    # object_multi5 includes gas pump, absent from the shared mapping.
    anchors["gas pump"] = "machine"
    anchors.update(settings.pop("anchor_overrides", {}))
    explicit = settings.get("pairs")
    settings["pairs"] = explicit or [
        {"target": target, "anchor": config.get("erasure_anchor") or anchors[target]}
        for target in config["targets"]]
    steps = config["sampling"]["steps"]
    settings["step_end"] = (settings["step_end"] if settings["step_end"] is not None else
                            math.ceil(settings["step_fraction"] * steps))
    problems = []
    if config["model_name"] not in SUPPORTED_MODELS:
        problems.append(f"supported models: {SUPPORTED_MODELS}")
    if not 0 <= settings["step_start"] < settings["step_end"] <= steps:
        problems.append("require 0 <= step_start < step_end <= sampling.steps")
    if settings["layer_start"] < 0 or (settings["layer_end"] is not None and
                                      settings["layer_end"] <= settings["layer_start"]):
        problems.append("require 0 <= layer_start < layer_end")
    if (not settings["offline_seeds"] or len(set(settings["offline_seeds"])) != len(settings["offline_seeds"])
            or any(type(seed) is not int or not 0 <= seed < 2 ** 63 for seed in settings["offline_seeds"])):
        problems.append("offline_seeds must be nonempty, unique integers in [0, 2**63)")
    if not settings["pairs"] or any(not p["target"].strip() or not p["anchor"].strip()
                                    for p in settings["pairs"]):
        problems.append("provide nonempty target/anchor pairs")
    if not all(math.isfinite(v) for v in [settings["tau"], *settings["tau_by_concept"].values()]):
        problems.append("tau values must be finite")
    if settings["intervention"] not in {"replace", "observe"}:
        problems.append("intervention must be replace or observe")
    if config["sampling"].get("true_cfg_scale", 1.0) != 1.0:
        problems.append("FLUX true CFG is unsupported; use true_cfg_scale=1")
    if config["sampling"].get("skip_guidance_layers"):
        problems.append("SD3 skip-layer guidance is unsupported")
    if problems:
        raise ValueError("; ".join(problems))
    settings["cache_root"] = str(Path(settings["cache_root"]).expanduser().resolve())
    if settings["trace_root"]:
        settings["trace_root"] = str(Path(settings["trace_root"]).expanduser().resolve())
    if settings["prototype_dir"]:
        settings["prototype_dir"] = str(Path(settings["prototype_dir"]).expanduser().resolve())
    return settings


def cache_identity(config, settings):
    """Only trajectory/collection settings belong in the offline identity, not tau."""
    from huggingface_hub import try_to_load_from_cache

    opts = config["sampling"]
    source = config["adapter"].get("model_path") or config["base_model"]
    index = (Path(source) / "model_index.json" if Path(source).is_dir() else
             try_to_load_from_cache(source, "model_index.json", revision=opts.get("model_revision")))
    resolved = str(Path(index).parent) if isinstance(index, (str, Path)) else None
    keys = ("dtype", "height", "width", "steps", "guidance_scale", "negative_prompt",
            "true_cfg_scale", "max_sequence_length", "scheduler", "scheduler_config", "model_revision", "noise")
    return {"schema": "attention-replacement-av-v1", "model": config["model_name"],
            "source": source, "resolved_snapshot": resolved,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "diffusers", "transformers", "safetensors")},
            "sampling": {key: opts.get(key) for key in keys},
            "capture": {key: settings[key] for key in
                        ("layer_start", "layer_end", "step_start", "step_end", "offline_seeds")}}


def prototype_directory(config, settings):
    identity = cache_identity(config, settings)
    path = (Path(settings["prototype_dir"]) if settings["prototype_dir"] else
            Path(settings["cache_root"]) / config["model_name"] / digest(identity)[:16])
    return path, identity


def read_prototypes(config, settings, *, verify=True):
    """Online entry: verify the selected collection and its means before model loading."""
    directory, identity = prototype_directory(config, settings)
    metadata = read_json(directory / "collection.json")
    stored = metadata["identity"]
    old, new = stored["capture"], identity["capture"]
    compatible = {k: v for k, v in stored.items() if k != "capture"} == {
        k: v for k, v in identity.items() if k != "capture"}
    compatible &= old["step_start"] <= new["step_start"] < new["step_end"] <= old["step_end"]
    compatible &= old["layer_start"] <= new["layer_start"]
    compatible &= (old["layer_end"] is None or
                   (new["layer_end"] is not None and new["layer_end"] <= old["layer_end"]))
    if verify and not compatible:
        raise ValueError(f"Prototype collection does not match sampling/window: {directory}")
    means = {}
    for prompt in dict.fromkeys(p[key] for p in settings["pairs"] for key in ("target", "anchor")):
        folder = directory / "prompts" / digest(prompt)[:16]
        manifest = read_json(folder / "means.json")
        valid = manifest["prompt"] == prompt and manifest["seeds"] == old["offline_seeds"]
        if verify:
            valid &= all((folder / name).is_file() and file_digest(folder / name) == checksum
                         for name, checksum in manifest["files"].items())
        if verify and not valid:
            raise ValueError(f"Incomplete or changed prototype means: {folder}; run collect again")
        means[prompt] = {"directory": str(folder), "digest": digest(manifest)}
    return directory, metadata, means


def replace_attention(x, prototypes, pairs, tau, tau_by_concept, intervene=True):
    """All concepts inspect the same x. Strict comparisons give stable first-pair ties.

    x: [B,H,W,D]; prototypes: prompt -> [H,W,D], FP32 on x.device.
    No normalization, channel averaging, radius gate, or soft replacement.
    """
    xf = x.float()
    best = torch.full(x.shape[:-1], -torch.inf, device=x.device)
    winner = torch.full(x.shape[:-1], -1, dtype=torch.int32, device=x.device)
    replacement = x.clone()
    traces = {}
    for i, pair in enumerate(pairs):
        target, anchor = prototypes[pair["target"]], prototypes[pair["anchor"]]
        d_tar = (xf - target).square().sum(dim=-1)
        d_anc = (xf - anchor).square().sum(dim=-1)
        margin = d_anc - d_tar
        mask = margin > tau_by_concept.get(pair["target"], tau)
        wins = mask & (margin > best)
        replacement = torch.where(wins[..., None], anchor.to(x.dtype), replacement)
        best = torch.where(wins, margin, best)
        winner = torch.where(wins, i, winner)
        traces.update({f"d_tar_{i}": d_tar, f"d_anc_{i}": d_anc,
                       f"margin_{i}": margin, f"mask_{i}": mask})
    result = replacement if intervene else x
    change = (result.float() - xf).square().sum(-1).mean(dim=(-2, -1))
    traces.update(I_user=x, I_new=result, mask_union=winner >= 0, winner=winner,
                  mean_squared_change=change)
    return result, traces


class AttentionReplacement(DiTAdapter):
    """One stateful owner for the pipeline, hooks, offline cache and per-sample trace."""

    def __init__(self, config):
        self.settings = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(config["adapter"]["attention_replacement"])}
        self.collecting = self.settings.get("collect", False)
        if self.collecting:
            self.cache_dir, identity = prototype_directory(config, self.settings)
            self.collection = {"identity": identity}
            if (self.cache_dir / "collection.json").exists():
                self.collection = read_json(self.cache_dir / "collection.json")
                if self.collection["identity"] != identity:
                    raise ValueError("collect requires an identical collection window and seed list")
        else:
            self.cache_dir, self.collection, self.means = read_prototypes(config, self.settings)
        native = copy.deepcopy(config)
        native["adapter"] = {"kind": "diffusers", "model_path": config["adapter"]["model_path"]} if (
            config["adapter"].get("model_path")) else {"kind": "original"}
        super().__init__(native)
        self.active = False
        self.handles = []
        self.sites = []
        transformer = self.pipe.transformer
        blocks = [(i, f"transformer_blocks.{i}", block, False)
                  for i, block in enumerate(transformer.transformer_blocks)]
        if self.flux:
            blocks += [(len(blocks) + j, f"single_transformer_blocks.{j}", block, True)
                       for j, block in enumerate(transformer.single_transformer_blocks)]
        end = self.settings["layer_end"] or len(blocks)
        if not self.settings["layer_start"] < end <= len(blocks):
            raise ValueError(f"Layer window exceeds the loaded model's {len(blocks)} blocks")
        self.grid = (self.config["sampling"]["height"] // self.pipe.vae_scale_factor,
                     self.config["sampling"]["width"] // self.pipe.vae_scale_factor)
        patch = 2 if self.flux else transformer.config.patch_size
        self.grid = tuple(size // patch for size in self.grid)
        for layer, path, block, single in blocks[self.settings["layer_start"]:end]:
            for branch in (["attn"] if single else [b for b in ("attn", "attn2")
                                                       if getattr(block, b, None) is not None]):
                attention = getattr(block, branch)
                site = {"id": f"layer_{layer:03d}_{branch}", "layer": layer, "branch": branch,
                        "path": path + (".proj_out" if single else f".{branch}.to_out.0"),
                        "single": single, "channels": attention.inner_dim, "grid": list(self.grid)}
                self.sites.append(site)
                projection = block.proj_out if single else attention.to_out[0]
                self.handles.append(projection.register_forward_pre_hook(partial(self._attention_hook, site)))
        self.handles.append(transformer.register_forward_pre_hook(self._transformer_hook, with_kwargs=True))
        if self.collecting:
            self.collection["sites"] = self.sites
            write_json(self.cache_dir / "collection.json", self.collection)
        else:
            stored = {s["id"]: s for s in self.collection["sites"]}
            if any(stored.get(s["id"]) != s for s in self.sites):
                raise ValueError("Loaded model attention sites differ from the prototype collection")

    def _transformer_hook(self, module, args, kwargs):
        if not self.active:
            return
        self.forward_calls += 1
        self.text_length = kwargs["encoder_hidden_states"].shape[1]
        if self.forward_calls == 1:
            self.schedule = {"timesteps": self.pipe.scheduler.timesteps.cpu().tolist(),
                             "sigmas": self.pipe.scheduler.sigmas.cpu().tolist()}
            if "schedule" in self.collection and self.collection["schedule"] != self.schedule:
                raise ValueError("Actual scheduler trajectory differs from the offline collection")

    def _attention_hook(self, site, module, args):
        if not self.active or not self.settings["step_start"] <= self.step < self.settings["step_end"]:
            return
        started = time.perf_counter()
        tensor = args[0]
        batch = len(self.records)
        # SD3 native CFG is [negative B, positive B]; leave negative rows unchanged.
        rows = slice(batch, 2 * batch) if self.cfg else slice(0, batch)
        tokens = slice(self.text_length, None) if site["single"] else slice(None)
        channels = slice(0, site["channels"])
        x = tensor[rows, tokens, channels].reshape(batch, *self.grid, site["channels"])
        filename = f"step_{self.step:04d}/{site['id']}.safetensors"
        if self.collecting:
            payload = {"I": x[0]}
        else:
            prototypes = {prompt: load_file(str(Path(info["directory"]) / "means" / filename))["mean"].to(x.device)
                          for prompt, info in self.means.items()}
            result, payload = replace_attention(x, prototypes, self.settings["pairs"],
                self.settings["tau"], self.settings["tau_by_concept"],
                self.settings["intervention"] == "replace")
        self.hook_seconds += time.perf_counter() - started
        if self.collecting or self.settings["trace"]:
            started = time.perf_counter()
            for i, directory in enumerate(self.trace_dirs):
                destination = directory / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                values = payload if self.collecting else {
                    key: value[i] for key, value in payload.items()
                    if self.settings["trace_activations"] or key not in {"I_user", "I_new"}}
                # clone removes storage aliases (observe mode has I_new == I_user).
                save_file({key: value.detach().cpu().contiguous().clone() for key, value in values.items()},
                          str(destination))
            self.io_seconds += time.perf_counter() - started
        if not self.collecting and self.settings["intervention"] == "replace":
            changed = tensor.clone()
            changed[rows, tokens, channels] = result.reshape(batch, -1, site["channels"])
            return (changed, *args[1:])

    def generate(self, records, sampling_config):
        opts, pipe = sampling_config, self.pipe
        self.records, self.step, self.forward_calls = records, 0, 0
        self.hook_seconds = self.io_seconds = 0.0
        self.cfg = not self.flux and opts["guidance_scale"] > 1
        if not self.collecting:
            self.trace_dirs = []
            for record in records:
                image = Path(record["image"])
                root = image.parent.parent.parent / "traces"
                if self.settings["trace_root"]:
                    root = (Path(self.settings["trace_root"]) / self.config["model_name"] /
                            self.config["task"] / self.config["case_id"] / self.config["method"])
                self.trace_dirs.append(root / image.parent.name / image.stem)
        channels = pipe.transformer.config.in_channels // 4 if self.flux else pipe.transformer.config.in_channels
        shape = (channels, opts["height"] // pipe.vae_scale_factor, opts["width"] // pipe.vae_scale_factor)
        latents = torch.stack([initial_noise(shape, r["seed"], dtype=self.dtype, device="cpu") for r in records])
        if self.flux:
            latents = pipe._pack_latents(latents, len(records), *shape)

        def step_end(pipeline, step, timestep, callback_kwargs):
            self.step = step + 1
            return {}

        kwargs = dict(prompt=[r["prompt"] for r in records], latents=latents.to(self.device),
                      generator=[torch.Generator("cpu").manual_seed(r["seed"]) for r in records],
                      height=opts["height"], width=opts["width"], num_inference_steps=opts["steps"],
                      guidance_scale=opts["guidance_scale"], max_sequence_length=opts["max_sequence_length"],
                      num_images_per_prompt=1, callback_on_step_end=step_end)
        if self.flux:
            kwargs["true_cfg_scale"] = opts.get("true_cfg_scale", 1.0)
            if opts.get("negative_prompt") is not None:
                kwargs["negative_prompt"] = [opts["negative_prompt"]] * len(records)
        else:
            kwargs["negative_prompt"] = [opts.get("negative_prompt") or ""] * len(records)
        started = time.perf_counter()
        self.active = True
        try:
            with torch.inference_mode():
                images = pipe(**kwargs).images
        finally:
            self.active = False
        elapsed = time.perf_counter() - started
        for record, directory in zip(records, self.trace_dirs):
            write_json(directory / "metadata.json", {
                "record": record, "settings": self.settings, "sites": self.sites, "schedule": self.schedule,
                "sampling": opts, "model": self.config["model_name"], "device": str(self.device),
                "dtype": str(self.dtype), "runtime_info": self.runtime_info,
                "prototype_dir": str(self.cache_dir),
                "prototype_means": getattr(self, "means", {}), "forward_calls": self.forward_calls,
                "batch_size": len(records), "batch_wall_seconds": elapsed,
                "batch_hook_host_seconds": self.hook_seconds, "batch_trace_io_seconds": self.io_seconds,
                "timing_note": "Host wall times; trace I/O includes CPU transfer. Batch times are shared."})
        if self.collecting:
            self.collection["schedule"] = self.schedule
            write_json(self.cache_dir / "collection.json", self.collection)
        return images

    def collect(self, prompts):
        """Independent native trajectories; stream FP32 means over seeds per site/step."""
        for prompt in dict.fromkeys(prompts):
            folder = self.cache_dir / "prompts" / digest(prompt)[:16]
            completed = folder / "means.json"
            if completed.exists():
                manifest = read_json(completed)
                if all((folder / name).is_file() and file_digest(folder / name) == checksum
                       for name, checksum in manifest["files"].items()):
                    print(f"[reuse prototype] {prompt!r}: {folder}", flush=True)
                    continue
            for seed in self.settings["offline_seeds"]:
                trajectory = folder / f"seed_{seed}"
                if (trajectory / "complete.json").exists():
                    continue
                self.trace_dirs = [trajectory]
                record = {"sample_id": f"seed_{seed}", "prompt": prompt, "seed": seed}
                images = self.generate([record], self.config["sampling"])
                images[0].save(trajectory / "final.png")
                write_json(trajectory / "complete.json", {"prompt": prompt, "seed": seed})
                print(f"[collect] {prompt!r} seed={seed}", flush=True)
            files = {}
            for step in range(self.settings["step_start"], self.settings["step_end"]):
                for site in self.sites:
                    name = f"step_{step:04d}/{site['id']}.safetensors"
                    total = torch.zeros((*self.grid, site["channels"]), dtype=torch.float32)
                    for seed in self.settings["offline_seeds"]:
                        total += load_file(str(folder / f"seed_{seed}" / name))["I"].float()
                    destination = folder / "means" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    save_file({"mean": total / len(self.settings["offline_seeds"])}, str(destination))
                    files["means/" + name] = file_digest(destination)
            write_json(completed, {"prompt": prompt, "seeds": self.settings["offline_seeds"], "files": files})
            print(f"[prototype ready] {prompt!r}: {folder}", flush=True)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        super().close()
