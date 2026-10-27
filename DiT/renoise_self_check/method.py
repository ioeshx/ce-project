"""Renoise Self Check: velocity correction, shared-noise scoring and acceptance."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from benchmark.common import ROOT, concept_key, read_json, write_json
from benchmark.dit_cases import definitions
from benchmark.layout import method_directory
from .model import FlowModel


DEFAULT_SETTINGS = dict(lambda_ref=.01, rho=.3, theta=0., margin=0., check_fraction=.4, probe_seed=0)
SUPPORTED_MODELS = ("sd35_medium", "flux1_dev")


def check_steps(steps, fraction):
    """Zero-based middle window: 40 steps at 0.4 gives indices 12..27."""
    count = max(1, math.floor(steps * fraction + .5))
    start = (steps - count) // 2
    return range(start, min(start + count, steps - 1))  # Each check needs a next state.


def _noun(text):
    text = text.strip().lower()
    return ("an " if text[:1] in "aeiou" else "a ") + text


class RenoiseSelfCheck(FlowModel):
    """Benchmark plugin; initialization is the single method-validation boundary."""

    def __init__(self, config):
        self.settings = {**DEFAULT_SETTINGS, **config["adapter"].get("self_check", {})}
        s = self.settings
        if (not all(math.isfinite(s[k]) for k in ("lambda_ref", "rho", "theta", "margin", "check_fraction"))
                or s["lambda_ref"] < 0 or s["margin"] < 0
                or not 0 < s["rho"] < 1 or not 0 < s["check_fraction"] <= 1
                or not isinstance(s["probe_seed"], int) or s["probe_seed"] < 0):
            raise ValueError("Use finite parameters, lambda_ref/margin >= 0, 0 < rho < 1, "
                             "0 < check_fraction <= 1, and a nonnegative integer probe_seed.")
        if config["model_name"] not in SUPPORTED_MODELS:
            raise ValueError(f"Supported models: {SUPPORTED_MODELS}.")
        schedule = config["sampling"].get("scheduler_config", {})
        if schedule.get("stochastic_sampling") or schedule.get("invert_sigmas"):
            raise ValueError("Euler previews require deterministic sampling with decreasing noise.")

        # The case supplies targets; the benchmark mapping supplies their anchors.
        group = definitions()["cases"][config["case_id"]]["group"]
        anchors = {concept_key(k): v for k, v in read_json(ROOT / "configs/dit_target_anchor.json").items()}
        self.concepts = []
        for target in config["targets"]:
            anchor = config.get("erasure_anchor") if len(config["targets"]) == 1 else None
            anchor = anchor or anchors[concept_key(target)]
            if group == "style":
                df, da = f"a painting by {target}", "a painting in a general artistic style"
                pf, pa = f"an artwork in the style of {target}", "an artwork in a general artistic style"
            elif group == "i2p" and concept_key(target) == "nudity" and anchor.casefold() == "dressed":
                df, da = target, anchor
                pf, pa = "a photo of a nude adult person", "a photo of a dressed adult person"
            elif group == "ip":
                df, da = f"an image of {target}", f"an image of {_noun(anchor)}"
                pf, pa = f"a picture depicting {target}", f"a picture depicting {_noun(anchor)}"
            else:
                df, da = f"a photo of {_noun(target)}", f"a photo of {_noun(anchor)}"
                pf, pa = f"an image depicting {_noun(target)}", f"an image depicting {_noun(anchor)}"
            self.concepts.append(dict(target=target, anchor=anchor, direction_forget=df,
                                      direction_anchor=da, probe_forget=pf, probe_anchor=pa))

        super().__init__(config)
        self.config = config
        self.probe_time = torch.tensor(s["rho"] * self.pipe.scheduler.config.num_train_timesteps,
                                       dtype=torch.float32, device=self.device)

    def score(self, clean, noise, concept):
        """R > 0 means the target condition better predicts the re-noising velocity."""
        rho = self.settings["rho"]
        probe = ((1 - rho) * clean.float() + rho * noise).to(self.dtype)
        expected_velocity = noise - clean.float()
        forget, anchor = self.pair(probe, self.probe_time, concept["probe_forget"], concept["probe_anchor"])
        dims = tuple(range(1, probe.ndim))
        anchor_error = (anchor.float() - expected_velocity).square().mean(dim=dims)
        forget_error = (forget.float() - expected_velocity).square().mean(dim=dims)
        return anchor_error - forget_error

    def denoise(self, latents, records):
        """Preview two candidate steps; commit exactly one real Euler step."""
        scheduler, settings = self.pipe.scheduler, self.settings
        times, sigmas = scheduler.timesteps, scheduler.sigmas
        checkpoints = check_steps(len(times), settings["check_fraction"])
        history, cached_user, check_index = [], None, 0

        for i, time in enumerate(times):
            velocity = (cached_user if cached_user is not None else
                        self.velocity(latents, time, self.user))
            cached_user = None
            chosen = velocity
            if i in checkpoints and settings["lambda_ref"] != 0:
                concept = self.concepts[check_index % len(self.concepts)]
                check_index += 1
                sigma, sigma_next = sigmas[i], sigmas[i + 1]
                next_time = times[i + 1]

                # Candidate 0: preview the original step, then estimate its clean latent.
                z0 = (latents.float() + (sigma_next - sigma) * velocity).to(velocity.dtype)
                future0 = self.velocity(z0, next_time, self.user)
                clean0 = z0.float() - sigma_next * future0.float()

                # One new noise draw per sample/check, shared by the two candidates.
                # Its RNG is separate from generation and stable across batch sizes.
                noise_rows = []
                for record in records:
                    key = (f"{settings['probe_seed']}|{record['sample_id']}|{record['seed']}|"
                           f"{i}|{concept['target']}")
                    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63)
                    rng = torch.Generator(device="cpu").manual_seed(seed)
                    noise_rows.append(torch.randn(latents.shape[1:], generator=rng, dtype=torch.float32))
                noise = torch.stack(noise_rows).to(latents.device)

                r0 = self.score(clean0, noise, concept)
                eligible = r0 > settings["theta"]
                r1 = torch.full_like(r0, float("nan"))
                accepted = torch.zeros_like(eligible)
                cached_user = future0

                # Candidate 1 is needed only when a sample exceeds the threshold.
                if bool(eligible.any()):
                    forget, anchor = self.pair(latents, time, concept["direction_forget"],
                                                concept["direction_anchor"])
                    corrected = (velocity.float() + settings["lambda_ref"] *
                                 (anchor.float() - forget.float())).to(velocity.dtype)
                    z1 = (latents.float() + (sigma_next - sigma) * corrected).to(corrected.dtype)
                    future1 = self.velocity(z1, next_time, self.user)
                    clean1 = z1.float() - sigma_next * future1.float()
                    r1 = self.score(clean1, noise, concept)

                    accepted = eligible & (r1 < r0 - settings["margin"])
                    mask = accepted.view(-1, *([1] * (latents.ndim - 1)))
                    chosen = torch.where(mask, corrected, velocity)
                    cached_user = torch.where(mask, future1, future0)
                history.append((concept["target"], torch.stack((r0, r1, accepted.float()), dim=1)))

            latents = scheduler.step(chosen, time, latents, return_dict=False)[0]
        return latents, history

    @torch.inference_mode()
    def generate(self, records, sampling_config):
        # benchmark passes the same resolved profile used when loading this adapter.
        before_calls, before_evals = self.forward_calls, self.sample_evaluations
        latents = self.prepare(records)
        for concept in self.concepts:
            for key in ("direction_forget", "direction_anchor", "probe_forget", "probe_anchor"):
                self.condition(concept[key], len(records))
        latents, history = self.denoise(latents, records)
        images = self.decode(latents)
        self.save_statistics(records, history, self.forward_calls - before_calls,
                             self.sample_evaluations - before_evals)
        return images

    def save_statistics(self, records, history, calls, evaluations):
        root = method_directory(self.config["output_dir"], self.config["task"], self.config["method"],
                                self.config["case_id"])
        image_run = Path(records[0]["image"]).parent.name
        path = root / "self_check" / image_run / "statistics.json"
        report = read_json(path) if path.exists() else {"samples": {}}
        arrays = [(target, values.float().cpu().numpy()) for target, values in history]
        for index, record in enumerate(records):
            values = np.stack([v[index] for _, v in arrays]) if arrays else np.empty((0, 3))
            concept_rows = {}
            for concept in self.concepts:
                rows = [v[index] for target, v in arrays if target == concept["target"]]
                concept_rows[concept["target"]] = _summarize(
                    np.stack(rows) if rows else np.empty((0, 3)), self.settings["theta"])
            # Replacing by sample ID keeps resumed runs from counting an image twice.
            report["samples"][record["sample_id"]] = {
                **_summarize(values, self.settings["theta"]), "concepts": concept_rows,
                "transformer_sample_evaluations": evaluations / len(records), "batch_forward_calls": calls}
        samples = list(report["samples"].values())
        report.update(
            schema="renoise-self-check-statistics-v1", settings=self.settings,
            model_name=self.config["model_name"], task=self.config["task"], case_id=self.config["case_id"],
            checkpoints_1based=[i + 1 for i in check_steps(self.options["steps"], self.settings["check_fraction"])],
            direction_and_probe_prompts=self.concepts,
            guidance="benchmark defaults for every branch, including probes",
            candidate1_statistics="evaluated for a batch when any R0 > theta; acceptance is per sample",
            summary=_aggregate(samples),
            by_concept={c["target"]: _aggregate([r["concepts"][c["target"]] for r in samples])
                        for c in self.concepts})
        write_json(path, report)


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]  # Skipped candidate scores are NaN, not zero.
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "q05": float(np.quantile(values, .05)), "q50": float(np.quantile(values, .5)),
            "q95": float(np.quantile(values, .95))}


def _summarize(values, theta):
    """Shared summary format for one image and for its individual target concepts."""
    r0, r1, accepted = values.T
    return {"checks": len(values), "eligible": int((r0 > theta).sum()),
            "accepted": int(accepted.sum()), "candidate1_evaluated": int(np.isfinite(r1).sum()),
            "R0": _distribution(r0), "R1": _distribution(r1), "improvement": _distribution(r0 - r1)}


def _aggregate(rows):
    result = {"images": len(rows), **{key: sum(r[key] for r in rows)
              for key in ("checks", "eligible", "accepted", "candidate1_evaluated")}}
    result["acceptance_rate"] = result["accepted"] / result["checks"] if result["checks"] else 0.
    for name in ("R0", "R1", "improvement"):
        summaries = [r[name] for r in rows if r[name]["count"]]
        n = sum(s["count"] for s in summaries)
        if not n:
            result[name] = {"count": 0}
            continue
        mean = sum(s["mean"] * s["count"] for s in summaries) / n
        variance = sum(s["count"] * (s["std"] ** 2 + (s["mean"] - mean) ** 2) for s in summaries) / n
        result[name] = {"count": n, "mean": mean, "std": variance ** .5,
                        "min": min(s["min"] for s in summaries), "max": max(s["max"] for s in summaries)}
    return result
