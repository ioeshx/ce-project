"""Average raw steering scores by evaluation concept, keeping every step and layer."""
from pathlib import Path

import torch

from benchmark.common import concept_key, slug, write_csv, write_json


def accumulate_scores(root, records, scores, config, spatial_shape, timesteps, *, position_kind=None, layer_ids=None):
    """scores: [target, step, layer, sample, position], before clipping.

    Persist means and sample IDs after each completed batch. Resumed generations
    can repeat a sample without counting its score twice.
    """
    kind = position_kind or ("image" if config["adapter"]["steer_target"] == "text_to_image" else "text")
    groups = {}
    for index, record in enumerate(records):
        concept = config["task"] if config["task"] in {"i2p", "mscoco"} else record["concept"]
        groups.setdefault(concept_key(concept), []).append(index)
    for concept, indices in groups.items():
        path = Path(root) / slug(concept) / "scores.pt"
        state = torch.load(path, weights_only=True) if path.exists() else {"sample_ids": []}
        seen = set(state["sample_ids"])
        indices = [index for index in indices if records[index]["sample_id"] not in seen]
        if not indices:
            continue
        selected = scores[:, :, :, indices, :]
        maps = selected.mean(dim=3)
        if kind == "image":
            maps = maps.reshape(*maps.shape[:-1], *spatial_shape)
        gate = selected.clamp_min(0) if config["adapter"]["clipping"] else selected
        moments = {"score_mean": maps, "gate_mean": gate.mean(dim=(3, 4)),
                   "absolute_score_mean": selected.abs().mean(dim=(3, 4)),
                   "positive_fraction": (selected > 0).float().mean(dim=(3, 4))}
        old_count, added = len(state["sample_ids"]), len(indices)
        for name, value in moments.items():
            state[name] = (state[name] * old_count + value * added) / (old_count + added) if old_count else value
        state["sample_ids"].extend(records[index]["sample_id"] for index in indices)
        state.update(concept=concept, targets=config["targets"], timesteps=timesteps,
                     steer_target="pool_embedding" if kind == "pool" else config["adapter"]["steer_target"],
                     position_kind=kind, layer_ids=layer_ids if layer_ids is not None else list(range(scores.shape[2])),
                     axes="target, step, layer, " + ("y, x" if kind == "image" else "global" if kind == "pool" else "token"))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)


def plot_scores(directory, state, limit, gate_limit):
    """Show the full step/layer summary and spatial snapshots with a shared scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for index, target in enumerate(state["targets"]):
        scores = state["score_mean"][index]
        raw = scores.flatten(2).mean(dim=-1)
        spatial = scores.ndim == 4
        pooled = state.get("position_kind") == "pool"
        position_name = "global embedding" if pooled else "text tokens"
        position_mean = scores.mean(dim=(0, 1)) if spatial else scores.mean(dim=1)
        panels = [(raw, "Raw dot product", "layer", "denoising step", "RdBu_r", -limit, limit),
                  (state["gate_mean"][index], "Applied gate", "layer", "denoising step", "RdBu_r", -gate_limit, gate_limit),
                  (state["positive_fraction"][index], "Fraction of scores > 0", "layer", "denoising step", "viridis", 0, 1),
                  (position_mean, "Mean position scores", "patch x" if spatial else "global embedding" if pooled else "token position",
                   "patch y" if spatial else "denoising step", "RdBu_r", -limit, limit)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        for ax, (values, title, xlabel, ylabel, cmap, low, high) in zip(axes.flat, panels):
            im = ax.imshow(values.numpy(), cmap=cmap, vmin=low, vmax=high, aspect="auto", interpolation="nearest")
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"{state['concept']} | direction: {target} | n={len(state['sample_ids'])}")
        fig.savefig(directory / f"{slug(target)}-overview.png", dpi=140)
        plt.close(fig)

        # The tensor retains all steps/layers; the preview selects evenly spaced sites.
        steps = torch.linspace(0, scores.shape[0] - 1, min(4, scores.shape[0])).long().tolist()
        layers = torch.linspace(0, scores.shape[1] - 1, min(4, scores.shape[1])).long().tolist()
        fig, axes = plt.subplots(len(steps), len(layers), squeeze=False,
                                 figsize=(3 * len(layers), 2.7 * len(steps)), layout="constrained")
        for row, step in enumerate(steps):
            for column, layer in enumerate(layers):
                values = scores[step, layer] if spatial else scores[step, layer].unsqueeze(0)
                im = axes[row, column].imshow(values.numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit,
                                              aspect="equal" if spatial else "auto", interpolation="nearest")
                layer_id = state.get("layer_ids", list(range(scores.shape[1])))[layer]
                axes[row, column].set_title(f"step {step}, t={state['timesteps'][step]:g}, layer {layer_id}")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Mean raw dot product")
        fig.suptitle(f"{state['concept']} | direction: {target} | {'image patches' if spatial else position_name}")
        fig.savefig(directory / f"{slug(target)}-positions.png", dpi=140)
        plt.close(fig)


def write_score_report(root, config):
    states = [(path.parent, torch.load(path, weights_only=True)) for path in sorted(Path(root).glob("*/scores.pt"))]
    limit = max(1e-8, max(state["score_mean"].abs().max().item() for _, state in states))
    gate_limit = max(1e-8, max(state["gate_mean"].abs().max().item() for _, state in states))
    summary, details = [], []
    for directory, state in states:
        means = state["score_mean"].flatten(3).mean(dim=-1)
        for index, target in enumerate(state["targets"]):
            common = {"concept": state["concept"], "steering_target": target, "n": len(state["sample_ids"]),
                      "site": state["steer_target"], "position_kind": state.get("position_kind")}
            summary.append({**common, "score_mean": means[index].mean().item(),
                **{name: state[name][index].mean().item() for name in
                   ("absolute_score_mean", "gate_mean", "positive_fraction")},
                "mean_update_coefficient": config["adapter"]["beta"] * state["gate_mean"][index].mean().item()})
            for step, timestep in enumerate(state["timesteps"]):
                for layer in range(means.shape[2]):
                    layer_id = state.get("layer_ids", list(range(means.shape[2])))[layer]
                    details.append({**common, "step": step, "timestep": timestep, "layer": layer_id,
                        "score_mean": means[index, step, layer].item(),
                        **{name: state[name][index, step, layer].item() for name in
                           ("absolute_score_mean", "gate_mean", "positive_fraction")}})
        plot_scores(directory, state, limit, gate_limit)
    write_csv(Path(root) / "concept-means.csv", summary)
    write_csv(Path(root) / "step-layer-means.csv", details)
    write_json(Path(root) / "summary.json", {"task": config["task"], "case_id": config.get("case_id"),
        "adapter": config["adapter"], "sampling": config["sampling"], "concepts": summary,
        "score": "Sum over channels of current output * steering direction, before clipping. Later targets see earlier edits.",
        "averaging": "Each sample has equal weight. scores.pt retains every step, layer and position; only samples are averaged.",
        "preview": "Shared color scales across evaluation concepts. positions.png shows up to four steps and four layers.",
        "scope": "Conditional CFG branch only. Images reused without generation add no new scores."}, portable=True)
    print(f"[score-trace] {root}", flush=True)
