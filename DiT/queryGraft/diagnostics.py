"""Save per-step diagnostics and temporal heatmap summaries for each sample."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def save_step_diagnostics(directory, step, tensors, activation_sums, info):
    """Inputs are small CPU tensors; full activations never leave the sampling code.

    activation_sums: [pairs, edit_layers, 6], containing squared norms of
    before/after/delta, max absolute delta, channel-value count, and token count.
    RMS and relative L2 are measured only on each pair's actual winning region.
    """
    directory = Path(directory)
    figures, numeric = directory / "timesteps", directory / "tensors"
    figures.mkdir(parents=True, exist_ok=True)
    numeric.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory.parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(directory.parent / ".cache"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    counts = activation_sums[..., 4].clamp_min(1)
    rms = (activation_sums[..., :3] / counts[..., None]).sqrt()
    relative = activation_sums[..., 2].sqrt() / (activation_sums[..., 0].sqrt() + 1e-8)
    tensors.update(
        activation_before_rms=rms[..., 0], activation_after_rms=rms[..., 1],
        activation_delta_rms=rms[..., 2], activation_relative_l2=relative,
        activation_abs_max=activation_sums[..., 3],
        activation_value_count=activation_sums[..., 4].long(),
        activation_token_count=activation_sums[..., 5].long())
    save_file({name: value.contiguous() for name, value in tensors.items()},
              str(numeric / f"step_{step:04d}.safetensors"))

    pairs, editors = info["pairs"], tensors["edit_layer_indices"].tolist()
    figure = Figure(figsize=(16, 3.8 * len(pairs)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(pairs), 4, squeeze=False)
    aggregation = "layer vote" if info["my_layer_aggre"] else "soft-map mean"
    policy = (info["mask_rule"] if info["my_layer_policy"] else
              "CA spatial mean" if len(pairs) == 1 else "CA concept argmax")
    limits = f"eta={info['eta']:g}, rho={info['rho']:g}" if info["my_layer_aggre"] else f"rho={info['rho']:g}"
    figure.suptitle(
        f"queryGraft | step {step:03d} | model timestep {info['timestep']:g} | "
        f"aggregation={aggregation} | policy={policy} | "
        f"{limits} | {info['representation']} | update={info.get('update_rule', 'replace')}\n"
        f"Decision input: {len(info['localization_layers'])} selected blocks "
        f"of {len(info['diagnostic_layers'])} executed blocks; softmax={info['softmax_scope']}",
        fontsize=12)
    summaries = []
    for pair, definition in enumerate(pairs):
        applied = tensors["applied_mask"][pair]
        area = float(applied.float().mean())
        target, anchor = definition["target"], definition["anchor"]
        destination = "zero" if info.get("update_rule") == "zero" else anchor
        if info["my_layer_aggre"]:
            decision_input = tensors["vote_fraction"][pair]
            input_title = "Decision input\nLayer-vote fraction"
        else:
            decision_input = tensors["localization_scores_mean"][pair, ..., 0]
            input_title = "Decision input\nMean soft maps: target channel"
        titles = [f"{target} -> {destination}\nApplied mask: {area:.1%}",
                  input_title,
                  f"Candidate mask: {float(tensors['mask'][pair].float().mean()):.1%}\nBefore area gate / overlap selection"]
        maps = [applied, decision_input, tensors["mask"][pair]]
        for column, (data, title, cmap) in enumerate(zip(maps, titles, ("Blues", "magma", "Blues"))):
            axis = axes[pair, column]
            plot = axis.imshow(data.float().numpy(), vmin=0, vmax=1, cmap=cmap,
                               interpolation="nearest", origin="upper")
            axis.set(title=title, xticks=[], yticks=[])
            figure.colorbar(plot, ax=axis, shrink=0.78)
        if info["decision_threshold"] is not None:
            kind = "vote fraction" if info["my_layer_aggre"] else "target spatial mean"
            rule = f"Strict > {info['decision_threshold'][pair]:.6g} ({kind})"
            if info["region_policy"] == "mean_target_max":
                rule += ("\nPer layer: target > spatial mean\nAND target > anchor AND target > background"
                         if info["my_layer_aggre"] else "\nAND target > anchor AND target > background")
        elif info["region_policy"] == "concept_argmax":
            rule = f"Global argmax over {3 * len(pairs)} concepts; target index={3 * pair}"
        else:
            rule = ("target > anchor and background" if info["region_policy"] == "target_max" else
                    "anchor < target and background")
        axes[pair, 1].set_xlabel(rule)

        # Aggregate energies over the edited layers, rather than averaging their RMS values.
        totals = activation_sums[pair].sum(0)
        aggregate_rms = (totals[:3] / totals[4].clamp_min(1)).sqrt()
        change = float(totals[2].sqrt() / (totals[0].sqrt() + 1e-8))
        maximum = float(activation_sums[pair, :, 3].max())
        token_count = int(totals[5])
        layers = [{"layer": layer,
                   "before_rms": float(rms[pair, j, 0]),
                   "after_rms": float(rms[pair, j, 1]),
                   "delta_rms": float(rms[pair, j, 2]),
                   "relative_l2": float(relative[pair, j]),
                   "max_abs_delta": float(activation_sums[pair, j, 3]),
                   "token_count": int(activation_sums[pair, j, 5])}
                  for j, layer in enumerate(editors)]
        summaries.append({"target": target, "anchor": anchor,
                          "before_rms": float(aggregate_rms[0]),
                          "after_rms": float(aggregate_rms[1]),
                          "delta_rms": float(aggregate_rms[2]),
                          "relative_l2": change, "max_abs_delta": maximum,
                          "token_count": token_count, "layers": layers})

        axis = axes[pair, 3]
        values = aggregate_rms.tolist()
        bars = axis.bar(["Before", "After", "Change"], values,
                        color=["#627d98", "#2a9d8f", "#e76f51"])
        axis.bar_label(bars, labels=[f"{value:.3g}" for value in values], padding=3)
        axis.set(title="Activation RMS in applied region", ylabel="RMS",
                 ylim=(0, (max(values) or 1.0) * 1.28))
        axis.grid(axis="y", alpha=0.2)
        detail = (f"Relative L2 change: {change:.2%}\nMax |delta|: {maximum:.3g}\n"
                  f"{token_count} token-layer occurrences, {len(editors)} edit layers")
        if token_count == 0:
            detail = "No applied tokens; no activation update."
        axis.text(0.02, 0.98, detail, transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        axes[pair, 0].set_xlabel(
            f"candidate area={info['target_area'][pair]:.1%}; "
            f"gate={'on' if info['triggered'][pair] else 'off'}")

    figure.savefig(figures / f"step_{step:04d}.png", dpi=130)
    figure.clear()
    return summaries


def save_sample_diagnostics(directory, tensors, pairs, step_count, *, mask_rule="target_max",
                            my_layer_aggre=False, my_layer_policy=False,
                            update_rule="replace",
                            localization_layers=None, diagnostic_layers=None):
    """Summarize all sampling steps equally, including steps with no applied mask."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(directory)
    save_file({name: value.contiguous() for name, value in tensors.items()},
              str(directory / "summary.safetensors"))
    figure = Figure(figsize=(10, 3.8 * len(pairs)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(pairs), 2, squeeze=False)
    aggregation = "layer vote" if my_layer_aggre else "soft-map mean"
    policy = mask_rule if my_layer_policy else "CA spatial mean" if len(pairs) == 1 else "CA concept argmax"
    scope = (f"Soft scores use {len(localization_layers)} selected blocks "
             f"of {len(diagnostic_layers)} executed blocks\n"
             if localization_layers is not None else "")
    figure.suptitle(
        f"queryGraft | sample summary | mean over all {step_count} sampling steps | "
        f"aggregation={aggregation}, policy={policy}, update={update_rule}\n"
        f"{scope}"
        "Applied mask: time average, no new threshold; target score: selected-layer soft mean, then time mean",
        fontsize=12)
    for pair, definition in enumerate(pairs):
        applied = tensors["applied_mask_mean"][pair]
        scores = tensors["localization_scores_mean"][pair, ..., 0]
        destination = "zero" if update_rule == "zero" else definition["anchor"]
        titles = [f"{definition['target']} -> {destination}\nMean applied mask",
                  "Mean target soft score over selected layers and time" +
                  ("\nAuxiliary score; masks use layer votes" if my_layer_aggre else "")]
        for column, (data, title, cmap) in enumerate(zip((applied, scores), titles, ("Blues", "magma"))):
            axis = axes[pair, column]
            plot = axis.imshow(data.float().numpy(), vmin=0, vmax=1, cmap=cmap,
                               interpolation="nearest", origin="upper")
            axis.set(title=title, xticks=[], yticks=[])
            figure.colorbar(plot, ax=axis, shrink=0.78)
        axes[pair, 0].set_xlabel(f"Mean applied area per step: {float(applied.mean()):.1%}")
    figure.savefig(directory / "summary.png", dpi=130)
    figure.clear()
