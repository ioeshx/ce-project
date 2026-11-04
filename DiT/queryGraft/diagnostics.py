"""Save one numeric trace and one four-panel diagnostic per sample and step."""
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
    directory.mkdir(parents=True, exist_ok=True)
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
              str(directory / f"step_{step:04d}.safetensors"))

    pairs, editors = info["pairs"], tensors["edit_layer_indices"].tolist()
    figure = Figure(figsize=(16, 3.8 * len(pairs)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(pairs), 4, squeeze=False)
    figure.suptitle(
        f"queryGraft | step {step:03d} | model timestep {info['timestep']:g} | "
        f"eta={info['eta']:g}, rho={info['rho']:g} | {info['representation']}\n"
        f"Heatmap averages: all {len(info['diagnostic_layers'])} executed blocks; "
        f"mask decision: {len(info['localization_layers'])} selected blocks",
        fontsize=12)
    summaries = []
    for pair, definition in enumerate(pairs):
        applied = tensors["applied_mask"][pair]
        area = float(applied.float().mean())
        target, anchor = definition["target"], definition["anchor"]
        titles = [f"{target} -> {anchor}\nApplied mask: {area:.1%}",
                  "Mean layer mask\nFraction of all blocks voting target",
                  "Mean target saliency\nLayer softmax, then layer average"]
        maps = [applied, tensors["layer_mask_mean"][pair],
                tensors["concept_scores_mean"][pair, ..., 0]]
        for column, (data, title, cmap) in enumerate(zip(maps, titles, ("Blues", "viridis", "magma"))):
            axis = axes[pair, column]
            plot = axis.imshow(data.float().numpy(), vmin=0, vmax=1, cmap=cmap,
                               interpolation="nearest", origin="upper")
            axis.set(title=title, xticks=[], yticks=[])
            figure.colorbar(plot, ax=axis, shrink=0.78)

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

    figure.savefig(directory / f"step_{step:04d}.png", dpi=130)
    figure.clear()
    return summaries
