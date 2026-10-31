import argparse
import os
import re
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


def rename_scenario(name):

    scenario_base = None

    # Inversion
    if "token_embed_inversion" in name:
        scenario_base = "CCE"
    elif "prompt_embed_inversion" in name:
        scenario_base = "PEA"

    # Custom nudity datasets
    elif name.startswith("simple_unsafe_nudity-gs=7"):
        scenario_base = "Simple"

    elif name == "multilingual_nudity":
        scenario_base = "Multilingual"

    # Custom object datasets
    elif name == "simple_imagenette":
        scenario_base = "IMAGENETTE"

    # Existing benchmarks
    elif name.startswith("i2p"):
        scenario_base = "I2P"
    elif name.startswith("rab"):
        scenario_base = "RAB"
    elif name.startswith("t2i-rp"):
        scenario_base = "T2I-RP"

    # Utility
    elif name.startswith("coco"):
        scenario_base = "COCO"

    if scenario_base is not None:

        # Capture n<NUMBER> → suffix _n<NUMBER>
        match = re.search(r"n(\d+)", name)
        if match:
            scenario_base += f"_n{match.group(1)}"

        # Capture trigger<ANYTHING> → suffix _trigger<ANYTHING>
        match = re.search(r"trigger<([^>]+)>", name)
        if match:
            trigger_value = match.group(1)  # contents inside <>
            scenario_base += f"_trigger<{trigger_value}>"

    scenario_base = scenario_base or name

    return scenario_base


metric_rename_map = {
    "safety_rate": "Safety",
    "clip_score": "CLIP",
    "psnr": "PSNR",
    "lpips": "LPIPS",
    "ssim": "SSIM",
    "mae": "MAE",
    "mse": "MSE",
    "target_acc": "Target",
    "other_acc": "Other",
}

metric_ranges = {
    "safety_rate": (0, 1),
    "attack_success_rate": (0, 1),
    "clip_score": (0.275, 0.33),  # diverging
    "lpips": (0, 0.5),  # typically 0 (best) to ~1 (worst)
    "psnr": (0, 50),  # typical PSNR range (higher better)
    "mse": (0, None),  # 0 to max in data
    "mae": (0, None),  # 0 to max in data
    "target_acc": (0, 1),  # lower better
    "other_acc": (0, 1),  # higher better
}

metric_colormaps = {
    "safety_rate": "RdYlGn",  # higher better (0-1)
    "attack_success_rate": "RdYlGn",  # higher better (0-1)
    "clip_score": "RdYlGn",  # diverging, centered 0 (-1 to 1)
    "lpips": "RdYlGn_r",  # lower better
    "psnr": "Greens",  # higher better
    "mse": "RdYlGn_r",  # lower better (reverse RdYlGn)
    "mae": "RdYlGn_r",  # lower better
    "target_acc": "RdYlGn_r",  # lower better
    "other_acc": "RdYlGn",  # higher better
}

scenario_color_map = {
    "Simple": "#2ca02c",    # green
    "I2P": "#1f77b4",       # blue
    "RAB": "orange",       # orange
    "CCE": "red",       # red
    "PEA": "#9467bd",       # purple
    "COCO": "gray",         # gray
    # Add all expected scenario names here
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clean, consistent-width multi-metric heatmap.")
    parser.add_argument('--exp_name', '-e', required=True, help='Experiment name to derive folder of log CSV files')
    parser.add_argument('--metrics', '-m', nargs='+', required=True, help='List of metric names to visualize')
    parser.add_argument('--exclude', nargs='+', default=None, help='List of keywords to exclude from runs')
    parser.add_argument('--filter', nargs='+', default=None, help='List of keywords to filter runs')
    return parser.parse_args()


def extract_group_and_scenario(path):
    parts = path.strip("/").split("/")
    model_type = parts[-3] if len(parts) >= 3 else "unknown"
    group = parts[-2] if len(parts) >= 2 else "unknown"
    scenario = parts[-1]
    return model_type, group, scenario


def extract_iteration(group_name: str) -> int:
    """
    Extracts the number inside parentheses in the group name, e.g. '[..._(1000)]' -> 1000.
    Returns 0 if no number found.
    """
    import re
    match = re.search(r"\((\d+)\)", group_name)
    return int(match.group(1)) if match else 0


def sort_groups(groups):
    non_operator = [g for g in groups if not g.startswith("[")]

    def operator_priority(g):
        # Prioritize [standard...] groups first, then by iteration number, then by full name
        is_identity = 0 if g.startswith("[identity") else 1
        is_standard = 0 if g.startswith("[standard") else 1
        iteration = extract_iteration(g)
        name_without_iteration = re.sub(r"\(\d+\)", "", g)
        return is_identity, is_standard, name_without_iteration, iteration

    operator = sorted([g for g in groups if g.startswith("[")], key=operator_priority)
    return sorted(non_operator) + operator


def sort_scenarios(scenarios):
    preferred_prefixes = ["Simple", "I2P", "CCE", "PEA"]

    # Bucketize scenarios by the first matching prefix
    buckets = {prefix: [] for prefix in preferred_prefixes}
    remaining = []

    for s in scenarios:
        matched = False
        for prefix in preferred_prefixes:
            if s.startswith(prefix):
                buckets[prefix].append(s)
                matched = True
                break
        if not matched:
            remaining.append(s)

    # Preserve preferred prefix order + alphabetize remaining
    sorted_remaining = sorted(remaining)

    # Flatten in prefix order
    result = []
    for prefix in preferred_prefixes:
        result.extend(buckets[prefix])

    return result + sorted_remaining


def load_all_csvs(eval_log_dir):
    all_dataframes = []
    for filename in os.listdir(eval_log_dir):
        if filename.endswith(".csv"):
            path = os.path.join(eval_log_dir, filename)
            df = pd.read_csv(path)
            df[['model_type', 'group', 'scenario']] = df['image_folder'].apply(lambda x: pd.Series(extract_group_and_scenario(x)))
            df['scenario'] = df['scenario'].apply(rename_scenario)
            df.loc[df['model_type'] == 'original', 'model_type'] = df['group']
            all_dataframes.append(df)
    if not all_dataframes:
        raise ValueError("No CSV files found in the specified directory.")
    return pd.concat(all_dataframes, ignore_index=True).drop_duplicates()


def create_and_save_table(
        exp_name: str,
        metrics: List[str] = None,
        exclude_keywords: List[str] = None,
        filter_keywords: List[str] = None,
        suffix: str = None,
        hide_row_labels: bool = True
):
    eval_log_dir = os.path.join(os.getenv("OUTPUT_DIR"), "metrics", exp_name)

    df = load_all_csvs(eval_log_dir)

    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metrics not found: {missing_metrics}. Available: {list(df.columns)}")

    records = []
    for _, row in df.iterrows():

        # Exclude rows containing certain keywords
        if exclude_keywords and any(e in row["group"] for e in exclude_keywords):
            continue

        # Filter rows based on keywords
        if filter_keywords and any(f not in row["group"] for f in filter_keywords):
            continue

        for metric in metrics:
            value = row.get(metric, None)
            records.append({
                "group": row["group"],
                "scenario": row["scenario"],
                "metric": metric,
                "value": value
            })
    long_df = pd.DataFrame(records)

    unique_groups = sort_groups(long_df["group"].unique())

    # Prepare widths proportional to the number of scenarios per metric
    scenario_counts = []
    for metric in metrics:
        metric_df = long_df[long_df["metric"] == metric]
        pivot = metric_df.pivot_table(
            index="group",
            columns="scenario",
            values="value",
            aggfunc="last"
        )
        pivot = pivot.reindex(index=unique_groups)
        scenario_counts.append(len(pivot.columns))

    # Avoid zero-width by setting min width=1
    width_ratios = [max(1, c) for c in scenario_counts]

    # Setup figure and GridSpec with width_ratios
    fig_height = max(5, int(0.5 * len(unique_groups)))

    fig_width = sum(width_ratios) * 1.0  # 1.0 inch per scenario approx

    if not hide_row_labels:
        fig_width *= 4

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, len(metrics), width_ratios=width_ratios, figure=fig)

    list_of_metric_pivots = []

    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        metric_df = long_df[long_df["metric"] == metric]
        pivot = metric_df.pivot_table(
            index="group",
            columns="scenario",
            values="value",
            aggfunc="first"
        )

        # Append for later concatenation
        list_of_metric_pivots.append((metric, pivot))

        pivot = pivot.reindex(index=unique_groups)#.dropna(axis=1, how='all')
        ordered_scenarios = [s for s in sort_scenarios(pivot.columns.tolist()) if s in pivot.columns]
        pivot = pivot[ordered_scenarios]

        vmin_default, vmax_default = metric_ranges.get(metric, (None, None))
        data_min = pivot.min().min()
        data_max = pivot.max().max()

        vmin = vmin_default if vmin_default is not None else data_min
        vmax = vmax_default if vmax_default is not None else data_max

        # For diverging colormaps (like clip_score), set center=0
        center = 0 if metric in ["clip_score"] else None

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f" if metric in ["safety_rate"] else ".3f",
            cmap=metric_colormaps.get(metric, "RdYlGn"),
            vmin=vmin,
            vmax=vmax,
            # center=center,
            cbar=False,
            xticklabels=pivot.columns,
            yticklabels=(i == 0),  # Show group labels only on first subplot
            ax=ax,
            linewidths=0.5,
            annot_kws={"fontsize": 8}
        )

        display_name = metric_rename_map.get(metric, metric)  # fallback to original if not found
        ax.set_title(display_name)

        ax.set_xlabel("")
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_xticklabels(pivot.columns)

        ax.set_ylabel("")

        if hide_row_labels:
            ax.set_yticks([])  # Remove y-axis ticks on non-first plots

        if i != 0:
            ax.set_yticklabels([])  # Remove y-axis labels on non-first plots

    # plt.subplots_adjust(left=0.5)  # increase left margin (default is around 0.1)
    modified_pivots_to_combine = []
    for metric, pivot in list_of_metric_pivots:
        # Rename columns to include metric name
        pivot.columns = [f"{metric_rename_map.get(metric, metric)}_{col}" for col in pivot.columns]
        modified_pivots_to_combine.append(pivot)

    combined_df = pd.concat(modified_pivots_to_combine, axis=1)

    # Optional: sort rows by your group order again, or fill missing values
    combined_df = combined_df.reindex(unique_groups)

    output_dir_path = os.path.join(os.getenv("OUTPUT_DIR"), "tables", exp_name)

    if suffix is None:
        suffix = ""
        if exclude_keywords:
            suffix += "exclude_" + "_".join(exclude_keywords)

    os.makedirs(output_dir_path, exist_ok=True)

    combined_df.to_csv(os.path.join(output_dir_path, f'table_{suffix}.csv'))
    print(combined_df.to_string())

    plt.tight_layout()
    heatmap_output_path = os.path.join(output_dir_path, f'heatmap_{suffix}.png')
    plt.savefig(heatmap_output_path)
    print(f"Saved heatmap to {heatmap_output_path}")

    return heatmap_output_path


# Extract chain info
def extract_chain_from_group(group):
    group = re.sub(r"_\(\d+\)", "", group)
    parts = os.path.normpath(group).split('->')
    operators = list(filter(lambda x: x.startswith('['), parts))
    return '->'.join(['base'] + operators)


def create_and_save_lineplots_per_combination(
        exp_name: str,
        metrics: List[str],
        exclude_keywords: List[str] = None,
        filter_keywords: List[str] = None,
        include_distil: bool = False
) -> List[str]:
    """
    Creates and saves separate line plots showing how each metric evolves over finetuning iterations
    for each scenario-metric combination with non-empty data.
    Also creates one aggregated plot per metric across all scenarios.
    Returns a list of paths of saved figures.

    Note: This function assumes that there is only one operator with this iteration information in the name!
    """
    eval_log_dir = os.path.join(os.getenv("OUTPUT_DIR"), "metrics", exp_name)
    df = load_all_csvs(eval_log_dir)

    # Apply exclusion and filtering
    if exclude_keywords:
        df = df[~df['group'].apply(lambda g: any(e in g for e in exclude_keywords))]
    if filter_keywords:
        df = df[df['group'].apply(lambda g: all(f in g for f in filter_keywords))]

    df["chain"] = df["group"].apply(extract_chain_from_group)
    df['iteration'] = df['group'].apply(extract_iteration)
    df['chain_pos'] = df["chain"].apply(lambda x: len(x.split('->')) - 1)

    def compute_longest_common_prefixes(chains: List[str]) -> List[str]:

        def common_token_prefix(s1: str, s2: str) -> str:
            tokens1 = s1.split("->")
            tokens2 = s2.split("->")
            prefix = []
            for t1, t2 in zip(tokens1, tokens2):
                if t1 == t2:
                    prefix.append(t1)
                else:
                    break
            return "->".join(prefix)

        prefixes = []
        for i, chain in enumerate(chains):
            best_prefix = ""
            for j, other_chain in enumerate(chains):
                if i == j :
                    continue
                prefix = common_token_prefix(chain, other_chain)
                if len(prefix) > len(best_prefix):
                    best_prefix = prefix
            prefixes.append(best_prefix)
        return prefixes

    # Compute longest common prefix with at least one other row
    df['longest_prefix'] = compute_longest_common_prefixes(df["chain"].tolist())
    df['color_field'] = df['longest_prefix'].apply(lambda x: '->'.join(x.split('->')[:-1]))


    # Melt metrics for plotting
    plot_df = df.melt(
        id_vars=['longest_prefix', 'color_field', 'chain', 'scenario', 'iteration', 'chain_pos'],
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )
    plot_df['metric_name'] = plot_df['metric'].map(metric_rename_map).fillna(plot_df['metric'])
    saved_paths = []
    output_dir = os.path.join(os.getenv("OUTPUT_DIR"), "plots", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    unique_scenarios = plot_df['scenario'].unique()
    colors = sns.color_palette("tab10", len(unique_scenarios))
    scenario_color_map = dict(zip(unique_scenarios, colors))

    # === Individual scenario-metric plots ===
    combinations = plot_df[['scenario', 'metric']].drop_duplicates()

    for _, row in combinations.iterrows():
        scenario, metric = row['scenario'], row['metric']
        combo_df = plot_df[(plot_df['scenario'] == scenario) & (plot_df['metric'] == metric)]

        if combo_df.empty or combo_df['value'].dropna().empty:
            continue

        vmin_default, vmax_default = metric_ranges.get(metric, (None, None))
        data_min = combo_df['value'].min().min()
        data_max = combo_df['value'].max().max()

        vmin = vmin_default if vmin_default is not None else data_min
        vmax = vmax_default if vmax_default is not None else data_max

        fig, ax = plt.subplots(figsize=(8, 5))

        # Unique hue levels
        longest_prefixes = combo_df['longest_prefix'].unique()

        # Plot each hue level separately, but color by 'color_field'
        for i, longest_prefix in enumerate(longest_prefixes):

            # Filter rows where 'longest_prefix' is a prefix of this value
            subset = combo_df[combo_df['longest_prefix'].fillna('').apply(lambda p: longest_prefix.startswith(p))]

            sns.lineplot(
                data=subset,
                x='iteration',
                y='value',
                ax=ax,
            )

        ax.set_ylim(vmin * 0.975, vmax * 1.025)
        ax.set_title(f"{scenario} - {metric_rename_map.get(metric, metric)}")
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric_rename_map.get(metric, metric))

        scenario_safe = re.sub(r'[^\w\-]', '_', str(scenario))
        metric_safe = re.sub(r'[^\w\-]', '_', str(metric))
        out_path = os.path.join(output_dir, f"{scenario_safe}_{metric_safe}_over_ft_iterations.png")
        plt.savefig(out_path, dpi=256, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved line plot for {scenario} - {metric} to {out_path}")
        saved_paths.append(out_path)

    # === New section for base model plot ===
    for metric in metrics:

        metric_df = plot_df[plot_df['metric'] == metric]

        if metric_df.empty or metric_df['value'].dropna().empty:
            continue

        vmin_default, vmax_default = metric_ranges.get(metric, (None, None))
        data_min = metric_df['value'].min().min()
        data_max = metric_df['value'].max().max()

        vmin = vmin_default if vmin_default is not None else data_min
        vmax = vmax_default if vmax_default is not None else data_max

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.lineplot(
            data=metric_df,
            x='iteration',
            y='value',
            hue='longest_prefix',
            marker='o',
            ax=ax
        )

        ax.set_ylim(vmin * 0.975, vmax * 1.1025)
        ax.set_title(f"All Scenarios - {metric_rename_map.get(metric, metric)}")
        ax.set_xlabel('Iterations')
        ax.set_ylabel(metric_rename_map.get(metric, metric))
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            title="Chain / Scenario"
        )

        metric_safe = re.sub(r'[^\w\-]', '_', str(metric))
        out_path = os.path.join(output_dir, f"BASE_ALLSCENARIOS_{metric_safe}_aggregated.png")
        plt.savefig(out_path, dpi=256, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved aggregated plot for metric {metric} to {out_path}")
        saved_paths.append(out_path)

    return saved_paths


if __name__ == "__main__":
    args = parse_args()

    create_and_save_table(
        exp_name=args.exp_name,
        metrics=args.metrics,
        exclude_keywords=args.exclude,
        filter_keywords=args.filter,
        hide_row_labels=False
    )

    create_and_save_lineplots_per_combination(
        exp_name=args.exp_name,
        metrics=args.metrics,
        exclude_keywords=args.exclude,
        filter_keywords=args.filter
    )