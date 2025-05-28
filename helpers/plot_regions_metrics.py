import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def extract_region_str(region):
    """Convert region list to readable string like [-1,0.5]x[-0.5,1]"""
    return f"[{region[0][0]},{region[0][1]}]x[{region[1][0]},{region[1][1]}]"


def load_results(directory):
    files = sorted(
        f for f in os.listdir(directory)
        if f.startswith("results_region") and f.endswith(".json")
    )
    metrics_by_region = {}

    for file in files:
        path = os.path.join(directory, file)
        with open(path, 'r') as f:
            content = json.load(f)

        region = content["parameters"].get("region")
        region_str = extract_region_str(region)

        metrics = content.get("metrics", {})
        flat_metrics = {
            "SMD": metrics.get("SMD"),
            "MAE": metrics.get("MAE"),
            "MaxAE": metrics.get("MaxAE"),
            "Intensity Min": metrics.get("intensity_stats", {}).get("min"),
            "Intensity Mean": metrics.get("intensity_stats", {}).get("mean"),
            "Intensity Max": metrics.get("intensity_stats", {}).get("max"),
        }

        metrics_by_region[region_str] = flat_metrics

    return metrics_by_region


def plot_metrics(metrics_by_region, save_dir):
    sns.set_theme(style="whitegrid")

    metric_names = list(next(iter(metrics_by_region.values())).keys())
    palette = sns.color_palette("Set2")  # Pleasant color palette

    for metric in metric_names:
        filtered = [
            (region, data[metric])
            for region, data in metrics_by_region.items()
            if data.get(metric) is not None
        ]
        if not filtered:
            print(f"⚠️ No data available for metric: {metric}")
            continue

        filtered.sort(key=lambda x: x[1])
        region_labels = [region for region, _ in filtered]
        values = [val for _, val in filtered]

        plt.figure(figsize=(12, 6))
        x = range(len(region_labels))
        plt.plot(
            x, values, marker='o', linestyle='-', color=palette[0], alpha=0.9,
        )

        for i, val in enumerate(values):
            plt.text(
                i, val, f"{val:.3f}", ha='center', va='bottom', fontsize=9,
            )

        plt.xticks(x, region_labels, rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylabel(metric, fontsize=11)
        plt.title(f"{metric} across Regions", fontsize=13, weight='bold')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        metric_filename = f"{metric.replace(' ', '_')}_region_plot.png"
        filename = os.path.join(save_dir, metric_filename)
        plt.savefig(filename, dpi=150)
        print(f"✅ Saved plot: {filename}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot metrics across region result files."
        )
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing results_region*.json files"
    )
    args = parser.parse_args()

    metrics_by_region = load_results(args.dir)
    if not metrics_by_region:
        print("⚠️ No valid region result files found.")
    else:
        plot_metrics(metrics_by_region, args.dir)
