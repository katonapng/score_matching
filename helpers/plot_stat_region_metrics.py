import os
import json
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict


def extract_region_str(region):
    """Convert region list to readable string like [-1,0.5]x[-0.5,1]"""
    return f"[{region[0][0]},{region[0][1]}]x[{region[1][0]},{region[1][1]}]"


def load_results_from_all_subdirs(parent_directory):
    """Load and aggregate results from all subdirectories in the given directory"""
    aggregated_metrics = defaultdict(list)

    for subdir in os.listdir(parent_directory):
        subdir_path = os.path.join(parent_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for filename in sorted(os.listdir(subdir_path)):
            if filename.startswith("results_region") and filename.endswith(".json"):
                path = os.path.join(subdir_path, filename)
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

                aggregated_metrics[region_str].append(flat_metrics)

    # Average the metrics for each region
    averaged_metrics_by_region = {}
    for region, entries in aggregated_metrics.items():
        avg_metrics = {}
        keys = entries[0].keys()
        for key in keys:
            valid_values = [e[key] for e in entries if e[key] is not None]
            avg_metrics[key] = sum(valid_values) / len(valid_values) if valid_values else None
        averaged_metrics_by_region[region] = avg_metrics

    return averaged_metrics_by_region


def plot_grouped_metrics(metrics_by_region, save_dir, sort_by="SMD"):
    primary_metrics = ["SMD", "MAE", "MaxAE"]
    intensity_metrics = ["Intensity Min", "Intensity Mean", "Intensity Max"]

    filtered = [
        (region, data)
        for region, data in metrics_by_region.items()
        if data.get(sort_by) is not None
    ]
    filtered.sort(key=lambda x: x[1][sort_by])
    region_labels = [region for region, _ in filtered]
    sorted_data = [data for _, data in filtered]

    def plot_metrics(metric_list, filename_suffix):
        plt.figure(figsize=(12, 6))
        x = range(len(region_labels))

        for metric in metric_list:
            values = [data.get(metric) for data in sorted_data]
            if any(v is not None for v in values):
                line, = plt.plot(x, values, marker='o', linestyle='-', label=metric)
                for i, val in enumerate(values):
                    if val is not None:
                        plt.text(
                            i, val, f"{val:.3f}",
                            ha='center', va='bottom',
                            fontsize=10, color=line.get_color(),
                        )

        plt.xticks(x, region_labels, rotation=45, ha='right')
        plt.legend()
        plt.title(f"Metrics across Regions (sorted by {sort_by})")
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(save_dir, f"region_plot_{filename_suffix}.png")
        plt.savefig(filename)
        print(f"✅ Saved plot: {filename}")
        plt.close()

    plot_metrics(primary_metrics, "primary")
    plot_metrics(intensity_metrics, "intensity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot aggregated metrics across regions from all subdirectories."
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Parent directory containing subdirectories with results_region*.json files"
    )
    parser.add_argument(
        "--sort-by", type=str, default="SMD",
        help="Metric to sort regions by (default: SMD)"
    )
    args = parser.parse_args()

    metrics_by_region = load_results_from_all_subdirs(args.dir)
    if not metrics_by_region:
        print("⚠️ No valid region result files found in any subdirectory.")
    else:
        plot_grouped_metrics(metrics_by_region, args.dir, args.sort_by)
