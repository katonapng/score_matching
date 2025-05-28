import os
import json
import matplotlib.pyplot as plt
import argparse

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

def plot_grouped_metrics(metrics_by_region, save_dir, sort_by="SMD"):
    primary_metrics = ["SMD", "MAE", "MaxAE"]
    intensity_metrics = ["Intensity Min", "Intensity Mean", "Intensity Max"]

    # Filter and sort regions by the sort_by metric
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
                plt.plot(x, values, marker='o', linestyle='-', label=metric)

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
    parser = argparse.ArgumentParser(description="Plot grouped metrics across region result files.")
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Directory containing results_region*.json files"
    )
    parser.add_argument(
        "--sort-by", type=str, default="SMD",
        help="Metric to sort regions by (default: SMD)"
    )
    args = parser.parse_args()

    metrics_by_region = load_results(args.dir)
    if not metrics_by_region:
        print("⚠️ No valid region result files found.")
    else:
        plot_grouped_metrics(metrics_by_region, args.dir, args.sort_by)
