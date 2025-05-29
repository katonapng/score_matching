import os
import json
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd
from collections import defaultdict

import re

def extract_common_job_id(parent_directory):
    subdirs = [name for name in os.listdir(parent_directory)
               if os.path.isdir(os.path.join(parent_directory, name))]
    job_ids = []
    for d in subdirs:
        match = re.search(r"(job_\d+)", d)
        if match:
            job_ids.append(match.group(1))

    unique_job_ids = set(job_ids)
    if len(unique_job_ids) == 1:
        return unique_job_ids.pop()
    elif len(unique_job_ids) > 1:
        print(f"⚠️ Multiple job IDs found: {unique_job_ids}. Using the first one.")
        return list(unique_job_ids)[0]
    else:
        print("⚠️ No job ID found in subdirectory names. Using 'default'.")
        return "default"


def extract_region_str(region):
    """Convert region list to readable string like [-1,0.5]x[-0.5,1]"""
    return f"[{region[0][0]},{region[0][1]}]x[{region[1][0]},{region[1][1]}]"


def load_results_recursively(parent_directory):
    """Recursively search for all results_region*.json files and aggregate metrics by region"""
    from collections import defaultdict
    aggregated_metrics = defaultdict(list)

    for root, _, files in os.walk(parent_directory):
        for filename in files:
            if filename.startswith("results_region") and filename.endswith(".json"):
                full_path = os.path.join(root, filename)
                with open(full_path, 'r') as f:
                    try:
                        content = json.load(f)
                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse JSON: {full_path}")
                        continue

                region = content.get("parameters", {}).get("region")
                if region is None:
                    print(f"⚠️ No region info found in: {full_path}")
                    continue

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

    # Average the metrics per region
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

    def make_dataframe(metric_list):
        rows = []
        for region, data in zip(region_labels, sorted_data):
            for metric in metric_list:
                value = data.get(metric)
                if value is not None:
                    rows.append({
                        "Region": region,
                        "Metric": metric,
                        "Value": value
                    })
        return pd.DataFrame(rows)

    def plot(df, filename_suffix):
        plt.figure(figsize=(14, 6))
        sns.set(style="whitegrid")
        ax = sns.lineplot(data=df, x="Region", y="Value", hue="Metric", marker='o')
        for line in ax.lines:
            for x, y in zip(line.get_xdata(), line.get_ydata()):
                ax.text(x=x, y=y + 0.01, s=f"{y:.3f}", ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Metrics across Regions (sorted by {sort_by})")
        plt.tight_layout()
        filename = os.path.join(save_dir, f"region_plot_{filename_suffix}.png")
        plt.savefig(filename)
        print(f"✅ Saved plot: {filename}")
        plt.close()

    df_primary = make_dataframe(primary_metrics)
    df_intensity = make_dataframe(intensity_metrics)

    plot(df_primary, "primary")
    plot(df_intensity, "intensity")

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

    metrics_by_region = load_results_recursively(args.dir)
    if not metrics_by_region:
        print("⚠️ No valid region result files found in any subdirectory.")
    else:
        job_id = extract_common_job_id(args.dir)
        parent_dir_name = os.path.basename(os.path.abspath(args.dir))
        parent_parent_dir = os.path.dirname(os.path.abspath(args.dir))
        save_dir = os.path.join(parent_parent_dir, "plots", job_id)
        
        os.makedirs(save_dir, exist_ok=True)
        plot_grouped_metrics(metrics_by_region, save_dir, args.sort_by)
