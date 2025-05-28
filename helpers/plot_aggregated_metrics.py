import json
import os
import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Plot metrics from summary files."
)
parser.add_argument(
    "--root_dir",
    type=str,
    required=True,
    help="Root directory containing experiment subfolders.",
)
args = parser.parse_args()

root_dir = args.root_dir
folders_data = {}
available_metrics = set()

print("Select folders to include in the plots:")
subfolders = sorted(
    [
        f for f in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, f))
    ]
)

for subfolder in subfolders:
    response = input(f"Include folder '{subfolder}'? [y/n]: ").strip().lower()
    if response != 'y':
        continue

    folder_path = os.path.join(root_dir, subfolder)
    agg_file = os.path.join(folder_path, "aggregated_metrics.json")

    if not os.path.exists(agg_file):
        print(f"⚠️ Skipping '{subfolder}' – No aggregated_metrics.json found.")
        continue

    with open(agg_file, 'r') as f:
        data = json.load(f)

    summary = data.get("summary", {})
    if not summary:
        print(f"⚠️ Skipping '{subfolder}' – No summary found.")
        continue

    mean_metrics = summary.get("mean_metrics", {})
    min_metrics = summary.get("min_metrics", {})
    max_metrics = summary.get("max_metrics", {})

    available_metrics.update(mean_metrics.keys())

    folders_data[subfolder] = {
        metric: {
            'mean': mean_metrics.get(metric),
            'min': min_metrics.get(metric, {}).get('value'),
            'max': max_metrics.get(metric, {}).get('value'),
        }
        for metric in mean_metrics
    }


def shorten_folder_name(folder_name):
    import re
    match = re.search(r"(\d+)l_(\d+)n", folder_name)
    layers_neurons = f"{match.group(1)}L-{match.group(2)}N" if match else ""

    percent_match = re.search(r"(\d+)%", folder_name)
    percent = f"_{percent_match.group(1)}%" if percent_match else ""

    if folder_name.startswith("no_weighting"):
        return f"{layers_neurons}{percent}"

    elif folder_name.startswith("weighting_"):
        parts = folder_name.split("_")
        if len(parts) > 1:
            weighting_type = parts[1]
            if "mirror" in root_dir:
                weighting_type += "_mirror"
            return f"{weighting_type}_{layers_neurons}{percent}"

    return folder_name


# Generate plot for each available metric
for metric in sorted(available_metrics):
    labels, means, mins, maxs = [], [], [], []

    for folder, metrics in folders_data.items():
        metric_data = metrics.get(metric)
        if metric_data:
            labels.append(shorten_folder_name(folder))
            means.append(metric_data['mean'])
            mins.append(metric_data['min'])
            maxs.append(metric_data['max'])

    if not labels:
        print(f"⚠️ No data for metric '{metric}', skipping plot.")
        continue

    combined = sorted(zip(labels, means, mins, maxs), key=lambda x: (x[1] is None, x[1]))
    labels, means, mins, maxs = zip(*combined)
    x = range(len(labels))

    plt.figure(figsize=(8, 6))
    plt.plot(x, means, label='Mean', marker='.')
    plt.plot(x, mins, label='Min', marker='.')
    plt.plot(x, maxs, label='Max', marker='.')

    for i in x:
        if means[i] is not None:
            plt.text(i, means[i], f"{means[i]:.2f}", ha='center', va='bottom', fontsize=6, color='darkblue')
        if mins[i] is not None:
            plt.text(i, mins[i], f"{mins[i]:.2f}", ha='center', va='top', fontsize=6, color='darkorange')
        if maxs[i] is not None:
            plt.text(i, maxs[i], f"{maxs[i]:.2f}", ha='center', va='bottom', fontsize=6, color='green')

    plt.xticks(x, labels, rotation=90)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} across Folders")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()

    filename = os.path.join(root_dir, f"{metric}_comparison_plot.png")
    plt.savefig(filename)
    print(f"✅ Saved plot: {filename}")
