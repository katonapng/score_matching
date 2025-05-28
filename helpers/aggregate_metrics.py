import argparse
import json
import os
from glob import glob
from statistics import mean


def safe_mean(values):
    clean_values = [v for v in values if v is not None]
    return mean(clean_values) if clean_values else None


def min_max_with_region(metrics, key):
    valid = [(m[key], m['region']) for m in metrics if key in m and m[key] is not None]
    if not valid:
        return None, None
    min_val, min_region = min(valid, key=lambda x: x[0])
    max_val, max_region = max(valid, key=lambda x: x[0])
    return (
        {'value': min_val, 'region': min_region},
        {'value': max_val, 'region': max_region}
    )


def flatten_metrics(metrics_dict):
    flat = {}
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                flat[f"{key}_{subkey}"] = subval
        else:
            flat[key] = value
    return flat


def main(results_dir):
    output_file = os.path.join(results_dir, "aggregated_metrics.json")
    json_files = glob(os.path.join(results_dir, "results_region*.json"))

    if not json_files:
        print(f"No result files found in {results_dir}")
        return

    all_metrics = []

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            region = data['parameters']['region']
            metrics = flatten_metrics(data['metrics'])
            metrics['region'] = region
            all_metrics.append(metrics)

    # Collect all metric keys except 'region'
    metric_keys = [k for k in all_metrics[0] if k != 'region']

    mean_metrics = {}
    min_metrics = {}
    max_metrics = {}

    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        mean_metrics[key] = safe_mean(values)
        min_val, max_val = min_max_with_region(all_metrics, key)
        min_metrics[key] = min_val
        max_metrics[key] = max_val

    summary = {
        'mean_metrics': mean_metrics,
        'min_metrics': min_metrics,
        'max_metrics': max_metrics
    }

    final_output = {
        'metrics_per_region': all_metrics,
        'summary': summary
    }

    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"Aggregated metrics saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate all scalar metrics from result JSON files."
    )
    parser.add_argument(
        "results_dir",
        help="Path to the directory containing results_region*.json files"
    )
    args = parser.parse_args()

    main(args.results_dir)

