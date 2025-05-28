import json
import time
import warnings

import psutil
from memory_profiler import memory_usage

from src.metrics import calculate_metrics
from src.models import Poisson_MLE, Poisson_SM, optimize_nn
from src.shared_args import get_shared_parser
from src.utils import (check_file_existence, convert_to_native,
                       generate_output_filenames,
                       generate_training_data_poisson, get_region_dimension,
                       read_args_from_file)
from src.weight_functions import (distance_window, distance_window_derivative,
                                  gaussian_window, gaussian_window_derivative)


def track_memory():
    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 ** 2)  # Convert to MB


def benchmark_model(
    ModelClass, name, loader_train, loader_val, loader_test, args
):
    print(f"\nBenchmarking {name}")
    start_time = time.time()

    # Function to wrap training
    def run_training():
        return optimize_nn(loader_train, loader_val, ModelClass, args)

    # Track peak memory during training
    peak_memory, result = memory_usage(
        run_training, retval=True, interval=0.1, max_usage=True
    )
    model, train_losses, val_losses, _, _, _, _, _ = result

    avg_smd, avg_mae, avg_maxae, _ = calculate_metrics(loader_test, model, args)

    end_time = time.time()
    time_per_epoch = (end_time - start_time) / len(train_losses)
    convergence_time = end_time - start_time

    print(f"{name} Results:")
    print(f"  - Epochs run: {len(train_losses)}")
    print(f"  - Score Matching Difference: {avg_smd:.4f}")
    print(f"  - Mean Absolute Error: {avg_mae:.4f}")
    print(f"  - Time per epoch: {time_per_epoch:.2f}s")
    print(f"  - Total time: {convergence_time:.2f}s")
    print(f"  - Peak memory usage: {peak_memory:.2f} MB")
    print(f"  - Final Val Loss: {val_losses[-1]:.6f}\n")

    return {
        'name': name,
        'epochs': len(train_losses),
        'time_per_epoch': time_per_epoch,
        'total_time': convergence_time,
        'memory_mb': peak_memory,
        'final_val_loss': val_losses[-1],
        'SMD': avg_smd,
        'MAE': avg_mae,
        'MaxAE': avg_maxae,
    }


def main(args):
    args.dimensions = get_region_dimension(args.region)
    args.output_json, args.output_image, args.gradient_dir, args.loss_image = \
        generate_output_filenames(args)
    check_file_existence(args.output_json, args.output_image)

    if args.weight_function == "gaussian":
        args.weight_function = gaussian_window
        args.weight_derivative = gaussian_window_derivative
    elif args.weight_function == "distance":
        args.weight_function = distance_window
        args.weight_derivative = distance_window_derivative
    else:
        args.weight_function = None
        args.weight_derivative = None

    train, val, test = generate_training_data_poisson(args)

    results = []
    model_classes = [
        (Poisson_SM, "Poisson_SM"),
        (Poisson_MLE, "Poisson_MLE")
    ]
    for ModelClass, name in model_classes:
        model_result = benchmark_model(
            ModelClass, name, train, val, test, args,
        )
        model_result["region"] = args.region  # Include region info
        results.append(model_result)

    # Optional: compare in table format
    print("\nSummary:")
    for res in results:
        print(f"{res['name']:<12} | Epochs: {res['epochs']:3d} | "
              f"Time/Epoch: {res['time_per_epoch']:.2f}s | "
              f"Total Time: {res['total_time']:.2f}s | "
              f"Memory: {res['memory_mb']:.2f}MB | "
              f"Val Loss: {res['final_val_loss']:.6f} | "
              f"SMD: {res['SMD']:.4f} | MAE: {res['MAE']:.4f}")

    # Save results to JSON
    with open(args.output_json, "w") as f:
        json.dump(convert_to_native(results), f, indent=4)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = get_shared_parser()
    args = parser.parse_args()

    if args.config:
        args = read_args_from_file(args.config, args)

    main(args)
