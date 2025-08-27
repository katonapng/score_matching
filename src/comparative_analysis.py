import json
import time
import warnings

import psutil
from memory_profiler import memory_usage

from src.metrics import calculate_metrics, plot_results
from src.models import Poisson_MLE, Poisson_SM, optimize_nn
from src.shared_args import get_shared_parser
from src.utils import (check_file_existence, convert_to_native,
                       generate_output_filenames,
                       generate_training_data_poisson, get_region_dimension,
                       read_args_from_file)
from src.weight_functions import (distance_window, distance_window_derivative,
                                  gaussian_window, gaussian_window_derivative,
                                  smooth_distance_window,
                                  smooth_distance_window_derivative)




def benchmark_model(
    ModelClass, name, loader_train, loader_val, loader_test, args
):
    print(f"\nBenchmarking {name}")
    start_time = time.time()

    model, train_losses, val_losses = optimize_nn(loader_train, loader_val, ModelClass, args)
    end_time = time.time()

    avg_smd, avg_mae, avg_maxae, _ = calculate_metrics(loader_test, model, args)

    time_per_epoch = (end_time - start_time) / len(train_losses)
    convergence_time = end_time - start_time

    return {
        'name': name,
        'epochs': len(train_losses),
        'time_per_epoch': time_per_epoch,
        'total_time': convergence_time,
        'final_val_loss': val_losses[-1],
        'SMD': avg_smd,
        'MAE': avg_mae,
        'MaxAE': avg_maxae,
    }


def main(args):
    args.dimensions = get_region_dimension(args.region)
    workspace = args.workspace if hasattr(args, 'workspace') else None
    args.output_json, args.output_image, args.gradient_dir, args.loss_image = \
        generate_output_filenames(args, workspace)
    check_file_existence(args.output_json, args.output_image)

    if args.weight_function == "gaussian":
        args.weight_function = gaussian_window
        args.weight_derivative = gaussian_window_derivative
    elif args.weight_function == "distance":
        args.weight_function = distance_window
        args.weight_derivative = distance_window_derivative
    elif args.weight_function == "smooth_distance":
        args.weight_function = smooth_distance_window
        args.weight_derivative = smooth_distance_window_derivative
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
