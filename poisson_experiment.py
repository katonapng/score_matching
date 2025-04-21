import argparse
import json
import os
import warnings

from metrics import compute_mse_r2, plot_results
from models import Poisson_NN, optimize_nn
from utils import generate_training_data_poisson
from weight_functions import (distance_window, distance_window_derivative,
                              gaussian_window, gaussian_window_derivative)


def read_args_from_file(file_path, default_args):
    with open(file_path, "r") as f:
        args_dict = json.load(f)

    # Override defaults with config values
    for key, value in args_dict.items():
        setattr(default_args, key, value)

    return default_args


def check_file_existence(output_json, output_image):
    if os.path.exists(output_json) or os.path.exists(output_image):
        response = input(
            (
                f"Output files '{output_json}' "
                f"or '{output_image}' already exist. "
                "Overwrite? (y/n): "
            )
            )
        if response.lower() != 'y':
            print("Aborting execution.")
            exit()


def generate_output_filenames(args):
    base_path = "results/1d/" if args.dimensions == 1 else "results/2d/"

    if args.weight_function is not None:
        weight_path = f"weighting_{args.weight_function}"
    else:
        weight_path = "no_weighting/"

    if args.folder_suffix:
        weight_path += f"_{args.folder_suffix}"
    weight_path += "/"

    # Create directory if it doesn't exist
    full_path = base_path + weight_path
    os.makedirs(full_path, exist_ok=True)

    # Ensure region suffix formatting
    region_suffix = f"_region{args.region}".replace(" ", "")

    # Generate output filenames
    output_json = f"{full_path}results{region_suffix}.json"
    output_image = f"{full_path}output_plot{region_suffix}.png"

    return output_json, output_image


def get_region_dimension(region):
    if isinstance(region[0], list):
        return len(region)
    else:
        return 1


def main(args):
    args.dimensions = get_region_dimension(args.region)
    args.output_json, args.output_image = generate_output_filenames(args)
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

    train, test = generate_training_data_poisson(args)

    # Train model with progress tracking
    print("Starting training...")
    model, _ = optimize_nn(
        args=args,
        loader_train=train,
        nn_model=Poisson_NN,
    )

    # Generate and save plot
    print("Generating plot...")
    plot_results(args, model, test)
    print("Plot saved to", args.output_image)

    # Compute metrics
    print("Computing metrics...")
    mse, r2, smd = compute_mse_r2(test, model, args)

    # Save metrics and parameters to JSON file
    if args.weight_function is not None:
        args.weight_function = args.weight_function.__name__
        args.weight_derivative = args.weight_derivative.__name__
    output_data = {
        "parameters": vars(args),
        "metrics": {"smd": smd, "r2": r2, "mse": mse}
    }
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print("Metrics saved to", args.output_json)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON file with arguments",
    )
    parser.add_argument(
        "--region",
        type=json.loads,
        help="Region of the domain: [0,3] for 1D or [[0,1], [0,1]] for 2D",
    )
    parser.add_argument(
        "--weight_function",
        default=None,
        choices=["gaussian", "distance"],
        type=str,
        help="Weight function to use: ['gaussian', 'distance']",
    )
    parser.add_argument(
        "--dist_params",
        type=str,
        help="JSON string of parameters for modeled distribution",
    )
    parser.add_argument(
        "--folder_suffix",
        default="",
        type=str,
        help="Suffix for the output folder name",
    )
    parser.add_argument(
        "--mirror_boundary",
        default=True,
        type=str,
        help="Use mirrored boundary for training data generation",
    )
    parser.add_argument(
        "--num_samples",
        default=100,
        type=int,
        help="Number of samples generated for training",
    )
    parser.add_argument(
        "--hidden_dims",
        default=[8],
        type=list,
        help="List of hidden dimensions for tanh layers for the neural network",
    )
    parser.add_argument(
        "--percent",
        default=10.0,
        type=float,
        help="Percent of the domain to be used for distance weighting",
    )
    
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--output_json", default="result.json", type=str)
    parser.add_argument("--output_image", default="output_plot.png", type=str)

    args = parser.parse_args()

    if args.config:
        args = read_args_from_file(args.config, args)

    main(args)
