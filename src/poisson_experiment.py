import json
import warnings

from src.metrics import calculate_metrics, plot_losses, plot_results
from src.models import Poisson_MLE, Poisson_SM, optimize_nn
from src.shared_args import get_shared_parser
from src.utils import (check_file_existence, convert_to_native,
                       generate_output_filenames, generate_training_data_poisson,
                       get_region_dimension, read_args_from_file)
from src.weight_functions import (distance_window, distance_window_derivative,
                              gaussian_window, gaussian_window_derivative)


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

    if args.model == "Poisson_SM":
        model_class = Poisson_SM
    elif args.model == "Poisson_MLE":
        model_class = Poisson_MLE
        if args.weight_function is not None:
            raise ValueError(
                "Weight function not supported for Poisson_MLE model."
            )
    else:
        raise ValueError(
            "Invalid model type. Choose 'Poisson_SM' or 'Poisson_MLE'."
        )
    
    # Train model with progress tracking
    print("Starting training...")
    model, train_losses, val_losses, norm_squared, divergence, weight, \
        log_density, psi_x = optimize_nn(
            args=args,
            loader_train=train,
            loader_val=val,
            nn_model=model_class,
        )

    # Generate and save training loss and validation smd plot
    print("Generating training loss and validation loss plot...")
    plot_losses(
        train_losses, val_losses, norm_squared, divergence, weight,
        log_density, psi_x, args,
    )
    print("Plot saved to", args.loss_image)

    # Generate and save plot
    print("Generating plot...")
    plot_results(args, model, test)
    print("Plot saved to", args.output_image)

    # Compute metrics
    print("Computing metrics...")
    avg_smd, avg_mae, avg_intensity_stats = calculate_metrics(test, model, args)

    # Save metrics and parameters to JSON file
    if args.weight_function is not None:
        args.weight_function = args.weight_function.__name__
        args.weight_derivative = args.weight_derivative.__name__
    output_data = {
        "parameters": vars(args),
        "metrics": {
            "SMD": avg_smd,
            "MAE": avg_mae,
            "intensity_stats": avg_intensity_stats}
    }
    with open(args.output_json, "w") as f:
        json.dump(convert_to_native(output_data), f, indent=4)

    print("Metrics saved to", args.output_json)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = get_shared_parser()
    args = parser.parse_args()

    if args.config:
        args = read_args_from_file(args.config, args)

    main(args)
