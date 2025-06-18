import argparse
import json


def get_shared_parser():
    """Create a shared argument parser for command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Poisson_SM",
        choices=["Poisson_SM", "Poisson_MLE"],
        help="Model to use: ['Poisson_SM', 'Poisson_MLE']",
    )
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
        help=("Weight function to use: ['gaussian', 'distance', "
              "'smooth_distance']"),
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
        help="List of hidden dimensions for tanh layers of the neural network",
    )
    parser.add_argument(
        "--percent",
        default=10.0,
        type=float,
        help="Percent of the domain to be used for distance weighting",
    )
    parser.add_argument(
        "--optimizer",
        default="rprop",
        choices=["adam", "rprop"],
        help="Optimizer to use: ['adam', 'rprop']",
    )
    parser.add_argument(
        "--alpha", 
        default=1e-3,
        type=float,
        help="Parameter for intensity penalty",    
    )
    parser.add_argument("--intensity_penalty", default=False, type=bool)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--plot_gradients", default=False, type=str)
    parser.add_argument("--plot_losses", default=False, type=str)
    parser.add_argument("--l2_regularization", default=True, type=str)
    parser.add_argument("--optuna", default=False, type=bool)

    return parser
