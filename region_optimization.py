import argparse
import copy
import json
import warnings
from functools import partial

import numpy as np
import optuna

from metrics import compute_smd
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


def objective(trial, args):
    smd_scores = []

    for region in args.regions:
        def get_region_dimension(region):
            if isinstance(region[0], list):
                return len(region)
            else:
                return 1

        args.dimensions = get_region_dimension(region)
        trial_args = copy.deepcopy(args)
        trial_args.region = region
        train, val, test = generate_training_data_poisson(trial_args)

        def model_fn(mod_args, input_dim, hidden_dims):
            return Poisson_NN(
                mod_args, input_dim=input_dim, hidden_dims=hidden_dims,
            )

        model, _, _ = optimize_nn(
            loader_train=train,
            loader_val=val,
            nn_model=model_fn,
            args=trial_args,
            trial=trial,
        )

        smd, _ = compute_smd(test, model, trial_args)
        smd_scores.append(smd)

    # Return the average SMD across all regions (can be weighted if needed)
    return np.mean(smd_scores)


def main(args):
    if args.weight_function == "gaussian":
        args.weight_function = gaussian_window
        args.weight_derivative = gaussian_window_derivative
    elif args.weight_function == "distance":
        args.weight_function = distance_window
        args.weight_derivative = distance_window_derivative
    else:
        args.weight_function = None
        args.weight_derivative = None

    # Minimazing SMD metric
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )
    study.optimize(
        partial(objective, args=args), n_trials=args.n_trials, n_jobs=5,
    )

    print("Best trial:")
    print(study.best_trial)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON file with arguments",
    )
    parser.add_argument("--dimensions", type=int, choices=[1, 2])

    parser.add_argument("--weighting", default=False, type=bool)
    parser.add_argument(
        "--weight_function",
        default=None,
        choices=["gaussian", "distance"],
        help="Weight function to use: ['gaussian', 'distance']",
        type=str
    )
    parser.add_argument(
        "--dist_params",
        type=str,
        help="JSON string of parameters",
    )
    parser.add_argument("--n_trials", default=30, type=int)
    parser.add_argument("--mirror_boundary", default=True, type=str)
    parser.add_argument("--weight_derivative", default=None, type=str)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--plot_gradients", default=False, type=str)
    parser.add_argument("--l2_regularization", default=True, type=str)
    parser.add_argument("--optuna", default=False, type=str)
    parser.add_argument(
        "--percent",
        default=10.0,
        type=float,
        help="Percent of the domain to be used for distance weighting",
    )

    args = parser.parse_args()

    if args.config:
        args = read_args_from_file(args.config, args)

    main(args)
