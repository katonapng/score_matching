import copy
import warnings
from functools import partial

import numpy as np
import optuna

from metrics import calculate_metrics
from models import Poisson_SM, optimize_nn
from shared_args import get_shared_parser
from utils import generate_training_data_poisson, read_args_from_file
from weight_functions import (distance_window, distance_window_derivative,
                              gaussian_window, gaussian_window_derivative)


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
            return Poisson_SM(
                mod_args, input_dim=input_dim, hidden_dims=hidden_dims,
            )

        model, _, _ = optimize_nn(
            loader_train=train,
            loader_val=val,
            nn_model=model_fn,
            args=trial_args,
            trial=trial,
        )

        smd, _, _ = calculate_metrics(test, model, trial_args)
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

    parser = get_shared_parser()
    args = parser.parse_args()

    if args.config:
        args = read_args_from_file(args.config, args)

    main(args)
