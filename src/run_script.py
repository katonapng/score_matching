import argparse
import json
import os
import subprocess
import sys

venv_python = os.path.join(os.path.dirname(sys.executable), "python")

# Define reusable region values
# REGION_VALUES_2D = [
#     [[-1, 1], [-1, 1]],
#     [[-0.5, 0.5], [-0.5, 0.5]],
#     [[0.5, 1], [0.5, 1]],
#     [[0, 1], [0, 1]],
#     [[-0.3, 0.3], [-0.3, 0.3]],
#     [[-0.5, 1], [-0.5, 1]],
#     [[-1, 0.5], [-0.5, 1]],
#     [[-1, 0], [-1, 0]],
#     [[0.3, 1], [-1, -0.3]],
#     [[-0.8, -0.2], [0.2, 0.8]],
#     [[0.2, 0.8], [-0.8, -0.2]],
#     [[-1, -0.5], [-1, -0.5]],
# ]

# REGION_VALUES_2D = [
#     [[-1.0, 1.0], [-1.0, 1.0]],
#     [[-0.95, 0.95], [-0.95, 0.95]],
#     [[-0.9, 0.9], [-0.9, 0.9]],
#     [[-0.85, 0.85], [-0.85, 0.85]],
#     [[-0.8, 0.8], [-0.8, 0.8]],
#     [[-0.75, 0.75], [-0.75, 0.75]],
#     [[-0.7, 0.7], [-0.7, 0.7]],
#     [[-0.65, 0.65], [-0.65, 0.65]],
#     [[-0.6, 0.6], [-0.6, 0.6]],
#     [[-0.55, 0.55], [-0.55, 0.55]],
#     [[-0.5, 0.5], [-0.5, 0.5]]
# ]

# REGION_VALUES_1D = [
#     [-1, 1], [-0.3, 0.3], [-0.5, 0.5], [0, 1],
#     [-1, 0.5], [-1, 0], [-0.5, 1], [-1, 0.7], [-1, 0.3],
#     [-0.2, 0.2], [-0.7, 0.7], [-0.8, -0.2], [0.2, 0.8],
#     [0.5, 1], [0.3, 1], [-1, -0.5], [-1, -0.3], [-0.9, 0.9],
# ]

REGION_VALUES_1D = [
    [-1, 1], [-0.3, 0.3], [-0.5, 0.5], [0, 1],
    [-1, 0.5], [-1, 0], [-0.5, 1], [-1, 0.7], [-1, 0.3],
    [-0.2, 0.2], [-0.7, 0.7], [-0.8, -0.2], [0.2, 0.8],
    [0.5, 1], [0.3, 1], [-1, -0.5], [-1, -0.3], [-0.9, 0.9],
]

REGION_VALUES_2D = [
    [[-1, 1], [-1, 1]],
    [[-0.5, 0.5], [-0.5, 0.5]],
    [[0.5, 1], [0.5, 1]],
    [[0, 1], [0, 1]],
    [[-0.3, 0.3], [-0.3, 0.3]],
    [[-0.5, 1], [-0.5, 1]],
    [[-1, 0.5], [-0.5, 1]],
    [[-1, 0], [-1, 0]],
    [[0.3, 1], [-1, -0.3]],
    [[-0.8, -0.2], [0.2, 0.8]],
    [[0.2, 0.8], [-0.8, -0.2]],
    [[-1, -0.5], [-1, -0.5]],
]

# Shared base configs
BASE_CONFIGS = {
    "poisson_experiment": {
        "model": "Poisson_SM",
        # "model": "Poisson_MLE",
        "weight_function": "distance",
        "mirror_boundary": False,
        "dist_params": {"n_expected": 100, "scale": 0.5},
        "hidden_dims": [32],
        "patience": 20,
        "learning_rate": 1e-03,
        "plot_gradients": False,
        "plot_losses": True,
        "l2_regularization": True,
        "optimizer": "rprop",
        "epochs": 200,
        "intensity_penalty": False,
        "folder_suffix": "poisson_experiment",
    },
    "comparative_analysis": {
        "model": "Poisson_MLE",
        "weight_function": "distance",
        "mirror_boundary": False,
        "dist_params": {"n_expected": 100, "scale": 0.5},
        "hidden_dims": [32],
        "patience": 20,
        "learning_rate": 1e-03,
        "plot_gradients": False,
        "plot_losses": False,
        "l2_regularization": True,
        "optimizer": "rprop",
        "epochs": 200,
        "intensity_penalty": False,
        "folder_suffix": "comparative_analysis",
    },
    "region_optimization": {
        "weight_function": "distance",
        "mirror_boundary": False,
        "dist_params": {"n_expected": 100, "scale": 0.5},
        "n_trials": 25,
        "study_name": "optimizer",
        "l2_regularization": False,
        "optuna": True,
        "hidden_dims": [32],
        "folder_suffix": "region_optimization",
    },
}


def save_config_and_run(
        script_name, config, region_key="region", region=None, index=None,
):
    # Build filename
    suffix = f"{script_name}_{index}" if index is not None else script_name
    config_filename = f"configs/config_{suffix}.json"
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    if region_key and region:
        config[region_key] = region
    with open(config_filename, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Running: {script_name} with config {config_filename}")
    subprocess.run(
        [venv_python, "-m", f"src.{script_name}", "--config", config_filename],
        check=True,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script", type=str, required=True,
        choices=[
            "poisson_experiment", "comparative_analysis", "region_optimization"
        ],
        help="Which script to run."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the configuration file."
    )
    parser.add_argument("--workspace", type=str, required=True)
    args = parser.parse_args()

    config = BASE_CONFIGS[args.script].copy()
    config["workspace"] = args.workspace

    if args.script == "region_optimization":
        config["regions"] = REGION_VALUES_2D
        save_config_and_run(args.script, config, region_key=None)

    elif args.script == "poisson_experiment":
        for i, region in enumerate(REGION_VALUES_1D):
            save_config_and_run(
                args.script, config.copy(),
                region_key="region", region=region, index=i,
            )

    elif args.script == "comparative_analysis":
        for i, region in enumerate(REGION_VALUES_1D):
            save_config_and_run(
                args.script, config.copy(),
                region_key="region", region=region, index=i,
            )


if __name__ == "__main__":
    """
        python run_script.py --script poisson_experiment
        python run_script.py --script comparative_analysis
        python run_script.py --script region_optimization
    """
    main()
