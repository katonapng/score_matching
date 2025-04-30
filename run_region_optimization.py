import json
import os
import subprocess
import sys

venv_python = os.path.join(os.path.dirname(sys.executable), "python")

# region_values = [
#     [-1, 0.5], [-1, 0], [-0.5, 1], [-0.3, 1], [-1, 0.3],
#     [-1, 1], [-0.3, 0.3], [-0.5, 0.5], [0, 1],
# ]
region_values = [
    [[0.5, 1], [0.5, 1]],
    [[-0.5, 0.5], [-0.5, 0.5]],
    [[-1, 1], [-1, 1]],
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

base_config = {
    "weight_function": "distance",
    "mirror_boundary": False,
    "dist_params": {"n_expected": 100, "scale": 0.5},
    "n_trials": 25,
    "study_name": "optimizer",
    "l2_regularization": False,
    "optuna": True,
    "hidden_dims": [32],
}

config = base_config.copy()
config["regions"] = region_values

# Save temporary config file
config_filename = (
    f"configs/config_opt_{config['weight_function']}_mirror_"
    f"{config['mirror_boundary']}.json"
)

with open(config_filename, "w") as f:
    json.dump(config, f, indent=4)

# Run the experiment
print("Running Optuna optimization:")
subprocess.run(
    [venv_python, "region_optimization.py", "--config", config_filename]
)

print("Optimization completed.")
# optuna-dashboard sqlite:///optuna_study.db
