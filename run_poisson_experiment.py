import json
import os
import subprocess
import sys

venv_python = os.path.join(os.path.dirname(sys.executable), "python")

# region_values = [[-1, 1], [-0.3, 0.3], [-0.5, 1], [-0.5, 0.5], [0, 1]]
region_values = [
    [-1, 0.5], [-1, 0], [-0.5, 1], [-1, 0.7], [-1, 0.3],
    [-1, 1], [-0.3, 0.3], [-0.5, 0.5], [0, 1],
]
# percent_values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
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
    "hidden_dims": [32],
    "learning_rate": 1e-02,
}

# for percent in percent_values:
for region in region_values:
    config = base_config.copy()
    config["region"] = region
    n_layers = len(config["hidden_dims"]) + 1
    dims_str = "_".join(str(d) for d in config["hidden_dims"])
    config["folder_suffix"] = f"{n_layers}_layers_{dims_str}_neurons"
    config["output_json"] = f"results_region{region[0]}_{region[1]}.json"
    config["output_image"] = f"output_region{region[0]}_{region[1]}.png"

    # Save temporary config file
    config_filename = f"configs/config_region{region[0]}_{region[1]}.json"
    with open(config_filename, "w") as f:
        json.dump(config, f, indent=4)

    # Run the experiment
    print(f"Running experiment for region={region}")
    subprocess.run(
        [venv_python, "poisson_experiment.py", "--config", config_filename]
    )

print("All experiments completed.")
