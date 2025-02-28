import itertools
import json
import os
import subprocess
import sys

venv_python = os.path.join(os.path.dirname(sys.executable), "python")

# p_values = [0.1, 0.3, 0.5, 1]
# p_values = [-1]
# region_values = [[-1, 1], [-0.3, 0.3], [-0.5, 1], [-0.5, 0.5], [0, 1]]
# region_values = [[0, 3]]
region_values = [
    # [[-0.5, 0.5], [-0.5, 0.5]],
    [[-1, 1], [-1, 1]],
    # [[0, 1], [0, 1]],
    # [[-0.3, 0.3], [-0.3, 0.3]],
    # [[-0.5, 1], [-0.5, 1]]
]

base_config = {
    "dimensions": 2,
    "weighting": False,
    "kappa": 1000,
    "scale": 0.5,
    "num_samples": 100,
    "train_ratio": 0.8,
    "batch_size": 32,
    "epochs": 100,
    "p": None,
    "learning_rate": 1e-3,
    "weight_function": "gaussian",
    "output_json": "results.json",
    "output_image": "output_plot.png"
}

# Iterate over all combinations
# for p, region in itertools.product(p_values, region_values):
for region in region_values:
    # Update config dynamically
    config = base_config.copy()
    # config["p"] = p
    config["region"] = region
    config["output_json"] = f"results_region{region[0]}_{region[1]}.json"
    config["output_image"] = f"output_region{region[0]}_{region[1]}.png"

    # Save temporary config file
    config_filename = f"configs/config_region{region[0]}_{region[1]}.json"
    with open(config_filename, "w") as f:
        json.dump(config, f, indent=4)

    # Run the experiment
    print(f"Running experiment for region={region}")
    subprocess.run([venv_python, "run_experiment_poisson.py", "--config", config_filename])

print("All experiments completed.")
