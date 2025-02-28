import argparse
import json
import os
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.nn.utils.rnn import pad_sequence

from dataloader import FastTensorDataLoader
from poisson_model import (Poisson_NN, generate_poisson_points,
                           generate_poisson_points_1d, optimize_nn)


def remove_trailing_zeros(arr, lengths):
    """Removes trailing zeros from a PyTorch tensor along the last dimension."""
    batch_size = arr.shape[0]
    
    # Create a list of tensors with trimmed sequences
    trimmed_tensors = [arr[i, :lengths[i], :] for i in range(batch_size)]
    
    return trimmed_tensors


def gaussian_window(x, *args):
    _, lengths = args
    """
        Gaussian window function with tails near zero at boundaries
        for multi-dimensional input.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    epsilon = 1e-3
    x_clean = remove_trailing_zeros(x, lengths)
    valid_means = [sample.mean(dim=0, keepdim=True) for sample in x_clean]
    valid_mins = [sample.min(dim=0, keepdim=True).values for sample in x_clean]
    valid_maxs = [sample.max(dim=0, keepdim=True).values for sample in x_clean]

    mean = torch.stack(valid_means).mean(dim=0, keepdim=True)
    x_min = torch.stack(valid_mins).min(dim=0, keepdim=True).values
    x_max = torch.stack(valid_maxs).max(dim=0, keepdim=True).values

    sigma = torch.minimum(
        (abs(x_min - mean) / torch.sqrt(-2 * torch.log(torch.tensor(epsilon)))),
        (abs(x_max - mean) / torch.sqrt(-2 * torch.log(torch.tensor(epsilon))))
    )

    return torch.exp(-0.5 * ((x - mean) ** 2) / (sigma ** 2))


def gaussian_window_derivative(x, *args):
    """
        Derivative of the Gaussian window function with controlled
        tails for multi-dimensional input.
    """
    _, lengths = args
    if x.dim() == 1:
        x = x.unsqueeze(0)

    epsilon = 1e-3
    x_clean = remove_trailing_zeros(x, lengths)
    valid_means = [sample.mean(dim=0, keepdim=True) for sample in x_clean]
    valid_mins = [sample.min(dim=0, keepdim=True).values for sample in x_clean]
    valid_maxs = [sample.max(dim=0, keepdim=True).values for sample in x_clean]

    mean = torch.stack(valid_means).mean(dim=0, keepdim=True)
    x_min = torch.stack(valid_mins).min(dim=0, keepdim=True).values
    x_max = torch.stack(valid_maxs).max(dim=0, keepdim=True).values

    sigma = torch.minimum(
        (abs(x_min - mean) / torch.sqrt(-2 * torch.log(torch.tensor(epsilon)))),
        (abs(x_max - mean) / torch.sqrt(-2 * torch.log(torch.tensor(epsilon))))
    )

    gauss = torch.exp(-0.5 * ((x - mean) ** 2) / sigma ** 2)
    return -((x - mean) / (sigma ** 2)) * gauss


def bump_function(x, p, epsilon=1e-5, tail_threshold=1e-3):
    """
    Adaptive bump function ensuring the tails approach zero at x.min() and x.max().

    Args:
        x (Tensor): Input tensor.
        p (float): Controls the sharpness.
        epsilon (float): Small value to prevent division by zero.
        tail_threshold (float): Defines how small the function should be 
                                at boundaries.

    Returns:
        Tensor: Smooth bump function output with controlled tails.
    """
    x_min, x_max = x.min(), x.max()
    x_mean = x.mean()

    lambda_decay = max(
        abs(x_min - x_mean) / torch.sqrt(-2 * torch.log(torch.tensor(tail_threshold))),
        abs(x_max - x_mean) / torch.sqrt(-2 * torch.log(torch.tensor(tail_threshold)))
    )

    bump_core = torch.exp(-1 / ((x**2 + epsilon) ** p))
    boundary_decay = torch.exp(-((x - x_mean) ** 2) / (lambda_decay ** 2))

    return bump_core * boundary_decay


def bump_derivative(x, p, epsilon=1e-5, tail_threshold=1e-3):
    """
    Derivative of the adaptive bump function.

    Args:
        x (Tensor): Input tensor.
        p (float): Controls sharpness.
        epsilon (float): Small value to prevent division by zero.
        tail_threshold (float): Defines how small the function should be
                                at boundaries.

    Returns:
        Tensor: Derivative of the bump function.
    """
    x_min, x_max = x.min(), x.max()
    x_mean = x.mean()

    lambda_decay = max(
        abs(x_min - x_mean) / torch.sqrt(-2 * torch.log(torch.tensor(tail_threshold))),
        abs(x_max - x_mean) / torch.sqrt(-2 * torch.log(torch.tensor(tail_threshold)))
    )

    bump = bump_function(x, p, epsilon, tail_threshold)
    denominator = (x**2 + epsilon) ** (p + 1)

    return bump * ((2 * p * x / denominator) - (2 * (x - x_mean) / lambda_decay ** 2))


def calculate_score_matching_difference(intensity_real, intensity_pred):
    """
    Calculate the Score Matching Difference (SMD) 
    between the real and predicted intensities.
    """
    intensity_real = np.asarray(intensity_real)
    intensity_pred = np.asarray(intensity_pred)

    gradient_real = np.gradient(np.log(intensity_real))
    gradient_pred = np.gradient(np.log(intensity_pred))

    return np.sum((gradient_real - gradient_pred) ** 2)


def compute_mse_r2(loader_test, model, args):
    mse_list = []
    r2_list = []

    for batch in loader_test:
        lengths = batch[0][:, 0, -1].to(dtype=torch.int64)
        cleaned_batch = remove_trailing_zeros(batch[0], lengths)
        for x_test in cleaned_batch:
            if args.dimensions == 1:
                x_test = x_test[:, 0].unsqueeze(1)
                intensity_real = args.kappa * torch.exp(-x_test**2 / args.scale**2)
                intensity_pred = model(x_test).squeeze().detach()
            else:
                intensity_real = args.kappa * np.exp(
                    -(x_test[:, 0]**2 + x_test[:, 1]**2) / args.scale**2
                    )
                intensity_pred = model(
                    torch.tensor(x_test[:, :-1], dtype=torch.float32)
                    ).squeeze().detach()
            
            intensity_pred /= torch.max(intensity_pred)
            intensity_real /= torch.max(intensity_real)

            mse = mean_squared_error(intensity_real, intensity_pred)
            r2 = r2_score(np.asarray(intensity_real), np.asarray(intensity_pred))

            mse_list.append(mse)
            r2_list.append(r2) 

    return np.mean(mse_list), np.mean(r2_list)


def plot_results(args, model, input_data):
    plt.figure(figsize=(10, 6))
    lengths = input_data[:, 0, -1].to(dtype=torch.int64)
    x = remove_trailing_zeros(input_data, lengths)[0]
    
    if args.dimensions == 1:
        x = x[:, 0]
        x_min, x_max = x.min(), x.max()
        x_lin = np.linspace(x_min, x_max, 100)
        intensity_pred = model(torch.tensor(x_lin[:, None], dtype=torch.float32)).squeeze().detach()
        intensity_real = args.kappa * np.exp(-x_lin[:, None]**2 / args.scale**2)

        plt.plot(x_lin, intensity_pred / torch.max(intensity_pred), label='Predicted Intensity', color='blue')
        plt.plot(x_lin, intensity_real / intensity_real.max(), label='True Intensity', color='green')
        plt.scatter(x, np.zeros_like(x), c='red', s=10, alpha=0.6, label='Poisson Points')
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'Intensity $\rho(x)$', fontsize=14)
        plt.title('Intensity', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
    
    else:
        x, y = x[:, 0].numpy(), x[:, 1].numpy()
        x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
        x_lin = np.linspace(x_min, x_max, 100)
        y_lin = np.linspace(y_min, y_max, 100)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        intensity_pred = model(torch.tensor(grid_points, dtype=torch.float32)).detach()
        intensity_pred_2d = intensity_pred.reshape(xx.shape)
        intensity_real = args.kappa * np.exp(-(xx**2 + yy**2) / args.scale**2)

        norm = mcolors.Normalize(vmin=0, vmax=1)
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))  # 3 subplots for predicted, actual, and difference

        # Predicted intensity
        c1 = axs[0].contourf(
            xx, yy, intensity_pred_2d / torch.max(intensity_pred_2d),
            levels=50, cmap='viridis', alpha=0.8, norm=norm
        )
        axs[0].scatter(x, y, c='red', s=5, alpha=0.6, label='Poisson Points')
        axs[0].set_title(r'Predicted Intensity $\rho(x)$', fontsize=16)
        axs[0].set_xlabel(r'$x$', fontsize=14)
        axs[0].set_ylabel(r'$y$', fontsize=14)
        axs[0].legend()
        axs[0].set_aspect('equal')
        axs[0].grid(True, linestyle='--', alpha=0.5)
        fig.colorbar(c1, ax=axs[0], label="Normalized Intensity")

        # True intensity
        c2 = axs[1].contourf(
            xx, yy, intensity_real / intensity_real.max(),
            levels=50, cmap='viridis', alpha=0.8, norm=norm
        )
        axs[1].scatter(x, y, c='red', s=5, alpha=0.6, label='Poisson Points')
        axs[1].set_title('True Intensity', fontsize=16)
        axs[1].set_xlabel(r'$x$', fontsize=14)
        axs[1].set_ylabel(r'$y$', fontsize=14)
        axs[1].legend()
        axs[1].set_aspect('equal')
        axs[1].grid(True, linestyle='--', alpha=0.5)
        fig.colorbar(c2, ax=axs[1], label="Normalized Intensity")

        # Difference plot
        difference = abs((intensity_pred_2d / torch.max(intensity_pred_2d)) - (intensity_real / np.max(intensity_real)))
        c3 = axs[2].contourf(
            xx, yy, difference, levels=50, cmap='cividis', alpha=0.8, #vmin=0, vmax=1
        )
        axs[2].scatter(x, y, c='red', s=5, alpha=0.6, label='Poisson Points')
        axs[2].set_title("Difference Between Actual and Predicted Intensities", fontsize=16)
        axs[2].set_xlabel(r'$x$', fontsize=14)
        axs[2].set_ylabel(r'$y$', fontsize=14)
        axs[2].legend()
        axs[2].set_aspect('equal')
        axs[2].grid(True, linestyle='--', alpha=0.5)
        fig.colorbar(c3, ax=axs[2], label="Difference")

        plt.tight_layout()

    plt.savefig(args.output_image)
    plt.close()


def generate_training_data(args):
    kappa = torch.tensor(args.kappa)
    scale = torch.tensor(args.scale)
    region = args.region
    num_samples = args.num_samples
    samples = []

    generate_func = generate_poisson_points_1d if args.dimensions == 1 else generate_poisson_points

    for _ in range(num_samples):
        x_t = generate_func(kappa, scale, region)
        x_t = torch.tensor(x_t).unsqueeze(-1) if args.dimensions == 1 else torch.tensor(x_t, dtype=torch.float32)
        samples.append(x_t)

    samples_torch = [torch.tensor(s, dtype=torch.float32) for s in samples]
    X = pad_sequence(samples_torch, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(s) for s in samples_torch], dtype=torch.int64)
    lengths_expanded = lengths.unsqueeze(-1).expand(-1, X.shape[1])
    X = torch.cat((X, lengths_expanded.unsqueeze(-1)), dim=-1)

    m = len(X)
    train_size = int(args.train_ratio * m)
    X_train, X_test = X[:train_size], X[train_size:]

    train_loader = FastTensorDataLoader(X_train, batch_size=args.batch_size, shuffle=True)
    test_loader = FastTensorDataLoader(X_test, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader, X_test


def read_args_from_file(file_path):
    with open(file_path, "r") as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def check_file_existence(output_json, output_image):
    if os.path.exists(output_json) or os.path.exists(output_image):
        response = input(f"Output files '{output_json}' or '{output_image}' already exist. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting execution.")
            exit()


def generate_output_filenames(args):
    base_path = "results/1d/" if args.dimensions == 1 else "results/2d/"
    
    # Determine the correct folder for weighting
    if args.weight_function == "gaussian" and args.weighting:
        weight_path = "weighting_gaussian/"
        weight_suffix = ""  # No _p{args.p} for Gaussian
    elif args.weighting:
        weight_path = "weighting/"
        weight_suffix = f"_p{args.p}"
    else:
        weight_path = "no_weighting/"
        weight_suffix = ""

    # Create directory if it doesn't exist
    full_path = base_path + weight_path
    os.makedirs(full_path, exist_ok=True)

    # Ensure region suffix formatting
    region_suffix = f"_region{args.region}".replace(" ", "")

    # Generate output filenames
    output_json = f"{full_path}results{weight_suffix}{region_suffix}.json"
    output_image = f"{full_path}output_plot{weight_suffix}{region_suffix}.png"

    return output_json, output_image


def main(args):
    args.output_json, args.output_image = generate_output_filenames(args)
    if args.weight_function == "gaussian":
        weight_function = gaussian_window
        weight_derivative = gaussian_window_derivative
    else:
        weight_function = bump_function
        weight_derivative = bump_derivative

    check_file_existence(args.output_json, args.output_image)

    train_loader, test_loader, X_test = generate_training_data(args)

    # Train model with progress tracking
    print("Starting training...")
    model, train_losses = optimize_nn(
        p=args.p,
        weighting_function=weight_function,
        weighting_derivative=weight_derivative,
        loader_train=train_loader,
        nn_model=Poisson_NN,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weighting=args.weighting
    )

    # Compute metrics
    print("Computing metrics...")

    # smd = calculate_score_matching_difference(intensity_real, intensity_pred)
    smd = None
    mse, r2 = compute_mse_r2(test_loader, model, args)

    # Save metrics and parameters to JSON file
    output_data = {
        "parameters": vars(args),
        "metrics": {"smd": smd, "r2": r2, "mse": mse}
    }
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print("Metrics saved to", args.output_json)

    # Generate and save plot
    print("Generating plot...")
    plot_results(args, model, X_test)
    print("Plot saved to", args.output_image)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON file with arguments")
    parser.add_argument("--dimensions", type=int, choices=[1, 2])
    parser.add_argument("--region", type=json.loads, help="Region of the domain: (0,3) for 1D or ((0,1), (0,1)) for 2D")
    parser.add_argument("--weighting", type=bool)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--p", type=float)
    parser.add_argument("--weight_function", type=float)
    parser.add_argument("--weight_derivative", type=float)
    parser.add_argument("--output_json", type=str)
    parser.add_argument("--output_image", type=str)

    args = parser.parse_args()

    if args.config:
        args = read_args_from_file(args.config)

    main(args)
