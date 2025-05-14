import itertools
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors
from PIL import Image
from scipy.integrate import nquad
from torch.nn.utils.rnn import pad_sequence

from dataloader import FastTensorDataLoader


def read_args_from_file(file_path, default_args):
    with open(file_path, "r") as f:
        args_dict = json.load(f)

    # Override defaults with config values
    for key, value in args_dict.items():
        setattr(default_args, key, value)

    return default_args


def get_region_dimension(region):
    if isinstance(region[0], list):
        return len(region)
    else:
        return 1


def generate_output_filenames(args):
    # Base path for 1d or 2d results
    base_path = "results/1d" if args.dimensions == 1 else "results/2d"

    # Convert model name to folder shorthand
    model_folder = "SM" if args.model == "Poisson_SM" else "MLE"

    # Convert boolean args to string
    mirror_folder = "mirror" if args.mirror_boundary else "no_mirror"
    l2_folder = "l2" if args.l2_regularization else "no_l2"

    # Construct full base path with new folders
    base_path = os.path.join(base_path, model_folder, mirror_folder, l2_folder)

    # Weighting path
    if args.weight_function is not None:
        weight_path = f"weighting_{args.weight_function}"
    else:
        weight_path = "no_weighting"

    if args.folder_suffix:
        weight_path += f"_{args.folder_suffix}"

    # Combine full path
    full_path = os.path.join(base_path, weight_path)
    os.makedirs(full_path, exist_ok=True)

    # Region suffix
    region_suffix = f"region{args.region}".replace(" ", "")

    # Gradient directory
    if args.plot_gradients:
        gradient_dir = os.path.join(full_path, "gradients", region_suffix)
        if os.path.exists(gradient_dir):
            shutil.rmtree(gradient_dir)
        os.makedirs(gradient_dir, exist_ok=True)
    else:
        gradient_dir = None

    # Losses directory
    losses_dir = os.path.join(full_path, "losses")
    os.makedirs(losses_dir, exist_ok=True)

    # Output files
    output_json = os.path.join(full_path, f"results_{region_suffix}.json")
    output_image = os.path.join(full_path, f"output_plot_{region_suffix}.png")
    loss_image = os.path.join(losses_dir, f"loss_plot_{region_suffix}.png")

    return output_json, output_image, gradient_dir, loss_image


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


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


def remove_trailing_zeros(arr, lengths):
    """
    Removes trailing zeros from a PyTorch tensor along the last dimension.
    """
    batch_size = arr.shape[0]

    # Create a list of tensors with trimmed sequences
    trimmed_tensors = [arr[i, :lengths[i], :] for i in range(batch_size)]

    return trimmed_tensors


def adjust_kappa_per_region(n_expected, scale, region):
    """
    Compute kappa to produce n_expected number of points per region.

    Parameters:
    - scale: The parameter for the Poisson distribution function.
    - region: List of lists [[xmin, xmax], [ymin, ymax], ...] specifying
      the region of integration.

    Returns:
    - Adjusted kappa value.
    """
    def intensity_function(*coords, scale):
        r2 = sum(x**2 for x in coords)
        return np.exp(-r2 / scale**2)

    # Bounds should be provided in the reverse order for nquad
    bounds = region.tolist()[::-1]
    integral, _ = nquad(
        lambda *coords: intensity_function(*coords, scale=scale),
        bounds
    )
    return n_expected / integral


def generate_poisson_points(scale, region, n_expected):
    """
    Generate a Poisson Point Process in a d-dimensional region
    based on an isotropic Gaussian-like intensity function.

    Parameters:
    - scale (torch.Tensor or float): Poisson scale parameter.
    - region (list of [min, max]): Observation window per dimension.
    - n_expected (torch.Tensor): Expected number of points per region.

    Returns:
    - accepted_points (torch.Tensor): The simulated points of the PPP.
    - kappa (float): Scaling constant for intensity.
    """
    dim = len(region)
    mins = np.array([r[0] for r in region])
    maxs = np.array([r[1] for r in region])
    sizes = maxs - mins

    # Volume of the hyperrectangle
    volume = np.prod(sizes)

    # Get the intensity scaling constant
    kappa = adjust_kappa_per_region(n_expected, scale, region)

    max_intensity = kappa * volume
    num_samples = np.random.poisson(lam=max_intensity)

    # Generate candidates uniformly in the region
    candidates_np = np.random.uniform(mins, maxs, size=(num_samples, dim))
    candidates = torch.tensor(candidates_np, dtype=torch.float32)

    # Compute isotropic squared norm
    squared_norm = torch.sum(candidates**2, dim=-1)

    # Intensity function (isotropic Gaussian shape)
    intensity = kappa * torch.exp(-squared_norm / scale**2)

    # Rejection sampling
    uniform_samples = torch.rand(num_samples)
    acceptance_mask = uniform_samples < (intensity / kappa)

    accepted_points = candidates[acceptance_mask]
    return accepted_points, kappa


def generate_training_data_poisson(args):
    for param in args.dist_params.keys():
        if param not in ["n_expected", "scale"]:
            msg = (
                f"Invalid parameter: {param}. "
                "Only 'n_expected' and 'scale' are allowed."
            )
            raise ValueError(msg)
    try:
        n_expected = args.dist_params.get("n_expected")
        scale = args.dist_params.get("scale")
    except KeyError as e:
        msg = f"Missing parameter: {e}. Provide 'n_expected' and 'scale'."
        raise KeyError(msg)

    region = args.region
    region = torch.tensor(region, dtype=torch.float32)
    if args.dimensions == 1:
        region = region[None, :]
    num_samples = args.num_samples
    samples = []

    for _ in range(num_samples):
        x_t, args.kappa = generate_poisson_points(scale, region, n_expected)
        if args.mirror_boundary:
            a = region[:, 0]
            b = region[:, 1]

            mid = (a + b) / 2
            proportion_region = (b - a) / 2

            mirrored_parts = [x_t]

            # Reflect along the left and right
            for dim in range(args.dimensions):
                left_thresh = a[dim] + proportion_region[dim]
                right_thresh = b[dim] - proportion_region[dim]

                # Mask for the left quarter
                mask_left = x_t[:, dim] <= left_thresh
                left_part = x_t[mask_left]

                left_mirror = left_part.clone()
                left_mirror[:, dim] = abs(
                    (left_part[:, dim] - proportion_region[dim]) - a[dim]
                ) + a[dim] - proportion_region[dim]

                # Mask for the right quarter
                mask_right = x_t[:, dim] >= right_thresh
                right_part = x_t[mask_right]

                right_mirror = right_part.clone()
                right_mirror[:, dim] = abs(
                    (right_part[:, dim] - proportion_region[dim]) - b[dim]
                ) + b[dim] - proportion_region[dim]

                mirrored_parts.extend([left_mirror, right_mirror])

            # Reflect corners: all combinations of axis halves (left and right)
            if args.dimensions > 1:
                corner_combinations = list(
                    itertools.product([0, 1], repeat=args.dimensions)
                )
            else:
                corner_combinations = []

            for combo in corner_combinations:
                mask = torch.ones(x_t.shape[0], dtype=torch.bool)
                for dim, side in enumerate(combo):
                    if side == 0:
                        mask &= x_t[:, dim] <= mid[dim]  # Left half
                    else:
                        mask &= x_t[:, dim] > mid[dim]  # Right half

                part = x_t[mask].clone()
                for dim, side in enumerate(combo):
                    if side == 0:
                        part[:, dim] = abs(
                            (part[:, dim] - proportion_region[dim]) - a[dim]
                        ) + a[dim] - proportion_region[dim]
                    else:
                        part[:, dim] = abs(
                            (part[:, dim] - proportion_region[dim]) - b[dim]
                        ) + b[dim] - proportion_region[dim]

                mirrored_parts.append(part)

            # Combine all mirrored parts
            mirrored = torch.cat(mirrored_parts, dim=0)
            samples.append(mirrored)
        else:
            samples.append(x_t)

    samples_torch = [torch.tensor(s, dtype=torch.float32) for s in samples]
    X = pad_sequence(samples_torch, batch_first=True, padding_value=0)

    lengths = torch.tensor([len(s) for s in samples_torch], dtype=torch.int64)
    lengths_expanded = lengths.unsqueeze(-1).expand(-1, X.shape[1])
    X = torch.cat((X, lengths_expanded.unsqueeze(-1)), dim=-1)

    # Split into train/val/test
    total_size = len(X)
    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)
    test_size = total_size - train_size - val_size
    X_train, X_val, X_test = torch.split(X, [train_size, val_size, test_size])

    train_loader = FastTensorDataLoader(
        X_train, batch_size=args.batch_size, shuffle=False
    )
    val_loader = FastTensorDataLoader(
        X_val, batch_size=args.batch_size, shuffle=False
    )
    test_loader = FastTensorDataLoader(
        X_test, batch_size=args.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


def get_real_poisson_gradient(points, kappa, scale):
    """
    Compute the gradient of the log intensity function:
        ∇ log λ(x) = -2x / s^2

    Parameters:
    - points: torch.Tensor of shape (N, d)
    - scale: float, Gaussian scale parameter

    Returns:
    - gradient: torch.Tensor of shape (N, d)
    """
    return -2 * points / scale**2


def save_gradients_as_gif(frames, save_path='gradients.gif'):
    """
    Saves the collected gradient frames as a GIF using Pillow.

    Args:
        frames (list): List of Image objects representing the frames.
        save_path (str): Path to save the GIF file.
    """
    frames[0].save(
        save_path, save_all=True, append_images=frames[1:], duration=500,
        loop=0,
    )


def visualize_gradients(model, args, epoch, frames):
    (xmin, xmax), (ymin, ymax) = args.region
    y_grid = np.linspace(xmin, xmax, 20)
    x_grid = np.linspace(ymin, ymax, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = torch.tensor(
        np.vstack([X.ravel(), Y.ravel()]).T,
        dtype=torch.float32, requires_grad=True,
    )

    # Get real and predicted gradients
    real_gradient = get_real_poisson_gradient(
        grid_points, args.kappa, args.dist_params.get("scale"),
    )
    if hasattr(model, "compute_psi"):  # SM model
        predicted_gradient = model.compute_psi(grid_points)[0]
    else:  # MLE model
        log_intensity = model(grid_points)
        predicted_gradient = torch.autograd.grad(
            outputs=log_intensity,
            inputs=grid_points,
            grad_outputs=torch.ones_like(log_intensity),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]

    real_grad_magnitude = torch.norm(real_gradient, dim=1)
    pred_grad_magnitude = torch.norm(predicted_gradient, dim=1)

    # Normalize the gradients and apply a scaling factor
    scaling_factor = 0.1
    scaled_real_gradient = (
        real_gradient / real_grad_magnitude.view(-1, 1)
    ) * scaling_factor
    scaled_predicted_gradient = (
        predicted_gradient / pred_grad_magnitude.view(-1, 1)
    ) * scaling_factor

    # Create a colormap and normalization
    norm_real = colors.Normalize(
        vmin=real_grad_magnitude.min().item(),
        vmax=real_grad_magnitude.max().item(),
    )
    norm_pred = colors.Normalize(
        vmin=pred_grad_magnitude.min().item(),
        vmax=pred_grad_magnitude.max().item(),
    )
    cmap = cm.viridis

    # Create a ScalarMappable to link the colorbar to the actual gradient magnitudes
    sm_real = cm.ScalarMappable(cmap=cmap, norm=norm_real)
    sm_real.set_array([])  # Empty array for the ScalarMappable
    sm_pred = cm.ScalarMappable(cmap=cmap, norm=norm_pred)
    sm_pred.set_array([])

    # Get log density values (assuming model.forward or similar)
    log_density = model.forward(grid_points)
    log_density_min = log_density.min().item()
    log_density_max = log_density.max().item()

    # Create the figure with subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Real intensity gradient
    ax[0].quiver(
        X, Y, scaled_real_gradient[:, 0].detach().numpy(),
        scaled_real_gradient[:, 1].detach().numpy(),
        angles='xy', scale_units='xy', scale=1,
        color=[sm_real.to_rgba(val) for val in real_grad_magnitude.detach().numpy()]
    )
    ax[0].set_title('Real Intensity Gradient')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    # Predicted intensity gradient
    ax[1].quiver(
        X, Y, scaled_predicted_gradient[:, 0].detach().numpy(),
        scaled_predicted_gradient[:, 1].detach().numpy(),
        angles='xy', scale_units='xy', scale=1,
        color=[sm_pred.to_rgba(val) for val in pred_grad_magnitude.detach().numpy()]
    )
    ax[1].set_title('Predicted Intensity Gradient')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    # Add colorbars using the ScalarMappable
    fig.colorbar(sm_real, ax=ax[0], orientation='vertical', label='Real Gradient Magnitude')
    fig.colorbar(sm_pred, ax=ax[1], orientation='vertical', label='Predicted Gradient Magnitude')

    # Add log density values as text
    ax[1].text(
        0.5, 1.05, 
        f'Log Density Min: {log_density_min:.2f}, Max: {log_density_max:.2f}',
        ha='center', va='bottom', transform=ax[1].transAxes, fontsize=10,
    )
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()

    frame_filename = f"{args.gradient_dir}/epoch_{epoch}.png"
    plt.savefig(frame_filename)
    plt.close(fig)

    # Open the saved image and append to the frames list
    image = Image.open(frame_filename)
    frames.append(image)
