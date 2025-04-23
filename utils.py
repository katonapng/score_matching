import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors
from PIL import Image
from scipy.integrate import nquad
from torch.nn.utils.rnn import pad_sequence

from dataloader import FastTensorDataLoader


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
            # Mirror split
            a = region[:, 0]
            b = region[:, 1]

            mid = (a + b) / 2
            half_length = (b - a) / 2

            mirrored_parts = [x_t]

            # Reflect along the left and right halves for each dimension
            for dim in range(args.dimensions):
                # Mask for the left part
                mask_left = x_t[:, dim] <= mid[dim]
                left_part = x_t[mask_left]

                left_mirror = left_part.clone()
                left_mirror[:, dim] = abs(
                    (left_part[:, dim] - half_length[dim]) - a[dim]
                ) + a[dim] - half_length[dim]

                mask_right = x_t[:, dim] > mid[dim]
                right_part = x_t[mask_right]

                right_mirror = right_part.clone()
                right_mirror[:, dim] = abs(
                    (right_part[:, dim] - half_length[dim]) - b[dim]
                ) + b[dim] - half_length[dim]

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
                            (part[:, dim] - half_length[dim]) - a[dim]
                        ) + a[dim] - half_length[dim]
                    else:
                        part[:, dim] = abs(
                            (part[:, dim] - half_length[dim]) - b[dim]
                        ) + b[dim] - half_length[dim]

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

    train_size = int(args.train_ratio * len(X))
    X_train, X_test = X[:train_size], X[train_size:]

    train_loader = FastTensorDataLoader(
        X_train, batch_size=args.batch_size, shuffle=False
    )
    test_loader = FastTensorDataLoader(
        X_test, batch_size=args.batch_size, shuffle=False
    )

    return train_loader, test_loader


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
        np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32,
    )

    # Get real and predicted gradients
    real_gradient = get_real_poisson_gradient(
        grid_points, args.kappa, args.dist_params.get("scale"),
    )
    predicted_gradient = model.compute_psi(grid_points)[0]

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

    norm = colors.Normalize(
        vmin=real_grad_magnitude.min(), vmax=real_grad_magnitude.max()
    )
    cmap = cm.viridis

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
        color=cmap(norm(real_grad_magnitude.detach().numpy())))
    ax[0].set_title('Real Intensity Gradient')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    # Predicted intensity gradient
    ax[1].quiver(
        X, Y, scaled_predicted_gradient[:, 0].detach().numpy(),
        scaled_predicted_gradient[:, 1].detach().numpy(),
        angles='xy', scale_units='xy', scale=1,
        color=cmap(norm(pred_grad_magnitude.detach().numpy()))
    )
    ax[1].set_title('Predicted Intensity Gradient')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    # Add log density values as text
    ax[1].text(
        0.5, 1.05, 
        f'Log Density Min: {log_density_min:.2f}, Max: {log_density_max:.2f}',
        ha='center', va='bottom', transform=ax[1].transAxes, fontsize=10,
    )

    plt.tight_layout()

    frame_filename = f"{args.gradient_dir}/epoch_{epoch}.png"
    plt.savefig(frame_filename)
    plt.close(fig)

    # Open the saved image and append to the frames list
    image = Image.open(frame_filename)
    frames.append(image)
