import itertools

import numpy as np
import torch
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
