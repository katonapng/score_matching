import torch

from models import WindowParams
from utils import remove_trailing_zeros


def gaussian_window(x, params: WindowParams):
    """
        Gaussian window function with tails near zero at boundaries
        for multi-dimensional input.
    """
    lengths = params.lengths
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


def gaussian_window_derivative(x, params: WindowParams):
    """
        Derivative of the Gaussian window function with controlled
        tails for multi-dimensional input.
    """
    lengths = params.lengths
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


def distance_window(x, params: WindowParams):
    """
    Compute min(1, L * g(x)) where g(x) is the distance to the boundary.

    Parameters:
    x : torch.Tensor
        Input tensor of points.
    region : list
        List defining the region boundaries.

    Returns:
    torch.Tensor
        Scaled distance function.
    """
    region = params.region
    percent = params.percent
    if isinstance(region[0], (int, float)):
        a = torch.tensor([region[0]], dtype=torch.float32)
        b = torch.tensor([region[1]], dtype=torch.float32)
    else:
        a = torch.tensor([r[0] for r in region], dtype=torch.float32)
        b = torch.tensor([r[1] for r in region], dtype=torch.float32)
    L = 100 / (percent * (b - a))  # Scaling factor

    # Compute distances to the boundaries for each dimension
    dist_to_a = torch.abs(x - a)
    dist_to_b = torch.abs(x - b)

    # Compute g(x) as the min distance in each dimension (independently for each dimension)
    g = torch.minimum(dist_to_a, dist_to_b)

    # Apply the scaled distance function and return min(1, L * g(x))
    return torch.minimum(L * g, torch.tensor(1.0))


def distance_window_derivative(x, params: WindowParams):
    """
    Compute the derivative of min(1, L * g(x)).

    Parameters:
    x : torch.Tensor
        Input tensor of points.
    region : list
        List defining the region boundaries.

    Returns:
    torch.Tensor
        Derivative of the scaled distance function.
    """
    region = params.region
    percent = params.percent
    if isinstance(region[0], (int, float)):
        a = torch.tensor([region[0]], dtype=torch.float32)
        b = torch.tensor([region[1]], dtype=torch.float32)
    else:
        a = torch.tensor([r[0] for r in region], dtype=torch.float32)
        b = torch.tensor([r[1] for r in region], dtype=torch.float32)
    L = percent / (b - a)  # Scaling factor

    # Compute distances to the boundaries for each dimension
    dist_to_a = torch.abs(x - a)
    dist_to_b = torch.abs(x - b)

    # Compute the minimum distance in each dimension (independently for each dimension)
    g = torch.minimum(dist_to_a, dist_to_b)

    # Initialize gradient as zero
    grad = torch.zeros_like(x)

    # Create masks for each boundary (where the distance is smaller)
    mask_a = dist_to_a < dist_to_b
    mask_b = dist_to_b < dist_to_a

    grad[mask_a] = torch.sign(x[mask_a] - a.expand_as(x)[mask_a])
    grad[mask_b] = torch.sign(x[mask_b] - b.expand_as(x)[mask_b])

    # Mask where L * g < 1
    L_expand = L.expand_as(x)
    mask = (L_expand * g) < 1.0

    grad = grad * L_expand * mask.float()

    return grad


def distance_mirror_window(x, params: WindowParams):
    """
    Compute min(1, L * g(x)) where g(x) is the distance to the extended boundary,
    and L is chosen such that the function equals 1 at x = a and x = b.

    Parameters:
    x : torch.Tensor
        Input tensor of points.
    a : float
        Left boundary of the original region.
    b : float
        Right boundary of the original region.

    Returns:
    torch.Tensor
        Scaled distance function.
    """
    a, b = params.region[0], params.region[1]
    half_region = (b - a) / 2
    a_ext = a - half_region
    b_ext = b + half_region

    # Distance to extended region boundaries
    dist_to_a_ext = torch.abs(x - a_ext)
    dist_to_b_ext = torch.abs(x - b_ext)
    g = torch.minimum(dist_to_a_ext, dist_to_b_ext)

    # Compute L such that L * g(a) = 1
    L = 1.0 / ((a - a_ext) if (a - a_ext) != 0 else 1e-8)  # Avoid division by zero

    result = torch.minimum(torch.tensor(1.0), L * g)

    return result


def distance_mirror_window_derivative(x, params: WindowParams):
    """
    Compute the derivative of min(1, L * g(x)).

    Parameters:
    x : torch.Tensor
        Input tensor of points.
    a : float
        Left boundary of the original region.
    b : float
        Right boundary of the original region.

    Returns:
    torch.Tensor
        Derivative of the scaled distance function.
    """
    a, b = params.region[0], params.region[1]
    half_region = (b - a) / 2
    a_ext = a - half_region
    b_ext = b + half_region

    dist_to_a_ext = torch.abs(x - a_ext)
    dist_to_b_ext = torch.abs(x - b_ext)

    g = torch.minimum(dist_to_a_ext, dist_to_b_ext)
    L = 1.0 / ((a - a_ext) if (a - a_ext) != 0 else 1e-8)

    # Gradient of g(x)
    grad = torch.zeros_like(x)
    mask_a = dist_to_a_ext < dist_to_b_ext
    mask_b = dist_to_b_ext < dist_to_a_ext

    grad[mask_a] = torch.sign(x[mask_a] - a_ext)
    grad[mask_b] = torch.sign(x[mask_b] - b_ext)

    # Apply scaling and cutoff where L*g < 1
    mask = (L * g) < 1.0
    grad = grad * L * mask.float()
