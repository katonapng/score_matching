import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Poisson_NN(nn.Module):
    def __init__(
            self, weighting_function,
            weighting_derivative, p=0.3,
            input_dim=None, hidden_dim=64, output_dim=1):
        super().__init__()
        torch.manual_seed(123)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_dim = input_dim  # Initially None, determined dynamically
        self.p = p
        self.weighting_function = weighting_function
        self.weighting_derivative = weighting_derivative

        # Placeholders for layers; 
        # they will be initialized once input_dim is set
        self.fc1 = None
        self.fc2 = None

    def initialize_layers(self, input_dim):
        """Initialize layers based on input_dim."""
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='tanh')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')

    def forward(self, x):
        if self.fc1 is None or self.fc2 is None:
            self.initialize_layers(x.shape[-1])

        x = torch.tanh(self.fc1(x))
        intensity = torch.exp(self.fc2(x))
        return intensity

    def compute_psi(self, x):
        x.requires_grad_(True)
        intensity = self.forward(x)
        nn_output = torch.log1p(intensity)
        psi = torch.autograd.grad(
            nn_output, x, grad_outputs=torch.ones_like(nn_output),
            create_graph=True)[0]
        return psi, intensity
    
    def loss(self, points, _lambda=0.01, weighting=True):
        lengths = points[:, 0, -1].to(dtype=torch.int64)
        max_length = lengths.max()
        x_t = points[:, :max_length, :-1]  # Pad to max length in batch

        # Create mask to remove padding
        mask = torch.arange(max_length, device=x_t.device).unsqueeze(0) < lengths.unsqueeze(1)
        psi_x, intensity = self.compute_psi(x_t)

        if weighting:
            h = self.weighting_function(x_t, self.p, lengths)
            grad_h = self.weighting_derivative(x_t, self.p, lengths)
            weight = (psi_x * grad_h).sum(dim=-1)
        else:
            h = torch.ones_like(x_t)
            weight = 0

        norm_squared = (psi_x ** 2 * h).sum(dim=-1)

        divergence = 0
        for i in range(x_t.shape[-1]):  # Iterate over the features of x
            gradient = torch.autograd.grad(
                psi_x[..., i].sum(), x_t, retain_graph=True, create_graph=True
                )[0]
            divergence += gradient[..., i] * h[..., i]

        divergence = divergence * mask
        norm_squared = norm_squared * mask
        weight = weight * mask
        intensity = intensity.squeeze(-1) * mask

        total_loss = 0.5 * norm_squared + divergence + weight  # + _lambda * (intensity**2)
        total_loss = total_loss.sum(dim=-1) / lengths

        batch_size = points.size(0)
        total_loss = total_loss.mean() / batch_size

        return total_loss


def optimize_nn(
        loader_train, nn_model, weighting_function, weighting_derivative,
        p=0.3, num_epochs=1000, learning_rate=1e-3, weighting=True):
    """
    Optimizes the model parameters using gradient descent.

    Args:
        loader_train (FastTensorDataLoader): DataLoader for training data,
                                             providing batches of input-output
                                             pairs.
        nn_model (torch.nn.Module): Neural network model to be optimized.
        num_epochs (int, optional): Number of training epochs. Default is 1000.
        learning_rate (float, optional): Learning rate for the optimizer.
        grad_clip_value (float, optional): Maximum allowed gradient norm
                                           for gradient clipping.
        weighting (bool, optional): Whether to apply sample weighting in
                                    loss computation.

    Returns:
        tuple:
            - torch.nn.Module: Trained neural network model.
            - list[float]: List of training loss values recorded per epoch.
    """
    def initialize_model_and_optimizer(p):
        model = nn_model(
            p=p, weighting_function=weighting_function,
            weighting_derivative=weighting_derivative
            )
        sample_input = next(iter(loader_train))[0][:, :, :-1].float()
        model(sample_input)  # Forward pass to initialize the layers
        optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return model, optimizer

    def run_epoch(loader, model, weighting, optimizer=None, total_samples=None):
        mode = model.train
        mode()

        loss_sum = 0.0
        for X_batch in loader:
            x_data = X_batch[0]

            optimizer.zero_grad()
            loss = model.loss(x_data, weighting=weighting)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        return loss_sum / total_samples if total_samples > 0 else 0.0

    model, optimizer = initialize_model_and_optimizer(p)
    train_losses = []
    train_samples = len(loader_train)

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for _ in pbar:
        start_time = time.time()

        # Training phase
        avg_train_loss = run_epoch(
            loader_train, model, weighting, optimizer,
            total_samples=train_samples,
            )
        train_losses.append(avg_train_loss)

        elapsed_time = time.time() - start_time
        pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Time/Epoch": f"{elapsed_time:.2f}s"
        })

    return model, train_losses


def generate_poisson_points(kappa, scale, region):
    """
    Generate a Poisson Point Process in a 2D region based
    on intensity function.

    Parameters:
    - kappa (torch.Tensor): The intensity parameter (scalar or vector).
    - scale (torch.Tensor): The scale parameter (scalar or vector).
    - region (tuple): The spatial domain as ((xmin, xmax), (ymin, ymax)).

    Returns:
    - points (numpy.ndarray): The simulated points of the PPP.
    """
    (xmin, xmax), (ymin, ymax) = region

    area = (xmax - xmin) * (ymax - ymin)
    max_intensity = kappa * area  # Maximum value of intensity
    num_samples = np.random.poisson(lam=max_intensity)

    x_candidates = np.random.uniform(xmin, xmax, size=num_samples)
    y_candidates = np.random.uniform(ymin, ymax, size=num_samples)
    candidates = torch.tensor(
        np.stack([x_candidates, y_candidates], axis=1), dtype=torch.float32,
        )

    squared_norm = torch.sum(candidates**2, dim=-1)
    intensity = kappa * torch.exp(-squared_norm / scale**2)

    uniform_samples = torch.rand(num_samples)  # Uniform samples for rejection
    acceptance_mask = uniform_samples < (intensity / kappa)

    accepted_points = candidates[acceptance_mask]
    return accepted_points.numpy()


def generate_poisson_points_1d(kappa, scale, region):
    """
    Generate a Poisson Point Process in a 1D region based on
    an intensity function.

    Parameters:
    - kappa (torch.Tensor): The intensity parameter (scalar or vector).
    - scale (torch.Tensor): The scale parameter (scalar or vector).
    - region (tuple): The spatial domain as (xmin, xmax).

    Returns:
    - points (numpy.ndarray): The simulated points of the PPP in 1D.
    """
    xmin, xmax = region

    # Length of the region
    length = xmax - xmin
    max_intensity = kappa * length
    num_samples = np.random.poisson(lam=max_intensity)

    # Generate candidate points
    x_candidates = np.random.uniform(xmin, xmax, size=num_samples)
    candidates = torch.tensor(x_candidates, dtype=torch.float32)

    # Calculate intensity
    squared_norm = candidates**2
    intensity = kappa * torch.exp(-squared_norm / scale**2)

    # Perform rejection sampling
    uniform_samples = torch.rand(num_samples)  # Uniform samples for rejection
    acceptance_mask = uniform_samples < (intensity / kappa)

    accepted_points = candidates[acceptance_mask]
    return accepted_points.numpy()
