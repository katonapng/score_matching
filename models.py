import time

import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class WindowParams:
    lengths: torch.Tensor = None
    region: list = None
    percent: float = None


class Poisson_NN(nn.Module):
    def __init__(self, args, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        torch.manual_seed(123)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.weighting_function = args.weight_function
        self.weighting_derivative = args.weight_derivative
        self.region = args.region
        self.percent = args.percent

        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.Tanh())
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return torch.exp(x)

    def compute_psi(self, x):
        x.requires_grad_(True)
        intensity = self.forward(x)
        nn_output = torch.log1p(intensity)
        psi = torch.autograd.grad(
            nn_output, x,
            grad_outputs=torch.ones_like(nn_output),
            create_graph=True
        )[0]
        return psi, intensity

    def loss(self, points):
        lengths = points[:, 0, -1].to(dtype=torch.int64)
        max_length = lengths.max()
        x_t = points[:, :max_length, :-1]  # Pad to max length in batch

        # Create mask to remove padding
        mask = torch.arange(max_length, device=x_t.device).unsqueeze(0)
        mask = mask < lengths.unsqueeze(1)
        psi_x, intensity = self.compute_psi(x_t)

        if self.weighting_function is not None:
            params = WindowParams(
                lengths=lengths, region=self.region, percent=self.percent,
            )
            h = self.weighting_function(x_t, params)
            grad_h = self.weighting_derivative(x_t, params)
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

        total_loss = 0.5 * norm_squared + divergence + weight
        total_loss = total_loss.sum(dim=-1) / lengths

        batch_size = points.size(0)
        total_loss = total_loss.mean() / batch_size

        return total_loss


def optimize_nn(loader_train, nn_model, args):
    """
    Optimizes the model parameters using gradient descent.

    Args:
        loader_train (FastTensorDataLoader): DataLoader for training data.
        nn_model (torch.nn.Module): Neural network model to be optimized.
        args (list): Arguments used in training.

    Returns:
        tuple:
            - torch.nn.Module: Trained neural network model.
            - list[float]: List of training loss values recorded per epoch.
    """
    def run_epoch(
            loader, model, optimizer=None, total_samples=None
    ):
        model.train()

        loss_sum = 0.0
        for X_batch in loader:
            x_data = X_batch[0]

            optimizer.zero_grad()
            loss = model.loss(x_data)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        return loss_sum / total_samples if total_samples > 0 else 0.0

    model = nn_model(
        args, input_dim=args.dimensions, hidden_dims=args.hidden_dims
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    train_samples = len(loader_train)

    pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for _ in pbar:
        start_time = time.time()

        # Training phase
        avg_train_loss = run_epoch(
            loader_train, model, optimizer,
            total_samples=train_samples,
            )
        train_losses.append(avg_train_loss)

        elapsed_time = time.time() - start_time
        pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Time/Epoch": f"{elapsed_time:.2f}s",
            "Region": f"{args.region}",
        })

    return model, train_losses


def optimize_nn_with_optuna(loader_train, nn_model, args, trial=None):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = [
        trial.suggest_int(
            f"n_neurons_l{i}", 8, 32, step=8
        ) for i in range(n_layers)
    ]
    # learning_rate = trial.suggest_float(
    #    "learning_rate", 1e-4, 1e-2, log=True,
    # )
    learning_rate = 1e-2
    input_dim = next(iter(loader_train))[0][:, :, :-1].float().shape[-1]
    model = nn_model(args, input_dim=input_dim, hidden_dims=hidden_dims)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def run_epoch(
            loader, model, optimizer=None, total_samples=None
    ):
        model.train()
        loss_sum = 0.0
        for X_batch in loader:
            x_data = X_batch[0]
            optimizer.zero_grad()
            loss = model.loss(x_data)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        return loss_sum / total_samples if total_samples > 0 else 0.0

    train_losses = []
    train_samples = len(loader_train)
    pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for _ in pbar:
        avg_train_loss = run_epoch(
            loader_train, model, optimizer,
            total_samples=train_samples,
        )
        train_losses.append(avg_train_loss)
        pbar.set_postfix({"Train Loss": f"{avg_train_loss:.4f}"})

    return model, train_losses
