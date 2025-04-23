import time

import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass

from utils import visualize_gradients, save_gradients_as_gif


@dataclass
class WindowParams:
    lengths: torch.Tensor = None
    region: list = None
    percent: float = None


class ScaledTanh(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return 0.5 * (self.a + self.b) + 0.5 * (self.b - self.a) * torch.tanh(x)


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

        layers.append(nn.Linear(last_dim, output_dim, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        log_intensity = self.network(x)
        return log_intensity

    def compute_psi(self, x):
        x.requires_grad_(True)
        intensity = self.forward(x)
        psi = torch.autograd.grad(
            intensity, x,
            grad_outputs=torch.ones_like(intensity),
            create_graph=True
        )[0]
        return psi, intensity

    def loss(self, x, lengths, mask):
        psi_x, intensity = self.compute_psi(x)

        if self.weighting_function is not None:
            params = WindowParams(
                lengths=lengths, region=self.region, percent=self.percent,
            )
            h = self.weighting_function(x, params)
            grad_h = self.weighting_derivative(x, params)
            weight = (psi_x * grad_h).sum(dim=-1)
        else:
            h = torch.ones_like(x)
            weight = 0

        norm_squared = (psi_x ** 2 * h).sum(dim=-1)

        divergence = 0
        for i in range(x.shape[-1]):  # Iterate over the features of x
            gradient = torch.autograd.grad(
                psi_x[..., i].sum(), x, retain_graph=True, create_graph=True
            )[0]
            divergence += gradient[..., i] * h[..., i]

        divergence = divergence * mask
        norm_squared = norm_squared * mask
        weight = weight * mask

        total_loss = 0.5 * norm_squared + divergence + weight # + 1e-2*torch.mean(intensity**2)
        total_loss = total_loss.sum(dim=-1) / lengths

        batch_size = x.size(0)
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

            lengths = x_data[:, 0, -1].to(dtype=torch.int64)
            max_length = lengths.max()
            padded_x = x_data[:, :max_length, :-1]

            mask = torch.arange(max_length).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)

            optimizer.zero_grad()
            loss = model.loss(padded_x, lengths, mask)
            loss.backward()

            # l2_lambda = 1e-3
            # l2_norm = sum(
            #     p.pow(2.0).sum() for p in model.parameters() if p.requires_grad
            # )
            # loss = loss + l2_lambda * l2_norm

            optimizer.step()

            loss_sum += loss.item()

        return loss_sum / total_samples if total_samples > 0 else 0.0

    model = nn_model(
        args, input_dim=args.dimensions, hidden_dims=args.hidden_dims
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate,
    )

    frames = []
    train_losses = []
    train_samples = len(loader_train)

    pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        start_time = time.time()

        # Training phase
        avg_train_loss = run_epoch(
            loader_train, model, optimizer,
            total_samples=train_samples,
            )
        train_losses.append(avg_train_loss)

        if args.plot_gradients and epoch % 5 == 0:
            visualize_gradients(model, args, epoch, frames)

        elapsed_time = time.time() - start_time
        pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Time/Epoch": f"{elapsed_time:.2f}s",
            "Region": f"{args.region}",
        })

    if args.plot_gradients:
        save_gradients_as_gif(frames, f"{args.gradient_dir}/gradient.gif")
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
