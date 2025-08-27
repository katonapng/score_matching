import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import save_gradients_as_gif, visualize_gradients


@dataclass
class WindowParams:
    lengths: torch.Tensor = None
    region: list = None
    percent: float = None
    mirror_boundary: bool = None


class Poisson_SM(nn.Module):
    def __init__(self, args, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        torch.manual_seed(123)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.weighting_function = args.weight_function
        self.weighting_derivative = args.weight_derivative
        self.region = args.region
        self.percent = args.percent
        self.mirror_boundary = args.mirror_boundary
        self.intensity_penalty = args.intensity_penalty
        self.alpha = args.alpha

        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.Tanh())
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming Uniform (good for tanh)."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')

    def forward(self, x):
        log_intensity = self.network(x)
        return log_intensity

    def compute_psi(self, x):
        x.requires_grad_(True)
        log_intensity = self.forward(x)
        psi = torch.autograd.grad(
            log_intensity, x,
            grad_outputs=torch.ones_like(log_intensity),
            create_graph=True
        )[0]
        return psi, log_intensity.squeeze(-1)

    def loss(self, x, lengths, mask):
        psi_x, log_intensity = self.compute_psi(x)
        psi_x = psi_x*mask.unsqueeze(-1)

        if self.weighting_function is not None:
            params = WindowParams(
                lengths=lengths,
                region=self.region,
                percent=self.percent,
                mirror_boundary=self.mirror_boundary,
            )
            h = self.weighting_function(x, params)
            grad_h = self.weighting_derivative(x, params)
            weight = (psi_x * grad_h).sum(dim=-1)
        else:
            h = torch.ones_like(x)
            weight = torch.zeros_like(x).sum(dim=-1)

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
        log_intensity = log_intensity * mask

        total_loss = (
            0.5 * norm_squared + divergence + weight
        )
        if self.intensity_penalty:
            total_loss += self.alpha * (log_intensity ** 2)
        total_loss = total_loss.sum(dim=-1) / lengths

        return total_loss.mean()


class Poisson_MLE(nn.Module):
    def __init__(self, args, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        torch.manual_seed(123)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.region = args.region
        self.grid_steps = 100

        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.Tanh())
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming Uniform (good for tanh)."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')

    def forward(self, x):
        log_intensity = self.network(x)
        return log_intensity

    def compute_integral(self):
        """Numerical integration over the grid (1D or 2D)"""
        # Create grid for 1D or 2D
        if self.input_dim == 1:
            x_range = torch.linspace(
                self.region[0], self.region[1], steps=self.grid_steps
            )
            dx = (x_range[1] - x_range[0]).item()
            grid = x_range.unsqueeze(1)  # Shape: (grid_steps, 1)

            log_lambda = self.forward(grid)
            lambda_vals = torch.exp(log_lambda).squeeze()

            integral = torch.trapz(lambda_vals, dx=dx)
        
        elif self.input_dim == 2:
            x_range = torch.linspace(
                self.region[0][0], self.region[0][1], steps=self.grid_steps
            )
            y_range = torch.linspace(
                self.region[1][0], self.region[1][1], steps=self.grid_steps
            )
            dx = (x_range[1] - x_range[0]).item()
            dy = (y_range[1] - y_range[0]).item()

            xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
            grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

            log_lambda = self.forward(grid)
            lambda_vals = torch.exp(log_lambda)

            lambda_grid = lambda_vals.reshape(self.grid_steps, self.grid_steps)
            integral = torch.trapz(
                torch.trapz(lambda_grid, dx=dy, dim=1), dx=dx, dim=0
            )

        return integral

    def loss(self, x_data, lengths, mask):
        """Negative log-likelihood: -∑log λ(x_i) + ∫ λ(x) dx"""
        log_lambda = self.forward(x_data)
        log_lambda = log_lambda * mask.unsqueeze(-1)

        log_likelihood = torch.sum(log_lambda)
        log_likelihood = log_lambda.sum(dim=1).sum()

        integral = self.compute_integral()

        nll = -log_likelihood + integral * x_data.shape[0] 
        return nll


def optimize_nn(loader_train, loader_val, nn_model, args, trial=None):
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
    def compute_loss(model, X_batch, l2_regularization=False):
        x_data = X_batch[0]
        lengths = x_data[:, 0, -1].to(dtype=torch.int64)
        max_length = lengths.max()
        padded_x = x_data[:, :max_length, :-1]

        mask = torch.arange(max_length).unsqueeze(0)
        mask = mask < lengths.unsqueeze(1)

        loss = model.loss(padded_x, lengths, mask)

        if l2_regularization:
            l2_lambda = 1e-3
            l2_norm = sum(
                p.pow(2.0).sum() for p in model.parameters()
                if p.requires_grad
            )
            loss = loss + l2_lambda * l2_norm

        return loss

    def run_epoch(
        loader_train, loader_val, model, optimizer=None,
        train_samples=None, val_samples=None
    ):
        model.train()
        loss_sum = 0.0
        for X_batch in loader_train:
            optimizer.zero_grad()
            loss= compute_loss(
                model, X_batch, l2_regularization=args.l2_regularization
            )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_train_loss = loss_sum / train_samples if train_samples > 0 else 0.0

        model.eval()
        val_loss_sum = 0.0
        with torch.enable_grad():
            for X_batch in loader_val:
                loss = compute_loss(
                    model, X_batch, l2_regularization=args.l2_regularization
                )
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / val_samples if val_samples > 0 else 0.0

        return avg_train_loss, avg_val_loss

    if args.optuna:
        # n_layers = trial.suggest_int("n_layers", 1, 4)
        # args.hidden_dims = [
        #     trial.suggest_int(
        #         f"n_neurons_l{i}", 8, 32, step=8
        #     ) for i in range(n_layers)
        # ]
        # args.learning_rate = trial.suggest_float(
        #     "learning_rate", 1e-4, 1e-2, log=True,
        # )
        args.optimizer = trial.suggest_categorical(
            'optimizer', ['adam', 'rprop']
        )
    model = nn_model(
        args, input_dim=args.dimensions, hidden_dims=args.hidden_dims
    )

    if args.optimizer == "rprop":
        optimizer = torch.optim.Rprop(
            model.parameters(), lr=args.learning_rate,
        )
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate,  # amsgrad=True,
        )

    frames = []
    train_losses, val_losses = [], []
    train_samples = len(loader_train)
    val_samples = len(loader_val)
    best_val_smd = float('inf')

    patience = args.patience  # e.g., 10
    wait = 0  # How many epochs since last improvement
    delta = 0.0  # Minimal improvement to reset patience (optional)

    best_val_smd = float('inf')
    best_model_state = None

    pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        start_time = time.time()

        # Training and Validation phase
        avg_train_loss, avg_val_loss, = run_epoch(
                loader_train, loader_val, model, optimizer,
                train_samples=train_samples,
                val_samples=val_samples,
            )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if args.plot_gradients and epoch % 5 == 0:
            visualize_gradients(model, args, epoch, frames)

        elapsed_time = time.time() - start_time
        pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Validation Loss": f"{avg_val_loss:.6f}",
            "Time/Epoch": f"{elapsed_time:.2f}s",
        })

        # Early stopping check
        if avg_val_loss < best_val_smd - delta:
            best_val_smd = avg_val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pth')
            wait = 0  # Reset patience counter
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)."
                )
                break

    if args.plot_gradients:
        save_gradients_as_gif(frames, f"{args.gradient_dir}/gradient.gif")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return (
        model,
        train_losses,
        val_losses,
    )
