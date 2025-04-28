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

        total_loss = 0.5 * norm_squared + divergence + weight
        total_loss = total_loss.sum(dim=-1) / lengths

        return total_loss.mean()


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
    def run_epoch(
            loader_train, loader_val, model, optimizer=None, total_samples=None
    ):
        # Training phase
        model.train()

        loss_sum = 0.0
        for X_batch in loader_train:
            x_data = X_batch[0]

            lengths = x_data[:, 0, -1].to(dtype=torch.int64)
            max_length = lengths.max()
            padded_x = x_data[:, :max_length, :-1]

            mask = torch.arange(max_length).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)

            optimizer.zero_grad()
            loss = model.loss(padded_x, lengths, mask)
            loss.backward()

            if args.l2_regularization:
                l2_lambda = 1e-3
                l2_norm = sum(
                    p.pow(2.0).sum() for p in model.parameters()
                    if p.requires_grad
                )
                loss = loss + l2_lambda * l2_norm

            optimizer.step()

            loss_sum += loss.item()

        avg_train_loss = loss_sum / total_samples if total_samples > 0 else 0.0

        # Validation phase
        model.eval()
        with torch.no_grad():
            from metrics import compute_smd
            avg_val_smd, _ = compute_smd(loader_val, model, args)

        return avg_train_loss, avg_val_smd

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
    train_losses, val_smds = [], []
    train_samples = len(loader_train)
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
        avg_train_loss, avg_val_smd = run_epoch(
            loader_train, loader_val, model, optimizer,
            total_samples=train_samples,
        )
        train_losses.append(avg_train_loss)
        val_smds.append(avg_val_smd)

        if args.plot_gradients and epoch % 5 == 0:
            visualize_gradients(model, args, epoch, frames)

        elapsed_time = time.time() - start_time
        pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val SMD": f"{avg_val_smd:.6f}",
            "Time/Epoch": f"{elapsed_time:.2f}s",
        })

        # Early stopping check
        if avg_val_smd < best_val_smd - delta:
            best_val_smd = avg_val_smd
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

    return model, train_losses, val_smds
