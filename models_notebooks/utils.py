import re
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

FILES = [
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.05_inhom.csv",
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.15_inhom.csv",
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.01_hom.csv",
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.025_hom.csv",
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.05_hom.csv",
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.075_hom.csv",
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.1_hom.csv",
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.125_hom.csv",
    "simulations/simulated_points_sftcr_kappa0.5_sigma0.15_hom.csv",
    "simulations/simulated_points_lj_epsilon0.1_sigma0.15_hom.csv",
    "simulations/simulated_points_lj_epsilon0.1_sigma0.15_inhom.csv"
]


def load_samples(selected_filename, device='cpu', cutoff=10):
    """
    Load samples from a CSV file and return them as tensors.

    Args:
        selected_filename (dict): Dictionary with key 'value' containing the CSV filename.
        device (str or torch.device): Device to put tensors on.
        cutoff (int): Number of sample groups to keep.

    Returns:
        samples (tuple of torch.Tensor): List of sample tensors truncated to `cutoff`.
        scale (torch.Tensor)
        parameter (torch.Tensor) or epsilon (torch.Tensor)
        sigma
        homogeneous
        process
    """
    filename = selected_filename["value"]
    df = pd.read_csv(filename)
    parameter, sigma, homogeneous, process = parse_filename(filename)
    
    scale = torch.tensor(0.5)

    samples = [
        torch.tensor(group[["x", "y"]].values, dtype=torch.float32, device=device)
        for _, group in df.groupby("sim")
    ]
    samples = tuple(samples)[:cutoff]

    return samples, scale, parameter, sigma, homogeneous, process


def parse_filename(filename):
    if "sftcr" in filename:
        process_type = "sftcr"
        pattern = r'kappa([0-9.]+)_sigma([0-9.]+)'
    elif "lj" in filename:
        process_type = "lj"
        pattern = r'epsilon([0-9.]+)_sigma([0-9.]+)'
    else:
        process_type = None
        pattern = None

    kappa_or_epsilon = None
    sigma = None
    if pattern:
        match = re.search(pattern, filename)
        if match:
            kappa_or_epsilon = torch.tensor(float(match.group(1)))
            sigma = torch.tensor(float(match.group(2)))

    if "inhom" in filename:
        homogeneous = False
    elif "hom" in filename:
        homogeneous = True
    else:
        homogeneous = None

    return kappa_or_epsilon, sigma, homogeneous, process_type


def get_min_distances(samples, device='cpu'):
    min_dists = []
    for points in samples:
        if len(points) < 2:
            continue
        dist_matrix = torch.cdist(points, points)
        min_dist = torch.min(dist_matrix + torch.eye(len(points), device=device) * 1e6)
        # Adjust with n/(n+1)
        min_dists.append(min_dist.item() * len(points) / (len(points) + 1))

    return min_dists


def json_safe(obj):
    if isinstance(obj, torch.Tensor):
        return json_safe(obj.item() if obj.numel() == 1 else obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def plot_param_convergence(param_history, true_params=None):
    """
    Plots the convergence of parameters over epochs.

    Args:
        param_history (dict): Dictionary where keys are parameter names and values are lists of per-epoch averages.
        true_params (dict, optional): Dictionary of true parameter values to plot as horizontal lines.
    """
    num_params = len(param_history)
    fig, axs = plt.subplots(1, num_params, figsize=(6 * num_params, 4))

    if num_params == 1:
        axs = [axs]  # Ensure axs is iterable if there's only one subplot

    for ax, (param_name, values) in zip(axs, param_history.items()):
        display_name = param_name.replace("_raw", "")  # Remove "raw" for display
        tensor_values = torch.tensor(values)

        if param_name == "sigma_raw" or param_name == "epsilon_raw" or param_name == "scale_raw" or param_name == "alpha_raw":
            ax.plot(torch.exp(tensor_values), label=f"Estimated {display_name}")
        elif param_name == "k_raw":
            ax.plot(torch.sigmoid(tensor_values), label=f"Estimated {display_name}")
        else:
            ax.plot(values, label=f"Estimated {display_name}")

        if true_params and param_name in true_params:
            ax.axhline(true_params[param_name], color='orange', linestyle='--', label=f"True {display_name}")

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title(f'Convergence of {display_name}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def optimize_parametric(
    loader_train, nn_model,
    num_epochs=1000, learning_rate=1e-3, K=None,
    homogeneous=True, canonical=False, initial_sigma=None,
):
    model = nn_model(homogeneous, canonical, K=K, initial_sigma=initial_sigma)
    model = model.double()

    optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate, step_sizes=(1e-06, 50))

    avg_epoch_losses = []
    avg_epoch_real_losses = []
    param_history = {}

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        epoch_loss_sum = 0
        epoch_real_loss_sum = 0
        num_batches = len(loader_train)

        batch_param_values = {}
        start_time = time.time()

        for X_batch in loader_train:
            x = X_batch[0]
            x = x.double()
            x.requires_grad_()

            optimizer.zero_grad()
            loss, real_loss = model.loss(x)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_real_loss_sum += real_loss.item()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    value = param.detach().cpu().numpy().copy()
                    batch_param_values.setdefault(name, []).append(value)

        avg_loss = epoch_loss_sum / num_batches
        avg_real_loss = epoch_real_loss_sum / num_batches
        avg_epoch_losses.append(avg_loss)
        avg_epoch_real_losses.append(avg_real_loss)

        for name, values in batch_param_values.items():
            avg_value = sum(values) / len(values)
            param_history.setdefault(name, []).append(avg_value)

        elapsed_time = time.time() - start_time
        postfix = {
            "Train Loss": f"{avg_loss:.4f}",
            "Real Loss": f"{avg_real_loss:.4f}",
            "Time/Epoch": f"{elapsed_time:.2f}s",
        }
        pbar.set_postfix(postfix)

    return {
        "model": model,
        "avg_epoch_losses": avg_epoch_losses,
        "avg_epoch_real_losses": avg_epoch_real_losses,
        "param_history": param_history,
    }


def optimize_nonparametric(
        loader_train, loader_val, nn_model,
        num_epochs=1000, learning_rate=1e-3, K=None, device="cpu"
    ):
    """
    Optimizes the model parameters.

    Args:
        loader_train (DataLoader): DataLoader for training data.
        nn_model (callable): Function that creates the neural network model.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.

    Returns:
        dict: Contains the trained model, losses, and parameter values.
    """
    model = nn_model(device=device, K=K)
    optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
    
    avg_epoch_losses = []
    avg_epoch_val_losses = []
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for _ in pbar:
        epoch_loss_sum = 0
        num_batches = len(loader_train)

        start_time = time.time()
        for X_batch in loader_train:
            optimizer.zero_grad()

            x = X_batch[0].to(device)
            x = x.double()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()

        avg_loss = epoch_loss_sum / num_batches
        avg_epoch_losses.append(avg_loss)

        model.eval()
        val_loss_sum = 0
        num_batches_val = len(loader_val)
        with torch.enable_grad():
            for X_batch in loader_val:
                x = X_batch[0].to(device)
                x = x.double()
                loss = model.loss(x)
                val_loss_sum += loss.item()
    
        avg_val_loss = val_loss_sum / num_batches_val
        avg_epoch_val_losses.append(avg_val_loss)
    
        elapsed_time = time.time() - start_time
    
        pbar.set_postfix({
            "Train Loss": f"{avg_loss:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}",
            "Time/Epoch": f"{elapsed_time:.2f}s"
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save best model weights
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {_+1} — no improvement for {patience} epochs.")
                model.load_state_dict(best_model_state)  # Restore best weights
                break

    return {
        "model": model,
        "avg_epoch_losses": avg_epoch_losses,
        "avg_val_epoch_losses": avg_epoch_val_losses,
    }


class Gibbs(nn.Module):
    def __init__(
            self,
            homogeneous=True, canonical=False, K=None,
            initial_sigma=None, compare_analytical=True,
            device='cpu',
        ):
        super().__init__()
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        self.homogeneous = homogeneous
        self.canonical = canonical
        self.compare_analytical = compare_analytical
        self.K = K
        
        if not self.homogeneous:
            self.scale_raw = nn.Parameter(torch.log(torch.rand(1, device=device)))
        
        if self.canonical:
            epsilon_raw = torch.log(torch.rand(1, device=device))
            sigma_raw = torch.log(torch.tensor([initial_sigma], device=device))
            self.sigma0 = initial_sigma
        
            sigma = torch.exp(sigma_raw)
            epsilon = torch.exp(epsilon_raw)
            ratio = sigma / self.sigma0
        
            self.theta1 = nn.Parameter(4 * epsilon * ratio**12)
            self.theta2 = nn.Parameter(4 * epsilon * ratio**6)
        else:  
            self.sigma_raw = nn.Parameter(torch.log(torch.tensor([initial_sigma], device=device)))
            self.epsilon_raw = nn.Parameter(torch.log(torch.rand(1, device=device)))
                
    @property
    def epsilon(self):
        return torch.exp(self.epsilon_raw)

    @property
    def sigma(self):
        return torch.exp(self.sigma_raw)

    @property
    def scale(self):
        return torch.exp(self.scale_raw)

    def forward(self, x, mask):
        _, N, _ = x.shape
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=x.device)

        xi, xj = x[:, i_idx, :], x[:, j_idx, :]
        diff = xi - xj
        r2 = (diff ** 2).sum(dim=-1).clamp(min=1e-10)
        r = torch.sqrt(r2)

        mask_i, mask_j = mask[:, i_idx], mask[:, j_idx]
        pairwise_mask = mask_i & mask_j
        
        phi = torch.zeros_like(r)
        if self.canonical:
            sigma0_r = (self.sigma0 / r[pairwise_mask])**6
            phi[pairwise_mask] =  self.theta1 * sigma0_r**2 - self.theta2 * sigma0_r
        else:
            sigma_r = (self.sigma / r[pairwise_mask])**6
            phi[pairwise_mask] = 4 * self.epsilon * (sigma_r**2 - sigma_r)

        energy = phi.sum(dim=-1)
        if self.homogeneous:
            return -energy
        return -x.pow(2).sum(dim=-1).sum(dim=-1) / self.scale**2 - energy

    def compute_psi(self, x, mask):
        output = self.forward(x, mask)
        grad_outputs = torch.ones_like(output)
        return torch.autograd.grad(
            output, x, grad_outputs=grad_outputs,
            retain_graph=True, create_graph=True,
        )[0]
    
    def lennard_jones_psi(self, x, mask):
        _, N, _ = x.shape
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=x.device)
    
        xi, xj = x[:, i_idx, :], x[:, j_idx, :]
        diff = xi - xj
        r2 = (diff ** 2).sum(dim=-1).clamp(min=1e-10)
        r = torch.sqrt(r2)
    
        mask_i, mask_j = mask[:, i_idx], mask[:, j_idx]
        pairwise_mask = mask_i & mask_j
            
        inv_r = 1.0 / r[pairwise_mask]
        if self.canonical:
            sigma0 = self.sigma0
            inv_r6 = sigma0 ** 6 * inv_r ** 7
            inv_r12 =  sigma0 ** 12 * inv_r ** 13
            coeff = (-12 * self.theta1 * inv_r12 + 6 * self.theta2 * inv_r6) * inv_r
        else:
            inv_r6 = self.sigma ** 6 * inv_r ** 7
            inv_r12 = self.sigma ** 12 * inv_r ** 13
            coeff = 4 * self.epsilon * (-12 * inv_r12 + 6 * inv_r6) * inv_r
        
        grad = torch.zeros_like(diff)
        grad[pairwise_mask] = coeff.unsqueeze(-1) * diff[pairwise_mask]
    
        force = torch.zeros_like(x)
        force.index_add_(1, i_idx, grad)
        force.index_add_(1, j_idx, -grad)

        if self.homogeneous:
            return -force
        return -2 * x / self.scale**2 - force

    def J(self, x, mask):
        B, N, D = x.shape
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=x.device)
    
        psi_x = self.lennard_jones_psi(x, mask)
        norm_squared = psi_x.pow(2).sum(dim=-1)
    
        xi, xj = x[:, i_idx, :], x[:, j_idx, :]
        diff = xi - xj
        r2 = (diff ** 2).sum(dim=-1).clamp(min=1e-10)
    
        mask_i, mask_j = mask[:, i_idx], mask[:, j_idx]
        pairwise_mask = mask_i & mask_j  # (B, K)
    
        diff_outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
        I = torch.eye(D, device=x.device).view(1, 1, D, D)
    
        term = torch.zeros_like(diff_outer)
        valid_mask = pairwise_mask
    
        d_outer = diff_outer[valid_mask]
        r2v = r2[valid_mask].unsqueeze(-1).unsqueeze(-1)
    
        if self.canonical:
            sigma0 = self.sigma0
            sigma0_6 = sigma0 ** 6
            sigma0_12 = sigma0_6 ** 2
            LJ_matrix = -(
                -12 * self.theta1 * sigma0_12 * (I / r2v**7 - 14 * d_outer / r2v**8) +
                 6 * self.theta2 * sigma0_6 * (I / r2v**4 -  8 * d_outer / r2v**5)
            )
        else:
            sigma6 = self.sigma ** 6
            sigma12 = sigma6 ** 2
            LJ_matrix = -4 * self.epsilon * (
                -12 * sigma12 * (I / r2v**7 - 14 * d_outer / r2v**8) +
                 6 * sigma6  * (I / r2v**4 -  8 * d_outer / r2v**5)
            )
    
        term[valid_mask] = LJ_matrix
        div_phi = term.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, K)
        div_phi[~pairwise_mask] = 0
    
        interaction_div = torch.zeros(B, N, device=x.device)
        for idx in (i_idx, j_idx):
            interaction_div.index_add_(1, idx, div_phi)
    
        if self.homogeneous:
            divergence = interaction_div
        else:
            divergence = -2 / self.scale**2 * D + interaction_div
    
        return divergence, norm_squared
        
    def loss(self, points, device='cpu'):
        lengths = points[:, 0, -1].to(dtype=torch.int64, device=device)
        max_length = lengths.max()
        x_t = points[:, :max_length, :-1]
        mask = torch.arange(max_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        psi_x = self.compute_psi(x_t, mask)
        if self.compare_analytical: 
            psi_x_real = self.lennard_jones_psi(x_t, mask)

        norm_squared = psi_x.pow(2).sum(dim=-1) * mask

        if self.K:
            divergence = 0
            for _ in range(self.K):
                epsilon = torch.randint(0, 2, x_t.shape, device=device).float() * 2 - 1
                eps_psi = (psi_x * epsilon).sum()
                divergence_est = torch.autograd.grad(
                    eps_psi, x_t, create_graph=True
                )[0]
                divergence += (divergence_est * epsilon).sum(dim=-1) 
            divergence = (divergence / self.K) * mask
        else:
            divergence = torch.zeros(x_t.shape[0], x_t.shape[1], device=x_t.device)
            for d in range(x_t.shape[-1]):
                for i in range(x_t.shape[-2]):
                    second_grad = torch.autograd.grad(
                        psi_x[:, i, d].sum(), x_t, retain_graph=True, create_graph=True
                    )[0][:, i, d]
                    divergence[:, i] += second_grad
            divergence = divergence * mask

        if self.compare_analytical: 
            divergence_real, norm_squared_real = self.J(x_t, mask)
            divergence_real *= mask
            norm_squared_real *= mask

        total_loss = (0.5 * norm_squared + divergence).sum(dim=-1) / lengths
        J = (0.5 * norm_squared_real + divergence_real).sum(dim=-1) / lengths if self.compare_analytical else torch.tensor([0.0])
        return total_loss.mean(), J.mean()


class Gibbs_Softcore(nn.Module):
    def __init__(
            self,
            homogeneous=True, canonical=False, fixed_k=0.5,
            K=None, initial_sigma=None, compare_analytical=True,
            device='cpu',
        ):
        super().__init__()
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        self.homogeneous = homogeneous
        self.canonical = canonical
        self.compare_analytical = compare_analytical
        self.fixed_k = fixed_k
        self.K = K

        if not self.homogeneous:
            self.scale_raw = nn.Parameter(torch.log(torch.rand(1, device=device)))

        if canonical:
            sigma_raw = torch.tensor([initial_sigma], device=device)
            self.gamma = nn.Parameter(sigma_raw**(2/self.fixed_k))
        else:
            self.sigma_raw = nn.Parameter(torch.log(torch.tensor([initial_sigma], device=device)))
            self.k_raw = nn.Parameter(torch.rand(1, device=device))

    @property
    def scale(self):
        return torch.exp(self.scale_raw)

    @property
    def sigma(self):
        return torch.exp(self.sigma_raw) if not self.canonical else None

    @property
    def k(self):
        if self.canonical:
            return self.fixed_k
        return torch.sigmoid(self.k_raw)

    def forward(self, x, mask):
        _, N, _ = x.shape
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=x.device)

        xi, xj = x[:, i_idx, :], x[:, j_idx, :]
        diff = xi - xj
        r = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-10)

        mask_i, mask_j = mask[:, i_idx], mask[:, j_idx]
        pairwise_mask = mask_i & mask_j

        phi = torch.zeros_like(r)
        if self.canonical:
            phi[pairwise_mask] = self.gamma * (1 / r[pairwise_mask]) ** (2 / self.k)
        else:
            phi[pairwise_mask] = (self.sigma / r[pairwise_mask]) ** (2 / self.k)

        energy = phi.sum(dim=-1)
        if self.homogeneous:
            return -energy
        return - x.pow(2).sum(dim=-1).sum(dim=-1) / self.scale**2 - energy

    def compute_psi(self, x, mask):
        output = self.forward(x, mask)
        grad_outputs = torch.ones_like(output, device=x.device)
        return torch.autograd.grad(
            output, x, grad_outputs=grad_outputs,
            retain_graph=True, create_graph=True
        )[0]

    def sftcr_psi(self, x, mask):
        B, N, D = x.shape

        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=x.device)

        xi, xj = x[:, i_idx, :], x[:, j_idx, :]
        dx = xi - xj
        r2 = (dx ** 2).sum(dim=-1) + 1e-10
        r = r2.sqrt()

        mask_i, mask_j = mask[:, i_idx], mask[:, j_idx]
        pairwise_mask = mask_i & mask_j

        frac = torch.zeros_like(r)
        power = torch.zeros_like(frac)
        
        if self.canonical:
            frac[pairwise_mask] = 1 / r[pairwise_mask]
            power[pairwise_mask] = self.gamma * frac[pairwise_mask] ** (2 / self.k)
        else:
            frac[pairwise_mask] = self.sigma / r[pairwise_mask]
            power[pairwise_mask] = frac[pairwise_mask] ** (2 / self.k)

        factor = torch.zeros_like(frac)
        factor[pairwise_mask] = (2 / self.k) * power[pairwise_mask] / r2[pairwise_mask]

        force = factor.unsqueeze(-1) * dx
        force[~pairwise_mask] = 0

        grad = torch.zeros_like(x)
        grad.index_add_(1, i_idx, force)
        grad.index_add_(1, j_idx, -force)

        if self.homogeneous:
            return grad
        return -2 * x / (self.scale ** 2) + grad

    def J(self, x, mask):
        B, N, D = x.shape
        i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=x.device)

        psi_x = self.sftcr_psi(x, mask)
        norm_squared = psi_x.pow(2).sum(dim=-1)

        xi, xj = x[:, i_idx], x[:, j_idx]
        diff = xi - xj
        r2 = (diff ** 2).sum(dim=-1) + 1e-10
        r = r2.sqrt()

        mask_i, mask_j = mask[:, i_idx], mask[:, j_idx]
        pairwise_mask = mask_i & mask_j

        diff_outer = diff.unsqueeze(-1) @ diff.unsqueeze(-2)
        I = torch.eye(D, device=x.device).view(1, 1, D, D)
        r2_ = r2.unsqueeze(-1).unsqueeze(-1)
        r4 = r2_ ** 2

        phi = torch.zeros(B, r.size(1), 1, 1, device=x.device)
        if self.canonical:
            valid_phi = self.gamma * (1 / r[pairwise_mask]).pow(2 / self.k).unsqueeze(-1).unsqueeze(-1)
        else:
            valid_phi = (self.sigma / r[pairwise_mask]).pow(2 / self.k).unsqueeze(-1).unsqueeze(-1)
        phi[pairwise_mask] = valid_phi

        term = torch.zeros_like(diff_outer)
        valid_term = I / r2_[pairwise_mask] - ((2 / self.k + 2) * diff_outer[pairwise_mask] / r4[pairwise_mask])
        term[pairwise_mask] = valid_term

        div_phi_matrix = (2 / self.k) * phi * term
        div_phi = div_phi_matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
        div_phi[~pairwise_mask] = 0

        interaction_div = torch.zeros(B, N, device=x.device)
        for idx in (i_idx, j_idx):
            interaction_div.index_add_(1, idx, div_phi)

        if self.homogeneous:
            divergence = interaction_div
        else:
            divergence = -2 / self.scale**2 * D + interaction_div
        return divergence, norm_squared

    def loss(self, points, device='cpu'):
        lengths = points[:, 0, -1].to(dtype=torch.int64, device=device)
        max_length = lengths.max()
        x_t = points[:, :max_length, :-1]
        mask = torch.arange(max_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        psi_x = self.compute_psi(x_t, mask)
        
        if self.compare_analytical: 
            psi_x_real = self.sftcr_psi(x_t, mask)

        norm_squared = psi_x.pow(2).sum(dim=-1) * mask

        B, N, D = x_t.shape
        if self.K:
            divergence = 0
            for _ in range(self.K):
                epsilon = torch.randint(0, 2, x_t.shape, device=device).float() * 2 - 1
                eps_psi = (psi_x * epsilon).sum()
                divergence_est = torch.autograd.grad(
                    eps_psi, x_t, create_graph=True
                )[0]
                divergence += (divergence_est * epsilon).sum(dim=-1) 
            divergence = (divergence / self.K) * mask
        else:
            divergence = torch.zeros(x_t.shape[0], x_t.shape[1], device=x_t.device)
            for d in range(x_t.shape[-1]):
                for i in range(x_t.shape[-2]):
                    second_grad = torch.autograd.grad(
                        psi_x[:, i, d].sum(), x_t, retain_graph=True, create_graph=True
                    )[0][:, i, d]
                    divergence[:, i] += second_grad
            divergence = divergence * mask

        if self.compare_analytical:
            divergence_real, norm_squared_real = self.J(x_t, mask)
            divergence_real *= mask
            norm_squared_real *= mask

        total_loss = (0.5 * norm_squared + divergence).sum(dim=-1) / lengths 
        J = (0.5 * norm_squared_real + divergence_real).sum(dim=-1) / lengths if self.compare_analytical else torch.tensor([0.0])
        return total_loss.mean(), J.mean()