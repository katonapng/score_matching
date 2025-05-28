import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize

from src.models import WindowParams
from src.utils import remove_trailing_zeros
from src.weight_functions import distance_window, gaussian_window


def calculate_score_matching_difference(
        log_intensity_real, log_intensity_pred, dim,
):
    """
    Calculate the Score Matching Difference (SMD)
    between the real and predicted log intensities.
    """
    log_intensity_real = np.asarray(log_intensity_real)
    log_intensity_pred = np.asarray(log_intensity_pred)
    if dim == 1:
        log_intensity_real = log_intensity_real.squeeze(-1)

    gradient_real = np.gradient(log_intensity_real)
    gradient_pred = np.gradient(log_intensity_pred)

    return np.sum((gradient_real - gradient_pred) ** 2)


def calculate_metrics(loader, model, args):
    smd_list = []
    mae_list = []
    maxae_list = []
    log_intensity_stats = {'min': [], 'mean': [], 'max': []}

    kappa = torch.tensor(args.kappa)
    scale = args.dist_params.get("scale")

    for batch in loader:
        lengths = batch[0][:, 0, -1].to(dtype=torch.int64)
        cleaned_batch = remove_trailing_zeros(batch[0], lengths)

        for x_test in cleaned_batch:
            if args.dimensions == 1:
                x_test = x_test[:, 0].unsqueeze(1)
                lower, upper = args.region
                region_mask = (x_test >= lower) & (x_test <= upper)
                region_volume = upper - lower
            else:
                (x_lower, x_upper), (y_lower, y_upper) = args.region
                region_mask = (
                    (x_test[:, 0] >= x_lower) & (x_test[:, 0] <= x_upper) &
                    (x_test[:, 1] >= y_lower) & (x_test[:, 1] <= y_upper)
                )
                region_volume = (x_upper - x_lower) * (y_upper - y_lower)

            x_test_filtered = (
                x_test[region_mask].unsqueeze(1)
                if args.dimensions == 1
                else x_test[region_mask]
            )

            # Compute true intensity
            if args.dimensions == 1:
                log_intensity_real = torch.log(kappa) - x_test_filtered**2 / scale**2
                model_input = x_test_filtered
            else:
                log_intensity_real = torch.log(kappa) - (
                    (x_test_filtered[:, 0]**2 + x_test_filtered[:, 1]**2) / scale**2
                )
                model_input = x_test_filtered[:, :-1]  # exclude timestamp if present

            # Predict intensity
            log_intensity_pred = model(model_input).detach().squeeze(-1)
            if args.model == "Poisson_SM":
                # log(kappa) = log(n_expected) - log_integral
                n_expected = torch.tensor(x_test_filtered.shape[0])
                log_integral = (
                    torch.logsumexp(log_intensity_pred, dim=0)
                    - torch.log(n_expected)
                )
                log_integral += torch.log(torch.tensor(region_volume))
                log_kappa = torch.log(n_expected) - log_integral

                log_intensity_pred = log_intensity_pred + log_kappa

            # Calculate SMD
            smd = calculate_score_matching_difference(
                log_intensity_real, log_intensity_pred, args.dimensions,
            )
            smd_list.append(smd)

            # Calculate MAE
            mae = F.l1_loss(
                log_intensity_pred, log_intensity_real, reduction='mean',
            )
            mae_list.append(mae.item())

            # Calculate MaxAE (L-infinity norm)
            maxae = torch.max(
                torch.abs(log_intensity_pred - log_intensity_real)
            ).item()
            maxae_list.append(maxae)

            # Save intensity statistics
            log_intensity_stats['min'].append(log_intensity_pred.min().item())
            log_intensity_stats['mean'].append(log_intensity_pred.mean().item())
            log_intensity_stats['max'].append(log_intensity_pred.max().item())

    avg_smd = np.mean(smd_list)
    avg_mae = np.mean(mae_list)
    avg_maxae = np.mean(maxae_list)
    avg_log_intensity_stats = {
        'min': np.mean(log_intensity_stats['min']),
        'mean': np.mean(log_intensity_stats['mean']),
        'max': np.mean(log_intensity_stats['max']),
    }

    return avg_smd, avg_mae, avg_maxae, avg_log_intensity_stats


def plot_results(args, model, test_loader):
    def sample_test_data():
        sample = test_loader.random_sample()[0]
        lengths = sample[:, 0, -1].to(dtype=torch.int64)
        return remove_trailing_zeros(sample, lengths)[0]

    def prepare_grid(x, y, num_points=100):
        x_lin = np.linspace(x.min(), x.max(), num_points)
        y_lin = np.linspace(y.min(), y.max(), num_points)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        return xx, yy, grid_points

    def plot_weight_function(x_lin, weight_fn_name):
        temp = torch.tensor(x_lin[:, None], dtype=torch.float32).unsqueeze(0)
        if weight_fn_name == "gaussian_window":
            params = WindowParams(
                lengths=torch.tensor([temp.shape[1]], dtype=torch.int32)
            )
            weights = gaussian_window(temp, params)
        elif weight_fn_name == "distance_window":
            params = WindowParams(
                region=args.region,
                percent=args.percent,
                mirror_boundary=args.mirror_boundary,
            )
            weights = distance_window(temp, params)
        else:
            return

        plt.plot(
            x_lin, weights.squeeze(0),
            label=weight_fn_name.replace('_', ' ').title(),
            color='orange', linestyle='--',
        )

    def mirrored_intensity(x, a, b, kappa, scale):
        length = b - a
        x_relative = (x - a) % (2 * length)
        x_mirrored = np.where(
            x_relative <= length, a + x_relative, b - (x_relative - length)
        )
        return kappa * np.exp(-x_mirrored**2 / scale**2)

    def plot_1d(x):
        kappa, scale = args.kappa, args.dist_params.get("scale")
        x = x[:, 0]
        x_min, x_max = x.min(), x.max()
        if args.mirror_boundary:
            x_lin = np.linspace(x_min, x_max, 100)
        else:
            x_lin = np.linspace(args.region[0], args.region[1], 100)

        with torch.no_grad():
            intensity_pred = model(
                torch.tensor(x_lin[:, None], dtype=torch.float32)
            ).squeeze()
            intensity_pred = torch.exp(intensity_pred)

        intensity_real = (
            mirrored_intensity(x_lin[:, None], *args.region, kappa, scale)
            if args.mirror_boundary
            else kappa * np.exp(-x_lin[:, None]**2 / scale**2)
        )

        if args.weight_function:
            plot_weight_function(x_lin, args.weight_function.__name__)

        # Compute max and min before normalization
        pred_max, pred_min = intensity_pred.max().item(), intensity_pred.min().item()
        real_max, real_min = intensity_real.max().item(), intensity_real.min().item()

        # Plot
        plt.plot(
            x_lin, intensity_pred / pred_max,
            label=f'Predicted Intensity\n(max={pred_max:.2f}, min={pred_min:.2f})',
            color='blue'
        )
        plt.plot(
            x_lin, intensity_real / real_max,
            label=f'True Intensity\n(max={real_max:.2f}, min={real_min:.2f})',
            color='green'
        )

        plt.scatter(
            x, np.zeros_like(x), c='red', s=10, alpha=0.6,
            label='Poisson Points',
        )

        if args.mirror_boundary:
            plt.axvspan(
                x_min, args.region[0], color='gray', alpha=0.2,
                label='Mirrored Region',
            )
            plt.axvspan(args.region[1], x_max, color='gray', alpha=0.2)

        plt.xlabel(r'$x$', fontsize=12)
        plt.ylabel(fr'Intensity ($\kappa$ = {kappa:.2f})', fontsize=12)
        plt.title("Normalized Intensities", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        if not args.mirror_boundary:
            plt.xlim(args.region[0], args.region[1])
        plt.legend()

    def plot_2d(x):
        kappa, scale = args.kappa, args.dist_params.get("scale")
        x_vals, y_vals = x[:, 0].numpy(), x[:, 1].numpy()
        xx, yy, grid_points = prepare_grid(x_vals, y_vals)

        with torch.no_grad():
            intensity_pred = model(
                torch.tensor(grid_points, dtype=torch.float32)
            )
            intensity_pred = torch.exp(intensity_pred).reshape(xx.shape)

        intensity_real = kappa * np.exp(-(xx**2 + yy**2) / scale**2)
        intensity_pred_np = intensity_pred.numpy()

        fig, axs = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        titles = [
            r'Predicted Intensity $\rho(x)$',
            'True Intensity',
            'Difference Between Actual and Predicted Intensities'
        ]
        cmaps = ['viridis', 'viridis', 'cividis']

        (xmin, xmax), (ymin, ymax) = args.region
        region_mask = ((xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax))
        opacity_mask = np.where(
            (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax), 1.0, 0.5
        )

        # Get normalization factor only from original region
        norm_factor_pred = intensity_pred_np[region_mask].max()
        norm_factor_real = intensity_real[region_mask].max()

        # Normalize only by the original region's max
        intensity_pred_norm = intensity_pred_np / norm_factor_pred
        intensity_real_norm = intensity_real / norm_factor_real

        abs_diff = abs(intensity_pred_norm - intensity_real_norm)

        norm_pred = Normalize(vmin=0, vmax=1)
        norm_real = Normalize(vmin=0, vmax=1)
        norm_diff = Normalize(vmin=0, vmax=1)

        plots = [
            intensity_pred_norm,
            intensity_real_norm,
            abs_diff
        ]

        for i, (plot_data, title, cmap) in enumerate(zip(plots, titles, cmaps)):
            if isinstance(plot_data, torch.Tensor):
                plot_data = plot_data.numpy()

            # Select the correct norm object
            norm_used = None
            if i == 0:
                norm_used = norm_pred
            elif i == 1:
                norm_used = norm_real
            elif i == 2:
                norm_used = norm_diff

            if args.mirror_boundary:
                c = axs[i].imshow(
                    plot_data,
                    extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
                    origin='lower', cmap=cmap, alpha=opacity_mask,
                    aspect='auto', norm=norm_used
                )
            else:
                c = axs[i].contourf(
                    xx, yy, plot_data, levels=50, alpha=0.8,
                    cmap=cmap, norm=norm_used
                )

            axs[i].scatter(
                x_vals, y_vals, c='red', s=5, alpha=0.6, label='Poisson Points'
            )
            axs[i].set_title(title, fontsize=12)
            axs[i].set_xlabel(r'$x$', fontsize=12)
            axs[i].set_ylabel(r'$y$', fontsize=12)
            axs[i].legend()
            axs[i].set_aspect('equal')
            axs[i].grid(True, linestyle='--', alpha=0.5)
            fig.colorbar(
                c, ax=axs[i],
                label="Normalized Intensity" if i < 2 else "Difference"
            )

            if i in [0, 1]:
                extrema = (intensity_pred if i == 0 else intensity_real)
                axs[i].text(
                    0.5, -0.15,
                    f'Max: {extrema.max():.2f}\nMin: {extrema.min():.2f}',
                    transform=axs[i].transAxes,
                    ha='center', va='top', fontsize=14
                )

    # ---- main plotting logic ----
    plt.figure(figsize=(10, 6))
    x = sample_test_data()

    if args.dimensions == 1:
        plot_1d(x)
    else:
        plot_2d(x)

    plt.savefig(args.output_image, bbox_inches='tight')
    plt.close()


def plot_losses(
        train_losses, val_losses, norm_squared_list, 
        divergence_list, weight_list, log_density_list, 
        psi_x_list, args
):
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    # Subplot 1: Train Loss and Validation Loss
    color1 = 'tab:blue'
    color2 = 'tab:red'
    ax1 = axs[0]
    ax1.set_ylabel('Train Loss', color=color1)
    ax1.plot(epochs, train_losses, color=color1, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Loss', color=color2)
    ax2.plot(epochs, val_losses, color=color2, label='Validation Loss')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Training and Validation Loss over Epochs')

    # Subplot 2: norm_squared + log_density
    ax3 = axs[1]
    ax3.plot(epochs, norm_squared_list, label='Norm Squared', color='tab:green')
    ax3.set_ylabel('Norm Squared', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.grid(True)

    ax3b = ax3.twinx()
    ax3b.plot(epochs, log_density_list, label='Log Density', color='tab:brown', linestyle='--')
    ax3b.set_ylabel('Log Density', color='tab:brown')
    ax3b.tick_params(axis='y', labelcolor='tab:brown')

    # Subplot 3: divergence + log_density
    ax4 = axs[2]
    ax4.plot(epochs, divergence_list, label='Divergence', color='tab:orange')
    ax4.set_ylabel('Divergence', color='tab:orange')
    ax4.tick_params(axis='y', labelcolor='tab:orange')
    ax4.grid(True)

    ax4b = ax4.twinx()
    ax4b.plot(epochs, log_density_list, label='Log Density', color='tab:brown', linestyle='--')
    ax4b.set_ylabel('Log Density', color='tab:brown')
    ax4b.tick_params(axis='y', labelcolor='tab:brown')

    # Subplot 4: weight + log_density
    ax5 = axs[3]
    ax5.plot(epochs, weight_list, label='Weight', color='tab:purple')
    ax5.set_ylabel('Weight', color='tab:purple')
    ax5.tick_params(axis='y', labelcolor='tab:purple')
    ax5.grid(True)

    ax5b = ax5.twinx()
    ax5b.plot(epochs, log_density_list, label='Log Density', color='tab:brown', linestyle='--')
    ax5b.set_ylabel('Log Density', color='tab:brown')
    ax5b.tick_params(axis='y', labelcolor='tab:brown')

    # Subplot 5: psi_x + log_density
    ax6 = axs[4]
    ax6.plot(epochs, psi_x_list, label='Psi(x)', color='tab:cyan')
    ax6.set_ylabel('Psi(x)', color='tab:cyan')
    ax6.set_xlabel('Epoch')
    ax6.tick_params(axis='y', labelcolor='tab:cyan')
    ax6.grid(True)

    ax6b = ax6.twinx()
    ax6b.plot(epochs, log_density_list, label='Log Density', color='tab:brown', linestyle='--')
    ax6b.set_ylabel('Log Density', color='tab:brown')
    ax6b.tick_params(axis='y', labelcolor='tab:brown')

    plt.tight_layout()
    plt.savefig(args.loss_image)
