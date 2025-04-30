import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import WindowParams
from utils import remove_trailing_zeros
from weight_functions import distance_window, gaussian_window


def calculate_score_matching_difference(intensity_real, intensity_pred, dim):
    """
    Calculate the Score Matching Difference (SMD)
    between the real and predicted intensities.
    """
    intensity_real = np.asarray(intensity_real)
    intensity_pred = np.asarray(intensity_pred)
    if dim == 1:
        intensity_real = intensity_real.squeeze(-1)

    gradient_real = np.gradient(np.log(intensity_real))
    gradient_pred = np.gradient(np.log(intensity_pred))

    return np.sum((gradient_real - gradient_pred) ** 2).astype(np.float64)


def compute_smd(loader, model, args):
    smd_list = []
    intensity_stats = {'min': [], 'mean': [], 'max': []}

    kappa = args.kappa
    scale = args.dist_params.get("scale")

    for batch in loader:
        lengths = batch[0][:, 0, -1].to(dtype=torch.int64)
        cleaned_batch = remove_trailing_zeros(batch[0], lengths)

        for x_test in cleaned_batch:
            if args.dimensions == 1:
                x_test = x_test[:, 0].unsqueeze(1)
                lower, upper = args.region
                region_mask = (x_test >= lower) & (x_test <= upper)
            else:
                (x_lower, x_upper), (y_lower, y_upper) = args.region
                region_mask = (
                    (x_test[:, 0] >= x_lower) & (x_test[:, 0] <= x_upper) &
                    (x_test[:, 1] >= y_lower) & (x_test[:, 1] <= y_upper)
                )

            x_test_filtered = (
                x_test[region_mask].unsqueeze(1)
                if args.dimensions == 1
                else x_test[region_mask]
            )

            # Compute true intensity
            if args.dimensions == 1:
                intensity_real = kappa * torch.exp(
                    -x_test_filtered**2 / scale**2
                )
                model_input = x_test_filtered
            else:
                intensity_real = kappa * np.exp(
                    -(
                        x_test_filtered[:, 0]**2 + x_test_filtered[:, 1]**2
                    ) / scale**2
                )
                model_input = torch.tensor(x_test_filtered[:, :-1])

            # Predict intensity
            intensity_pred = torch.exp(model(model_input).detach()).squeeze(-1)

            # Normalize for score matching
            intensity_pred_norm = intensity_pred / intensity_pred.max()
            intensity_real_norm = intensity_real / intensity_real.max()

            # Calculate SMD
            smd = calculate_score_matching_difference(
                intensity_real_norm, intensity_pred_norm, args.dimensions
            )
            smd_list.append(smd)

            # Save intensity statistics
            intensity_stats['min'].append(intensity_pred.min().item())
            intensity_stats['mean'].append(intensity_pred.mean().item())
            intensity_stats['max'].append(intensity_pred.max().item())

    avg_smd = np.mean(smd_list)
    avg_intensity_stats = {
        'min': np.mean(intensity_stats['min']),
        'mean': np.mean(intensity_stats['mean']),
        'max': np.mean(intensity_stats['max']),
    }

    return avg_smd, avg_intensity_stats


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
            params = WindowParams(region=args.region, percent=args.percent)
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
        norm = mcolors.Normalize(vmin=0, vmax=1)

        fig, axs = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        titles = [
            r'Predicted Intensity $\rho(x)$',
            'True Intensity',
            'Difference Between Actual and Predicted Intensities'
        ]
        plots = [
            intensity_pred / intensity_pred.max(),
            intensity_real / intensity_real.max(),
            abs(
                (intensity_pred / intensity_pred.max()) -
                (intensity_real / intensity_real.max())
            )
        ]
        cmaps = ['viridis', 'viridis', 'cividis']

        (xmin, xmax), (ymin, ymax) = args.region
        opacity_mask = np.where(
            (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax), 1.0, 0.5
        )

        for i, (plot_data, title, cmap) in enumerate(zip(plots, titles, cmaps)):
            if isinstance(plot_data, torch.Tensor):
                plot_data = plot_data.numpy()

            if args.mirror_boundary:
                c = axs[i].imshow(
                    plot_data,
                    extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
                    origin='lower', cmap=cmap, alpha=opacity_mask,
                    aspect='auto', norm=norm if i < 2 else None
                )
            else:
                c = axs[i].contourf(
                    xx, yy, plot_data, levels=50, alpha=0.8,
                    cmap=cmap, norm=norm if i < 2 else None
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


def plot_loss_smd(train_losses, val_smds, args):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots()

    # Plot training losses on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Validation SMD', color=color)
    ax2.plot(epochs, val_smds, color=color, label='Validation SMD')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.grid(True)
    plt.title('Training Loss and Validation SMD over Epochs')
    fig.tight_layout()
    plt.savefig(args.loss_image)
