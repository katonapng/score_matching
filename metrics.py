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
        intensity_pred = intensity_pred.squeeze(-1)

    gradient_real = np.gradient(np.log(intensity_real))
    gradient_pred = np.gradient(np.log(intensity_pred))

    return np.sum((gradient_real - gradient_pred) ** 2).astype(np.float64)


def compute_smd(loader_test, model, args):
    smd_list = []

    kappa = args.kappa
    scale = args.dist_params.get("scale")

    for batch in loader_test:
        lengths = batch[0][:, 0, -1].to(dtype=torch.int64)
        cleaned_batch = remove_trailing_zeros(batch[0], lengths)
        for x_test in cleaned_batch:
            if args.dimensions == 1:
                x_test = x_test[:, 0].unsqueeze(1)
                lower, upper = args.region
                region_mask = (x_test >= lower) & (x_test <= upper)
                x_test_filtered = x_test[region_mask].unsqueeze(1)
                intensity_real = kappa * torch.exp(-x_test_filtered**2 / scale**2)
                intensity_pred = torch.exp(model(x_test_filtered).detach())
            else:
                (x_lower, x_upper), (y_lower, y_upper) = args.region
                region_mask = (
                    (x_test[:, 0] >= x_lower) & (x_test[:, 0] <= x_upper)
                    & (x_test[:, 1] >= y_lower) & (x_test[:, 1] <= y_upper)
                )
                x_test_filtered = x_test[region_mask]
                intensity_real = kappa * np.exp(
                    -(
                        x_test_filtered[:, 0]**2 + x_test_filtered[:, 1]**2
                    ) / scale**2
                )
                intensity_pred = torch.exp(model(
                    torch.tensor(x_test_filtered[:, :-1], dtype=torch.float32),
                )).squeeze(-1).detach()

            intensity_pred /= torch.max(intensity_pred)
            intensity_real /= torch.max(intensity_real)

            smd = calculate_score_matching_difference(
                intensity_real, intensity_pred, args.dimensions,
            )

            smd_list.append(smd)

    return np.mean(smd_list)


def plot_results(args, model, test_loader):
    plt.figure(figsize=(10, 6))
    random_test_sample = test_loader.random_sample()[0]
    lengths = random_test_sample[:, 0, -1].to(dtype=torch.int64)
    x = remove_trailing_zeros(random_test_sample, lengths)[0]

    kappa = args.kappa
    scale = args.dist_params.get("scale")

    if args.dimensions == 1:
        x = x[:, 0]
        x_min, x_max = x.min(), x.max()
        x_lin = np.linspace(args.region[0], args.region[1], 100)
        intensity_pred = model(
            torch.tensor(x_lin[:, None], dtype=torch.float32),
        ).squeeze().detach()
        intensity_pred = torch.exp(intensity_pred)

        def mirrored_intensity(x, a, b, kappa, scale):
            length = b - a
            x_relative = (x - a) % (2 * length)
            x_mirrored = np.where(
                x_relative <= length, a + x_relative, b - (x_relative - length)
            )
            return kappa * np.exp(-x_mirrored**2 / scale**2)

        if args.mirror_boundary:
            intensity_real = mirrored_intensity(
                x_lin[:, None], a=args.region[0], b=args.region[1],
                kappa=kappa, scale=scale
            )
        else:
            intensity_real = kappa * np.exp(-x_lin[:, None]**2 / scale**2)

        if args.weight_function.__name__ == "gaussian_window":
            temp = torch.tensor(
                x_lin[:, None], dtype=torch.float32
            ).unsqueeze(0)
            lengths = torch.tensor([temp.shape[1]], dtype=torch.int32)
            params = WindowParams(lengths=lengths)
            x_gaussian = gaussian_window(temp, params)

            plt.plot(
                x_lin,
                x_gaussian.squeeze(0),
                label='Gaussian Window',
                color='orange',
                linestyle='--'
            )

        if args.weight_function.__name__ == "distance_window":
            temp = torch.tensor(
                x_lin[:, None], dtype=torch.float32
            ).unsqueeze(0)
            params = WindowParams(region=args.region, percent=args.percent)
            x_distance = distance_window(temp, params)

            plt.plot(
                x_lin,
                x_distance.squeeze(0),
                label='Distance Window',
                color='orange',
                linestyle='--'
            )

        plt.plot(
            x_lin,
            intensity_pred / torch.max(intensity_pred),
            label='Predicted Intensity',
            color='blue',
        )
        plt.plot(
            x_lin,
            intensity_real / intensity_real.max(),
            label='True Intensity',
            color='green'
        )
        plt.scatter(
            x, np.zeros_like(x),
            c='red', s=10, alpha=0.6, label='Poisson Points'
        )

        if args.mirror_boundary:
            plt.axvspan(
                x_min, args.region[0], color='gray',
                alpha=0.2, label='Mirrored Region',
            )
            plt.axvspan(args.region[1], x_max, color='gray', alpha=0.2)

        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(fr'Intensity ($\kappa$ = {kappa:.2f})', fontsize=14)
        plt.title('Intensity', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(args.region[0], args.region[1])
        plt.legend()

    else:
        x, y = x[:, 0].numpy(), x[:, 1].numpy()
        x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
        x_lin = np.linspace(x_min, x_max, 100)
        y_lin = np.linspace(y_min, y_max, 100)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        intensity_pred = model(
            torch.tensor(grid_points, dtype=torch.float32),
        ).detach()
        intensity_pred = torch.exp(intensity_pred)
        intensity_pred = intensity_pred.reshape(xx.shape)
        intensity_real = kappa * np.exp(-(xx**2 + yy**2) / scale**2)

        norm = mcolors.Normalize(vmin=0, vmax=1)
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))

        titles = [
            r'Predicted Intensity $\rho(x)$',
            'True Intensity',
            'Difference Between Actual and Predicted Intensities'
        ]
        plots = [
            intensity_pred / torch.max(intensity_pred),
            intensity_real / intensity_real.max(),
            abs(
                (intensity_pred / torch.max(intensity_pred))
                - (intensity_real / np.max(intensity_real))
            )
        ]
        # plots = [
        #     intensity_pred,
        #     intensity_real,
        #     abs(
        #         (intensity_pred / torch.max(intensity_pred))
        #         - (intensity_real / np.max(intensity_real))
        #     )
        # ]
        cmaps = ['viridis', 'viridis', 'cividis']

        (xmin, xmax), (ymin, ymax) = args.region
        opacity_mask = np.where(
            (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax),
            1.0, 0.5
        )

        for i in range(3):
            plot_data = plots[i]

            # If it's a torch tensor, convert to numpy
            if isinstance(plot_data, torch.Tensor):
                plot_data = plot_data.numpy()

            if args.mirror_boundary:
                c = axs[i].imshow(
                    plot_data,
                    extent=[x_min, x_max, y_min, y_max],
                    origin='lower',
                    cmap=cmaps[i],
                    alpha=opacity_mask,
                    aspect='auto',
                    norm=norm if i < 2 else None
                )
            else:
                c = axs[i].contourf(
                    xx, yy, plot_data,
                    levels=50,
                    alpha=0.8,
                    cmap=cmaps[i],
                    norm=norm if i < 2 else None
                )

            axs[i].scatter(
                x, y, c='red', s=5, alpha=0.6, label='Poisson Points'
            )
            axs[i].set_title(titles[i], fontsize=16)
            axs[i].set_xlabel(r'$x$', fontsize=14)
            axs[i].set_ylabel(r'$y$', fontsize=14)
            axs[i].legend()
            axs[i].set_aspect('equal')
            axs[i].grid(True, linestyle='--', alpha=0.5)
            fig.colorbar(
                c, ax=axs[i],
                label="Normalized Intensity" if i < 2 else "Difference",
            )

        plt.tight_layout()

    plt.savefig(args.output_image)
    plt.close()
