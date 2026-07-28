import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import apply_paper_style, masked_brain_imshow, add_shared_colorbar, SKULL_THRESH, add_panel_label, save_figure

import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Frozen 200-sample subset of reference_seed0/samples.npy
GEN_DATA = os.path.join(PAPER_DIR, "data", "fig3_samples_subset.npy")
GT_DIR = os.path.join(PAPER_DIR, "data", "fig3", "gt_samples")

REAL_COLOR = "#4477AA"
GEN_COLOR = "#CC6677"


# ══════════════════════════════════════════
# Sample grid (generated vs ground truth)
# ══════════════════════════════════════════
class ReverseAcousticNormalization(object):
    def __call__(self, tensor):
        tensor = torch.exp(tensor - 1.0)
        tensor = tensor * 3000.0
        return tensor


def postprocess_vp(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    tensor = transforms.Normalize(mean=[-1.0], std=[2.0])(tensor)
    tensor = ReverseAcousticNormalization()(tensor)
    return tensor


def resize_gt_like_training(matrix, target_dim=128):
    tensor = torch.from_numpy(matrix).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    resize_transform = transforms.Resize(
        (target_dim, target_dim),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resize_transform(tensor).squeeze().numpy()


def draw_velocity_grid(axes, gen_npy_path, gt_folder_path, n_samples=4,
                        cmap=None, vmin=1400, vmax=3000, mask=False,
                        flip_vertical=True, seed=0):
    """
    Draw the generated/ground-truth velocity grid onto an existing (2,
    n_samples) array of axes. Returns the last AxesImage drawn (for a
    shared colorbar).
    """
    rng = random.Random(seed)

    if not os.path.exists(gen_npy_path):
        raise FileNotFoundError(f"Generated file not found: {gen_npy_path}")
    gen_samples = np.load(gen_npy_path)

    gt_files = [os.path.join(gt_folder_path, f) for f in os.listdir(gt_folder_path)
                if f.endswith(".npy")]
    if len(gt_files) < n_samples:
        raise ValueError(f"Need at least {n_samples} files in GT folder, found {len(gt_files)}.")
    if len(gen_samples) < n_samples:
        raise ValueError(f"Need at least {n_samples} generated samples, found {len(gen_samples)}.")

    gen_indices = rng.sample(range(len(gen_samples)), n_samples)
    gt_paths = rng.sample(gt_files, n_samples)

    mask_arg = None if mask else False  # None -> masked_brain_imshow's own
                                         # Guasch default; False -> disabled

    im = None
    for col, gen_idx in enumerate(gen_indices):
        ax = axes[0, col]
        data_ms = postprocess_vp(gen_samples[gen_idx]).squeeze().numpy()
        if flip_vertical:
            data_ms = np.flipud(data_ms)
        im = masked_brain_imshow(ax, data_ms, mask=mask_arg, vmin=vmin, vmax=vmax,
                                  cmap=cmap, aspect="equal", interpolation="nearest")
        ax.axis("off")
        if col == 0:
            ax.text(-0.12, 0.5, "Generated", transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=12, weight="bold")

    for col, gt_path in enumerate(gt_paths):
        ax = axes[1, col]
        raw_gt = np.load(gt_path).squeeze()
        if raw_gt.ndim > 2:
            raw_gt = raw_gt[0]
        data_ms = resize_gt_like_training(raw_gt, target_dim=128)
        if flip_vertical:
            data_ms = np.flipud(data_ms)
        im = masked_brain_imshow(ax, data_ms, mask=mask_arg, vmin=vmin, vmax=vmax,
                                  cmap=cmap, aspect="equal", interpolation="nearest")
        ax.axis("off")
        if col == 0:
            ax.text(-0.12, 0.5, "Ground Truth", transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=12, weight="bold")

    return im


def plot_velocity_comparison(gen_npy_path, gt_folder_path, n_samples=4,
                              cmap=None, vmin=None, vmax=None, mask=True,
                              flip_vertical=True, seed=0, save_dir=None,
                              fig_name="fig3_sample_grid",
                              formats=("pdf", "svg"), show=True):
    """Standalone sample grid (Fig 3, left half only). Returns (fig, axes)."""
    apply_paper_style()

    fig, axes = plt.subplots(2, n_samples, figsize=(2.0 * n_samples, 4.2))
    fig.subplots_adjust(wspace=0.06, hspace=0.08)

    im = draw_velocity_grid(axes, gen_npy_path, gt_folder_path, n_samples=n_samples,
                             cmap=cmap, vmin=vmin, vmax=vmax, mask=mask,
                             flip_vertical=flip_vertical, seed=seed)

    add_shared_colorbar(fig, im, axes, label="Velocity (m/s)")

    if save_dir:
        save_figure(fig, fig_name, save_dir, close=False, formats=formats)
    if show:
        plt.show()
    return fig, axes


# ════════════════════════════
# Tissue / skull histograms
# ════════════════════════════
def load_real_histogram_volumes(data_dir, true_model, x_dim=128, augment=False):
    """
    Returns a list of 2D numpy arrays (m/s), one per val+test sample.
    """
    from diffusionfwi.dataset import build_dataset, postprocess_vp as pp_vp

    _, val_dataset, test_dataset = build_dataset(
        data_dir, true_model=true_model, x_dim=x_dim, augment=augment,
    )
    val_data = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    test_data = torch.stack([test_dataset[i] for i in range(len(test_dataset))])
    all_images = torch.cat([val_data, test_data], dim=0)  # (N, 1, H, W), network-input space

    real_ms = pp_vp(all_images)  # (N, 1, H, W), back to m/s
    real_ms = F.interpolate(real_ms, size=(320, 256), mode='bilinear', align_corners=True)

    return [real_ms[i, 0].detach().cpu().numpy() for i in range(real_ms.shape[0])]


def load_generated_histogram_volumes(generated_npy_path, n_samples=200):
    """
    Returns a list of 2D numpy arrays (m/s), one per selected sample.
    """
    from diffusionfwi.dataset import postprocess_vp as pp_vp

    generated_images = np.load(generated_npy_path)
    gen_tensor = torch.from_numpy(generated_images).float()
    if gen_tensor.ndim == 3:               # (N, H, W) -> (N, 1, H, W)
        gen_tensor = gen_tensor.unsqueeze(1)
    elif gen_tensor.ndim != 4:
        raise ValueError(f"Expected samples.npy to be 3D or 4D, got shape {gen_tensor.shape}")

    selected = gen_tensor[:n_samples]
    gen_ms = pp_vp(selected)  # (n_samples, 1, H, W), back to m/s
    gen_ms = F.interpolate(gen_ms, size=(320, 256), mode='bilinear', align_corners=True)

    return [gen_ms[i, 0].detach().cpu().numpy() for i in range(gen_ms.shape[0])]


def center_crop(image_2d, crop_size):
    """Center crop a single 2D array."""
    H, W = image_2d.shape
    start_y = (H - crop_size) // 2
    start_x = (W - crop_size) // 2
    return image_2d[start_y:start_y + crop_size, start_x:start_x + crop_size]


def collect_tissue_values(volumes, crop_size=110):
    """Soft-tissue pixel values across a batch: center_crop + flatten, pooled."""
    return np.concatenate([center_crop(vp, crop_size).ravel() for vp in volumes])


def collect_skull_values(volumes, skull_thresh=SKULL_THRESH):
    """Skull pixel values across a batch: vp > skull_thresh, pooled."""
    return np.concatenate([vp[vp > skull_thresh] for vp in volumes])


def draw_tissue_skull_histograms(ax_tissue, ax_skull, real_volumes, gen_volumes,
                                  crop_size=110,
                                  tissue_bin_range=(1400, 1650), tissue_bins=100,
                                  tissue_xlim=(1480, 1650), tissue_ylim=0.03,
                                  skull_thresh=SKULL_THRESH, skull_bins=100, skull_ylim=0.002,
                                  panel_labels=None,
                                  precomputed_values=None):
    """
    Draw the soft-tissue (ax_tissue) and skull (ax_skull) histograms onto
    a pair of existing axes.

    precomputed_values : optional dict with keys "real_tissue", "gen_tissue",
        "real_skull", "gen_skull" (1D pixel-value arrays)
    """
    if precomputed_values is not None:
        real_tissue = precomputed_values["real_tissue"]
        gen_tissue = precomputed_values["gen_tissue"]
        real_skull = precomputed_values["real_skull"]
        gen_skull = precomputed_values["gen_skull"]
    else:
        real_tissue = collect_tissue_values(real_volumes, crop_size)
        gen_tissue = collect_tissue_values(gen_volumes, crop_size)
        real_skull = collect_skull_values(real_volumes, skull_thresh)
        gen_skull = collect_skull_values(gen_volumes, skull_thresh)

    # --- soft tissue ---
    tissue_edges = np.linspace(tissue_bin_range[0], tissue_bin_range[1], tissue_bins + 1)
    ax_tissue.hist(real_tissue, bins=tissue_edges, density=True, color=REAL_COLOR,
                   histtype="step", linewidth=1.5, label="Real", zorder=3)
    ax_tissue.hist(real_tissue, bins=tissue_edges, density=True, color=REAL_COLOR,
                   histtype="stepfilled", alpha=0.12, edgecolor="none", zorder=2)

    ax_tissue.hist(gen_tissue, bins=tissue_edges, density=True, color=GEN_COLOR,
                   histtype="step", linewidth=1.5, label="Generated", zorder=3)
    ax_tissue.hist(gen_tissue, bins=tissue_edges, density=True, color=GEN_COLOR,
                   histtype="stepfilled", alpha=0.12, edgecolor="none", zorder=2)

    ax_tissue.set_xlim(tissue_xlim)
    ax_tissue.set_ylim(top=tissue_ylim)
    ax_tissue.set_title("Soft tissue", fontsize=12, pad=6, fontweight="bold")
    ax_tissue.set_xlabel("Velocity (m/s)", fontsize=10)
    ax_tissue.set_ylabel(r"Density", fontsize=10)

    ax_tissue.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    ax_tissue.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_tissue.yaxis.get_offset_text().set_visible(True)
    ax_tissue.yaxis.get_offset_text().set_fontsize(7)

    ax_tissue.legend(frameon=False, loc="upper right", fontsize=7)

    # --- skull ---
    combined = np.concatenate([real_skull, gen_skull])
    skull_edges = np.linspace(combined.min(), combined.max(), skull_bins + 1)
    ax_skull.hist(real_skull, bins=skull_edges, density=True, color=REAL_COLOR,
                  histtype="step", linewidth=1.5, label="Real", zorder=3)
    ax_skull.hist(real_skull, bins=skull_edges, density=True, color=REAL_COLOR,
                  histtype="stepfilled", alpha=0.12, edgecolor="none", zorder=2)

    ax_skull.hist(gen_skull, bins=skull_edges, density=True, color=GEN_COLOR,
                  histtype="step", linewidth=1.5, label="Generated", zorder=3)
    ax_skull.hist(gen_skull, bins=skull_edges, density=True, color=GEN_COLOR,
                  histtype="stepfilled", alpha=0.12, edgecolor="none", zorder=2)

    ax_skull.set_ylim(top=skull_ylim)
    ax_skull.set_title("Skull", fontsize=12, pad=6, fontweight="bold")
    ax_skull.set_xlabel("Velocity (m/s)", fontsize=10)
    ax_skull.set_ylabel(r"Density", fontsize=10)

    ax_skull.yaxis.set_major_locator(ticker.MultipleLocator(0.0005))
    ax_skull.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_skull.yaxis.get_offset_text().set_visible(True)
    ax_skull.yaxis.get_offset_text().set_fontsize(7)

    ax_skull.legend(frameon=False, loc="upper right", fontsize=7)

    if panel_labels:
        add_panel_label(ax_tissue, panel_labels[0])
        add_panel_label(ax_skull, panel_labels[1])


def plot_tissue_skull_histograms(real_volumes, gen_volumes,
                                  crop_size=110,
                                  tissue_bin_range=(1400, 1700), tissue_bins=100,
                                  tissue_xlim=(1450, 1700), tissue_ylim=0.03,
                                  skull_thresh=SKULL_THRESH, skull_bins=100, skull_ylim=0.005,
                                  save_dir=None, fig_name="fig3_tissue_skull_hist",
                                  formats=("pdf", "svg"), show=True):
    """Standalone tissue/skull histogram pair (Fig 3, right half only)."""
    apply_paper_style()

    print(f"Real volumes: {len(real_volumes)} samples, shape {real_volumes[0].shape}")
    print(f"Generated volumes: {len(gen_volumes)} samples, shape {gen_volumes[0].shape}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    draw_tissue_skull_histograms(
        axes[0], axes[1], real_volumes, gen_volumes,
        crop_size=crop_size, tissue_bin_range=tissue_bin_range, tissue_bins=tissue_bins,
        tissue_xlim=tissue_xlim, tissue_ylim=tissue_ylim, skull_thresh=skull_thresh,
        skull_bins=skull_bins, skull_ylim=skull_ylim,
    )
    plt.tight_layout()

    if save_dir:
        save_figure(fig, fig_name, save_dir, close=False, formats=formats)
    if show:
        plt.show()
    return fig, axes


# ═════════════════════════════════════════════
# Combined figure — sample grid + histograms
# ═════════════════════════════════════════════
def plot_fig3_combined(gen_npy_path, gt_folder_path, real_volumes, gen_volumes,
                        n_samples=4, hist_col_width=1.8,
                        grid_kwargs=None, hist_kwargs=None,
                        save_dir=None, fig_name="fig3_combined",
                        formats=("pdf", "svg"), show=True,
                        precomputed_values=None):
    """
    Plot brain grid on the left, tissue/skull histograms on
    the right.

    Returns (fig, brain_axes, (ax_tissue, ax_skull)).
    """
    apply_paper_style()
    grid_kwargs = grid_kwargs or {}
    hist_kwargs = hist_kwargs or {}
    hist_kwargs["panel_labels"] = None
    hist_kwargs["precomputed_values"] = precomputed_values

    cbar_col_width = 0.10
    spacer_width = 0.65
    outer_widths = [n_samples, cbar_col_width, spacer_width, hist_col_width, hist_col_width]
    total_units = sum(outer_widths)
    fig_width = 2.0 * total_units 
    fig = plt.figure(figsize=(fig_width, 4.2))
    outer_gs = fig.add_gridspec(nrows=1, ncols=5, width_ratios=outer_widths, wspace=0.25)

    brain_gs = outer_gs[0].subgridspec(nrows=2, ncols=n_samples, wspace=0.06, hspace=0.08)
    brain_axes = np.empty((2, n_samples), dtype=object)
    for row in range(2):
        for col in range(n_samples):
            brain_axes[row, col] = fig.add_subplot(brain_gs[row, col])

    im = draw_velocity_grid(brain_axes, gen_npy_path, gt_folder_path,
                             n_samples=n_samples, **grid_kwargs)

    cax = fig.add_subplot(outer_gs[0, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label("Velocity (m/s)", fontsize=10, labelpad=4)

    ax_tissue = fig.add_subplot(outer_gs[0, 3])
    ax_skull = fig.add_subplot(outer_gs[0, 4])
    draw_tissue_skull_histograms(ax_tissue, ax_skull, real_volumes, gen_volumes, **hist_kwargs)

    fig.canvas.draw()  

    pos_brain_left = brain_axes[0, 0].get_position()
    pos_brain_right = brain_axes[0, -1].get_position()

    brain_x0 = pos_brain_left.x0
    brain_x1 = pos_brain_right.x1
    brain_y0 = brain_axes[1, 0].get_position().y0
    brain_y1 = pos_brain_left.y1
    brain_height = brain_y1 - brain_y0

    cbar_width = 0.01
    cbar_x0 = brain_x1 + 0.012
    cax.set_position([cbar_x0, brain_y0, cbar_width, brain_height])
    cax.set_axes_locator(None)

    pos_tissue = ax_tissue.get_position()
    pos_skull = ax_skull.get_position()

    tissue_x0 = cbar_x0 + 0.1
    ax_tissue.set_position([tissue_x0, brain_y0, pos_tissue.width, brain_height])
    ax_tissue.set_axes_locator(None)

    skull_x0 = tissue_x0 + pos_tissue.width + 0.07
    ax_skull.set_position([skull_x0, brain_y0, pos_skull.width, brain_height])
    ax_skull.set_axes_locator(None)

    y_label_baseline = brain_y1 + 0.028
    fig.text(brain_x0 - 0.012, y_label_baseline, "A", fontsize=14, fontweight="bold", va="bottom", ha="right")
    fig.text(tissue_x0 - 0.025, y_label_baseline, "B", fontsize=14, fontweight="bold", va="bottom", ha="right")
    fig.text(skull_x0 - 0.025, y_label_baseline, "C", fontsize=14, fontweight="bold", va="bottom", ha="right")

    if save_dir:
        save_figure(fig, fig_name, save_dir, close=False, formats=formats)
    if show:
        plt.show()
    return fig, brain_axes, (ax_tissue, ax_skull)


def main(save_dir=None, gen_npy_path=GEN_DATA, gt_folder_path=GT_DIR, show=False,
         precomputed_histogram_dir=None):
    """
    precomputed_histogram_dir : directory containing real_tissue_values.npy /
        real_skull_values.npy (frozen, since real GT data isn't otherwise
        available without cluster access)
    """
    if save_dir is None:
        save_dir = os.path.join(PAPER_DIR, "results", "figures")

    gen_volumes = load_generated_histogram_volumes(generated_npy_path=gen_npy_path, n_samples=200)

    if precomputed_histogram_dir:
        real_volumes = None
        precomputed_values = {
            "real_tissue": np.load(os.path.join(precomputed_histogram_dir, "real_tissue_values.npy")),
            "real_skull": np.load(os.path.join(precomputed_histogram_dir, "real_skull_values.npy")),
            "gen_tissue": collect_tissue_values(gen_volumes),
            "gen_skull": collect_skull_values(gen_volumes),
        }
    else:
        real_volumes = load_real_histogram_volumes(data_dir=gt_folder_path, true_model="vp_996782.npy", x_dim=128)
        precomputed_values = None

    return plot_fig3_combined(
        gen_npy_path=gen_npy_path, gt_folder_path=gt_folder_path,
        real_volumes=real_volumes, gen_volumes=gen_volumes,
        save_dir=save_dir, fig_name="fig3_samples", formats=("pdf", "svg"), show=show,
        precomputed_values=precomputed_values,
    )


if __name__ == "__main__":
    main()
