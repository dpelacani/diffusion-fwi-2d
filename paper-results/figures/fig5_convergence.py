import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import apply_paper_style, save_figure

import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.ndimage import binary_closing, binary_fill_holes, label, binary_erosion, zoom
from skimage.morphology import disk
from matplotlib.gridspec import GridSpecFromSubplotSpec

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# EXPERIMENTS / METHODS dict keys
SCENARIOS = ["standard", "reduced_compute", "missing_low_freq"]
METHODS = ["baseline", "dg", "dg_c", "sg", "sg_c"]

SCENARIO_RUN_DIR_SUFFIX = {
    "standard":         "reference_v2",
    "reduced_compute":  "restricted",
    "missing_low_freq": "missing_low_freq_v2",
}
METHOD_RUN_DIR_SUFFIX = {
    "baseline": "baseline",
    "dg":       "postprocessing_merged",
    "dg_c":     "postprocessing_split",
    "sg":       "gradient_merged",
    "sg_c":     "gradient_split",
}

SCENARIO_LABELS = {
    "standard":         "Standard",
    "reduced_compute":  "Reduced compute",
    "missing_low_freq": "Missing low frequencies",
}

CUSTOM_COLORS = {
    "baseline": "#1e293b",  
    "dg":       "#7bc0e3",  
    "dg_c":     "#2b6cb0",  
    "sg":       "#eeb987",  
    "sg_c":     "#c66a4e", 
}

METHOD_LABELS = {
    "baseline": "Base FWI",
    "dg":       "DiffFWI-Refine",
    "dg_c":     "DiffFWI-Refine-C",
    "sg":       "DiffFWI-Score",
    "sg_c":     "DiffFWI-Score-C",
}

DEFAULT_SEARCH_DIRS = ["/cluster/scratch/fscharitzer/inversion/final_runs_reference"]
DEFAULT_FINAL_ITER = 80


# ==================================
# METRIC convergence criterion
# ==================================
def make_tissue_skull_masks(gt_2d, name, skull_thresh_ms=1650.0, closing_radius=5,
                             tissue_erosion=10, skull_erosion=3,
                             manual_closing_radius=None, manual_patches=None):
    manual_closing_radius = manual_closing_radius if manual_closing_radius is not None else \
        {"vp_182032": 13, "vp_268850": 7}
    manual_patches = manual_patches if manual_patches is not None else \
        {"vp_170934": [(slice(225, 240), slice(208, 212))]}

    closing_r = manual_closing_radius.get(name, closing_radius)
    patches = manual_patches.get(name, [])
    skull_raw = (gt_2d >= skull_thresh_ms).copy()
    for rs, cs in patches:
        skull_raw[rs, cs] = True

    closed = binary_closing(skull_raw, structure=disk(closing_r))
    filled = binary_fill_holes(closed)
    tissue_raw = filled & ~skull_raw
    labeled, n = label(tissue_raw)
    if n == 0:
        tissue_mask = np.zeros_like(skull_raw, dtype=bool)
        valid = False
    else:
        if n > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            tissue_clean = (labeled == sizes.argmax())
        else:
            tissue_clean = tissue_raw.astype(bool)
        tissue_mask = binary_erosion(tissue_clean, structure=disk(tissue_erosion))
        valid = bool(tissue_mask.any())

    skull_mask = binary_erosion(closed, structure=disk(skull_erosion)) \
        if skull_erosion > 0 else closed
    valid = valid and bool(skull_mask.any())

    return tissue_mask.astype(bool), skull_mask.astype(bool), valid


def load_fwi(name, scenario, method, search_dirs, final_iter):
    scenario_suffix = SCENARIO_RUN_DIR_SUFFIX[scenario]
    method_suffix = METHOD_RUN_DIR_SUFFIX[method]
    for base_dir in search_dirs:
        run_dir = os.path.join(base_dir, f"{name}_{scenario_suffix}_{method_suffix}")
        if not os.path.isfile(os.path.join(run_dir, "done")):
            continue
        path = os.path.join(run_dir, f"{name}-Vp-{final_iter:05d}.h5")
        if not os.path.isfile(path):
            candidates = sorted(glob.glob(os.path.join(run_dir, f"{name}-Vp-*.h5")))
            if not candidates:
                continue
            path = candidates[-1]
        with h5py.File(path, "r") as f:
            return f["data"][()].astype(np.float32)
    return None


def load_gt(name, gt_dir):
    path = os.path.join(gt_dir, f"{name}.npy")
    return np.load(path).astype(np.float32) if os.path.isfile(path) else None


def align(recon, gt):
    if recon.shape == gt.shape:
        return recon
    zf = (gt.shape[0] / recon.shape[0], gt.shape[1] / recon.shape[1])
    return zoom(recon, zf, order=1)


def pct_within(gt_vals, recon_vals, tol):
    if len(gt_vals) == 0:
        return float("nan")
    return float((np.abs(recon_vals - gt_vals) <= tol).mean() * 100)


def compute_convergence_metrics(models_file, gt_dir, search_dirs, final_iter=DEFAULT_FINAL_ITER,
                                 scenarios=SCENARIOS, methods=METHODS,
                                 tissue_tolerance=30, skull_tolerance=250,
                                 mask_kwargs=None):
    """
    Compute tissue_pct/skull_pct for every (brain, scenario, method) with a
    completed FWI run.

    Returns a DataFrame with columns: brain, scenario, method, tissue_pct, skull_pct.
    """
    mask_kwargs = mask_kwargs or {}

    with open(models_file) as f:
        brains = [l.strip() for l in f if l.strip()]

    rows = []
    for brain in brains:
        gt = load_gt(brain, gt_dir)
        if gt is None:
            continue
        tissue_mask, skull_mask, valid = make_tissue_skull_masks(gt, brain, **mask_kwargs)
        if not valid:
            continue

        for scenario in scenarios:
            for method in methods:
                recon = load_fwi(brain, scenario, method, search_dirs, final_iter)
                if recon is None:
                    continue
                recon = align(recon, gt)
                t_pct = pct_within(gt[tissue_mask], recon[tissue_mask], tissue_tolerance)
                s_pct = pct_within(gt[skull_mask], recon[skull_mask], skull_tolerance)
                rows.append({"brain": brain, "scenario": scenario, "method": method,
                             "tissue_pct": t_pct, "skull_pct": s_pct})

    return pd.DataFrame(rows)


# ============
# PLOT                                                          
# ============
def plot_convergence_scatter(df, tissue_tolerance=30, skull_tolerance=250,
                              tissue_threshold=50, tissue_buffer=(40, 60),
                              skull_threshold=50, skull_buffer=(40, 60),
                              save_dir=None, fig_name="fig5_convergence_metric",
                              formats=("pdf", "svg"), show=True):
    """
    Plot tissue% x skull% (within tolerance), faceted by scenario, colored
    by method, with marginal histograms and the threshold from the
    convergence criterion.
    """
    apply_paper_style()

    present_scenarios = [s for s in SCENARIOS if s in df["scenario"].unique()]
    n_sc = len(present_scenarios)
    fig = plt.figure(figsize=(4.8 * n_sc, 5.2))
    outer = fig.add_gridspec(1, n_sc, wspace=0.22, left=0.06, right=0.97, top=0.93, bottom=0.16)

    axes_dict = {}
    top_axes = []
    right_axes = []

    for i, scenario in enumerate(present_scenarios):
        sub = df[df["scenario"] == scenario].dropna(subset=["tissue_pct", "skull_pct"])

        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[i],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.11, wspace=0.11,
        )

        ax_top = fig.add_subplot(inner[0, 0])
        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")
        ax_main = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])

        top_axes.append(ax_top)
        right_axes.append(ax_right)

        ax_top.sharex(ax_main)
        ax_right.sharey(ax_main)
        axes_dict[scenario] = {"main": ax_main, "top": ax_top, "right": ax_right}

        bins = np.linspace(0, 100, 35)

        for method_raw in sub["method"].unique():
            color = CUSTOM_COLORS.get(method_raw, "#7f8c8d")
            d = sub[sub["method"] == method_raw]

            ax_main.scatter(d["tissue_pct"], d["skull_pct"], c=color, s=14,
                             alpha=0.65, edgecolors="none", zorder=5)

            ax_top.hist(d["tissue_pct"].dropna(), bins=bins, color=color,
                        histtype="step", linewidth=0.9, alpha=0.6, zorder=4)
            ax_right.hist(d["skull_pct"].dropna(), bins=bins, color=color,
                          histtype="step", linewidth=0.9, alpha=0.6, orientation="horizontal", zorder=4)
            ax_top.hist(d["tissue_pct"].dropna(), bins=bins, color=color,
                        histtype="stepfilled", edgecolor="none", alpha=0.07, zorder=3)
            ax_right.hist(d["skull_pct"].dropna(), bins=bins, color=color,
                          histtype="stepfilled", edgecolor="none", alpha=0.07,
                          orientation="horizontal", zorder=3)

        ax_main.axvline(tissue_threshold, color="#94a3b8", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
        ax_main.axhline(skull_threshold, color="#94a3b8", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)

        ax_top.axvline(tissue_threshold, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.3)
        ax_right.axhline(skull_threshold, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.3)

        if i == 0:
            ax_main.text(tissue_threshold + 2.5, 95, "Threshold (50%)",
                        fontsize=9, color="#8892b0", va="center", ha="left", style="italic")
            ax_main.text(2.5, skull_threshold + 5, "Threshold (50%)",
                        fontsize=9, color="#8892b0", va="bottom", ha="left", style="italic")

        ax_main.scatter([100], [100], marker="*", s=200, color="#f1c40f",
                        edgecolors="none", linewidth=0.7, zorder=100, clip_on=False)

        ax_main.set_xlim(0, 100)
        ax_main.set_ylim(0, 100)
        ax_main.set_xlabel(f"Soft tissue within {tissue_tolerance} m/s of GT (%)", fontsize=12)
        ax_main.set_ylabel(f"Skull within {skull_tolerance} m/s of GT (%)" if i == 0 else "", fontsize=12)

        ax_main.spines["top"].set_visible(False)
        ax_main.spines["right"].set_visible(False)

        for spine in ["top", "right", "left", "bottom"]:
            ax_top.spines[spine].set_visible(False)
            ax_right.spines[spine].set_visible(False)

        ax_top.spines["bottom"].set_visible(True)
        ax_top.spines["bottom"].set_color("#cbd5e1")
        ax_right.spines["left"].set_visible(True)
        ax_right.spines["left"].set_color("#cbd5e1")

        ax_top.set_ylabel("Count", fontsize=9, labelpad=2)
        ax_right.set_xlabel("Count", fontsize=9, labelpad=2)

        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
        ax_top.tick_params(axis="both", which="both", length=0, labelsize=7)
        ax_right.tick_params(axis="both", which="both", length=0, labelsize=7)
        ax_top.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=16, pad=8, fontweight="bold")

    global_max_count = max(
        max(ax.get_ylim()[1] for ax in top_axes),
        max(ax.get_xlim()[1] for ax in right_axes)
    )

    for ax_t, ax_r in zip(top_axes, right_axes):
        ax_t.set_ylim(0, global_max_count)
        ax_r.set_xlim(0, global_max_count)
        ax_t.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, integer=True))
        ax_r.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, integer=True))

    legend_handles = [mpatches.Patch(color=c, label=METHOD_LABELS.get(m, m)) for m, c in CUSTOM_COLORS.items()]
    star_proxy = mlines.Line2D([], [], color="none", marker="*", linestyle="None",
                                markersize=11, markerfacecolor="#f1c40f", markeredgecolor="none",
                                markeredgewidth=0.7, label="Target convergence")
    legend_handles.append(star_proxy)

    fig.legend(handles=legend_handles, frameon=False, fontsize=12, loc="upper center",
               ncol=len(legend_handles), bbox_to_anchor=(0.5, 0.03), handletextpad=0.5,
               columnspacing=1.6)

    if save_dir:
        save_figure(fig, fig_name, save_dir, close=False, formats=formats)
    if show:
        plt.show()
    return fig, axes_dict


def main(search_dirs=DEFAULT_SEARCH_DIRS, final_iter=DEFAULT_FINAL_ITER,
         models_file=None, gt_dir="/cluster/scratch/fscharitzer/fwi-data/Ultrasound-Vp-axial-models",
         save_dir=None, show=False, use_precomputed=None):

    if save_dir is None:
        save_dir = os.path.join(PAPER_DIR, "results", "figures")

    if use_precomputed:
        df = pd.read_csv(use_precomputed)
    else:
        if models_file is None:
            models_file = os.path.join(os.path.dirname(PAPER_DIR), "fwi_brains.txt")

        SWEEP_BRAINS = {
            "vp_121416", "vp_177746", "vp_178950", "vp_268749", "vp_385046",
            "vp_395251", "vp_580347", "vp_581450", "vp_732243", "vp_922854",
        }

        df = compute_convergence_metrics(models_file, gt_dir, search_dirs, final_iter=final_iter,
                                          scenarios=SCENARIOS, methods=METHODS)
        df = df[~df["brain"].isin(SWEEP_BRAINS)]

        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, "fig5_metrics.csv"), index=False)

    os.makedirs(save_dir, exist_ok=True)
    fig, axes_dict = plot_convergence_scatter(df, save_dir=save_dir, show=show, formats=("pdf", "svg"))
    return fig, axes_dict, df


if __name__ == "__main__":
    main()
