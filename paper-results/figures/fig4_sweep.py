import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import apply_paper_style, masked_brain_imshow, guasch_mask, add_panel_label, VMIN, VMAX

import glob
import re
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_closing, binary_fill_holes,
                            label as scipy_label, binary_erosion, zoom)
from skimage.morphology import disk

# ═════════════════════════════════
# CONFIGURATION & RESOURCE PATHS
# ═════════════════════════════════
PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SWEEP_ROOT = ""  # set before running
GT_DIR = ""  # set before running
OUT_DIR = os.path.join(PAPER_DIR, "results", "figures")

SWEEP_BRAINS = [
    "vp_121416", "vp_177746", "vp_178950", "vp_268749",
    "vp_385046", "vp_395251", "vp_580347", "vp_581450",
    "vp_732243", "vp_922854",
]

TARGET_VIS_BRAIN = "vp_268749"

SKULL_THRESH_MS = 1650.0
CLOSING_RADIUS = 5
TISSUE_EROSION = 10
SKULL_EROSION = 3
MANUAL_CLOSING_RADIUS = {"vp_182032": 13, "vp_268850": 7}
MANUAL_PATCHES = {"vp_170934": [(slice(225, 240), slice(208, 212))]}

INNER_WSPACE = 0.04
INNER_HSPACE = 0.10
BUFFER_ROW_RATIO = 0.45

ROW_LABEL_GAP = 0.024
TSTART_TITLE_GAP = 0.050
COL_LABEL_GAP = 0.022
ALPHA_TITLE_GAP = 0.050

IMG_PAD_FRAC = 0.10
TICK_FONTSIZE = 7.5
AXIS_TITLE_FONTSIZE = 10
AXIS_LABELPAD = 3
BC_WIDTH_SCALE = 0.5

COLORBAR_WIDTH = 0.012
COLORBAR_PAD = 0.012


# ══════════════════════════════
# DIRECTORY & FILE HELPERS
# ══════════════════════════════
def find_sweep_dir(brain, default_suffix, keywords):
    base_dir = os.path.join(SWEEP_ROOT, f"{brain}_{default_suffix}")
    if os.path.isdir(base_dir):
        return base_dir
    if os.path.isdir(SWEEP_ROOT):
        for entry in os.listdir(SWEEP_ROOT):
            if brain in entry and any(k in entry for k in keywords):
                candidate = os.path.join(SWEEP_ROOT, entry)
                if os.path.isdir(candidate):
                    return candidate
    return None


def _get_iteration_56(paths):
    if not paths:
        return None
    for p in paths:
        m = re.search(r"-Vp-(\d+)", os.path.basename(p))
        if m and int(m.group(1)) == 56:
            return p
    best_it, best_p = -1, None
    for p in paths:
        m = re.search(r"-Vp-(\d+)", os.path.basename(p))
        if m:
            it = int(m.group(1))
            if it > best_it:
                best_it, best_p = it, p
    return best_p


def discover_diff_sweep(brain):
    sweep_dir = find_sweep_dir(brain, "diff_sweep", ["diff", "pp_merged", "merged"])
    if not sweep_dir:
        return {}
    grouped = {}
    for fpath in glob.glob(os.path.join(sweep_dir, "*.h5")):
        m = re.search(r"_t(\d+)_a(\d+)", os.path.basename(fpath))
        if m:
            grouped.setdefault((int(m.group(1)), round(int(m.group(2)) / 100.0, 1)), []).append(fpath)
    return {k: _get_iteration_56(v) for k, v in grouped.items() if _get_iteration_56(v)}


def discover_diff_split(brain):
    sweep_dir = find_sweep_dir(brain, "split_sweep", ["split", "pp_split"])
    if not sweep_dir:
        return {}
    grouped = {}
    for fpath in glob.glob(os.path.join(sweep_dir, "*.h5")):
        m = re.search(r"_as(\d+)_at(\d+)", os.path.basename(fpath))
        if m:
            grouped.setdefault((round(int(m.group(1)) / 100.0, 1), round(int(m.group(2)) / 100.0, 1)), []).append(fpath)
    return {k: _get_iteration_56(v) for k, v in grouped.items() if _get_iteration_56(v)}


def discover_grad_split(brain):
    sweep_dir = find_sweep_dir(brain, "grad_sweep_split", ["grad_split", "split"])
    if not sweep_dir:
        return {}
    grouped = {}
    for fpath in glob.glob(os.path.join(sweep_dir, "*.h5")):
        m = re.search(r"_ls(\d+)_lt(\d+)", os.path.basename(fpath))
        if m:
            grouped.setdefault((round(int(m.group(1)) / 1000.0, 3), round(int(m.group(2)) / 1000.0, 3)), []).append(fpath)
    return {k: _get_iteration_56(v) for k, v in grouped.items() if _get_iteration_56(v)}


# ════════════════════════════════════════════════
# DATA PROCESSING & MATRIX GENERATION METRICS
# ════════════════════════════════════════════════
def make_tissue_skull_masks(gt_2d, name):
    closing_r = MANUAL_CLOSING_RADIUS.get(name, CLOSING_RADIUS)
    patches = MANUAL_PATCHES.get(name, [])
    skull_raw = (gt_2d >= SKULL_THRESH_MS).copy()
    for rs, cs in patches:
        skull_raw[rs, cs] = True
    closed = binary_closing(skull_raw, structure=disk(closing_r))
    filled = binary_fill_holes(closed)
    tissue_raw = filled & ~skull_raw
    labeled, n = scipy_label(tissue_raw)
    if n == 0:
        return None, None, False
    if n > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        tissue_clean = (labeled == sizes.argmax())
    else:
        tissue_clean = tissue_raw.astype(bool)
    tissue_mask = binary_erosion(tissue_clean, structure=disk(TISSUE_EROSION))
    skull_mask = binary_erosion(closed, structure=disk(SKULL_EROSION)) if SKULL_EROSION > 0 else closed.astype(bool)
    return tissue_mask.astype(bool), skull_mask.astype(bool), bool(tissue_mask.any() and skull_mask.any())


def load_h5(path):
    with h5py.File(path, "r") as f:
        return f["data"][()].astype(np.float32)


def load_gt(brain):
    p = os.path.join(GT_DIR, f"{brain}.npy")
    return np.load(p).astype(np.float32) if os.path.isfile(p) else None


def align(recon, gt):
    if recon.shape == gt.shape:
        return recon
    return zoom(recon, (gt.shape[0] / recon.shape[0], gt.shape[1] / recon.shape[1]), order=1)


def sweep_metrics_df(brain_list, discover_fn, prefix):
    mask_cache, gt_cache = {}, {}
    for b in brain_list:
        gt = load_gt(b)
        if gt is not None:
            gt_cache[b] = gt
            tm, sm, valid = make_tissue_skull_masks(gt, b)
            if valid:
                mask_cache[b] = (tm, sm)

    brain_configs, all_keys = {}, set()
    for b in brain_list:
        cfgs = discover_fn(b)
        brain_configs[b] = cfgs
        all_keys |= set(cfgs.keys())

    rows = []
    for cfg_key in sorted(all_keys):
        t_rmses = []
        for b in brain_list:
            if b not in mask_cache or cfg_key not in brain_configs[b]:
                continue
            try:
                pred = load_h5(brain_configs[b][cfg_key])
                img_aligned = align(pred, gt_cache[b])
                tm_mask = mask_cache[b][0]
                t_rmse = float(np.sqrt(np.mean((img_aligned[tm_mask] - gt_cache[b][tm_mask]) ** 2)))
                if not np.isnan(t_rmse):
                    t_rmses.append(t_rmse)
            except Exception:
                continue
        if t_rmses:
            rows.append({"config_key": cfg_key, "mean_tissue_rmse": float(np.mean(t_rmses))})
    return pd.DataFrame(rows)


def df_to_matrix(df, row_fn, col_fn, allowed_rows, allowed_cols):
    if df.empty:
        return list(allowed_rows), list(allowed_cols), np.full((len(allowed_rows), len(allowed_cols)), np.nan)

    present_rows = sorted(list({round(row_fn(k), 1) if isinstance(row_fn(k), float) else row_fn(k) for k in df["config_key"]}))
    present_cols = sorted(list({round(col_fn(k), 1) if isinstance(col_fn(k), float) else col_fn(k) for k in df["config_key"]}))

    row_vals = allowed_rows if any(r in present_rows for r in allowed_rows) else present_rows
    col_vals = allowed_cols if any(c in present_cols for c in allowed_cols) else present_cols

    rmse_mat = np.full((len(row_vals), len(col_vals)), np.nan)
    for _, row in df.iterrows():
        r_val = round(row_fn(row["config_key"]), 1) if isinstance(row_fn(row["config_key"]), float) else row_fn(row["config_key"])
        c_val = round(col_fn(row["config_key"]), 1) if isinstance(col_fn(row["config_key"]), float) else col_fn(row["config_key"])
        if r_val in row_vals and c_val in col_vals:
            i = row_vals.index(r_val)
            j = col_vals.index(c_val)
            rmse_mat[i, j] = row["mean_tissue_rmse"]
    return row_vals, col_vals, rmse_mat


# ════════════════════════════════════════
# PLOTTING PANEL RENDERING PIPELINE
# ════════════════════════════════════════s
def render_diff_fwi_refine_grid(fig, sub_gs, brain_name, discover_fn, row_vals, col_vals):
    """Renders Panel A: multi-row structural grid."""
    cfgs = discover_fn(brain_name)
    gt = load_gt(brain_name)
    shared_mask = guasch_mask(gt) if gt is not None else None

    nrows, ncols = len(row_vals), len(col_vals)
    inner_grid = sub_gs.subgridspec(nrows, 7, wspace=INNER_WSPACE, hspace=INNER_HSPACE,
                                     width_ratios=[1.0, 0.55, 1.0, 1.0, 1.0, 1.0, 1.0])

    ax_gt = fig.add_subplot(inner_grid[0, 0])
    if gt is not None:
        masked_brain_imshow(ax_gt, gt, mask=shared_mask, vmin=VMIN, vmax=VMAX, aspect="equal", origin="lower")
        ax_gt.axis("off")
        ax_gt.text(0.5, -0.16, "GT", transform=ax_gt.transAxes, ha="center", va="top",
                    fontsize=10, fontweight="bold", color="#334155")

    last_im = None
    first_col_axes = {}
    last_row_axes = {}
    for i, r_val in enumerate(row_vals):
        for j, c_val in enumerate(col_vals):
            ax = fig.add_subplot(inner_grid[i, j + 2])
            ax.axis("off")
            if j == 0:
                first_col_axes[i] = ax
            if i == nrows - 1:
                last_row_axes[j] = ax

            cfg_tuple = (r_val, c_val)
            if cfg_tuple in cfgs:
                try:
                    img = align(load_h5(cfgs[cfg_tuple]), gt)
                    last_im = masked_brain_imshow(ax, img, mask=shared_mask, vmin=VMIN, vmax=VMAX,
                                                   aspect="equal", origin="lower")
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
                    xpad = IMG_PAD_FRAC * (xlim[1] - xlim[0])
                    ypad = IMG_PAD_FRAC * (ylim[1] - ylim[0])
                    ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
                    ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)
                except Exception:
                    pass

    ax_axes_lines = fig.add_subplot(inner_grid[0:nrows, 2:7])
    ax_axes_lines.set_facecolor("none")
    for spine_name, spine in ax_axes_lines.spines.items():
        if spine_name in ["left", "bottom"]:
            spine.set_visible(True)
            spine.set_color("#1e293b")
            spine.set_linewidth(0.9)
        else:
            spine.set_visible(False)

    ax_axes_lines.set_xlim(-0.5, ncols - 0.5)
    ax_axes_lines.set_xticks(range(ncols))
    ax_axes_lines.set_xticklabels([f"{v:.1f}" for v in col_vals], fontsize=TICK_FONTSIZE, color="#334155")
    ax_axes_lines.tick_params(axis="x", length=3, color="#1e293b")
    ax_axes_lines.set_xlabel(r"$\alpha_{\rm start}$", fontsize=AXIS_TITLE_FONTSIZE, fontweight="bold",
                              labelpad=AXIS_LABELPAD, color="#1e293b")

    ax_axes_lines.set_ylim(-0.5, nrows - 0.5)
    ax_axes_lines.invert_yaxis()
    ax_axes_lines.set_yticks(range(nrows))
    ax_axes_lines.set_yticklabels([str(int(v)) for v in row_vals], fontsize=TICK_FONTSIZE, color="#334155")
    ax_axes_lines.tick_params(axis="y", length=3, color="#1e293b")
    ax_axes_lines.set_ylabel(r"$t_{\rm start}$", fontsize=AXIS_TITLE_FONTSIZE, fontweight="bold",
                              labelpad=AXIS_LABELPAD, color="#1e293b")

    if last_im is not None:
        add_matched_colorbar(fig, last_im, ax_axes_lines, "Velocity (m/s)")


def render_annotated_heatmap(fig, ax, row_vals, col_vals, rmse_mat, xlabel, ylabel,
                              cbar_label="Soft Tissue RMSE (m/s)"):
    """Renders clean evaluation heatmaps, colorbar matched to other panels."""
    vmin = float(np.nanmin(rmse_mat)) if not np.isnan(rmse_mat).all() else 40.0
    vmax = float(np.nanmax(rmse_mat)) if not np.isnan(rmse_mat).all() else 130.0
    if vmin == vmax:
        vmin, vmax = max(0.0, vmin - 10.0), vmax + 10.0

    im = ax.imshow(rmse_mat, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")

    nrows, ncols = rmse_mat.shape
    for x in range(ncols + 1):
        ax.axvline(x - 0.5, color="#e2e8f0", linewidth=0.5)
    for y in range(nrows + 1):
        ax.axhline(y - 0.5, color="#e2e8f0", linewidth=0.5)

    for i in range(nrows):
        for j in range(ncols):
            val = rmse_mat[i, j]
            if not np.isnan(val):
                norm_val = (val - vmin) / (vmax - vmin) if (vmax - vmin) != 0 else 0.5
                text_color = "black" if (0.32 < norm_val < 0.68) else "white"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7.0, color=text_color)

    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels([f"{v:.1f}" for v in col_vals], fontsize=TICK_FONTSIZE)
    ax.set_yticklabels([f"{v:.1f}" for v in row_vals], fontsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_TITLE_FONTSIZE, fontweight="bold", labelpad=AXIS_LABELPAD, color="#1e293b")
    ax.set_ylabel(ylabel, fontsize=AXIS_TITLE_FONTSIZE, fontweight="bold", labelpad=AXIS_LABELPAD, color="#1e293b")
    ax.tick_params(axis="x", length=3)
    ax.tick_params(axis="y", length=3)

    add_matched_colorbar(fig, im, ax, cbar_label)


def add_matched_colorbar(fig, mappable, ref_ax, label, fontsize=8.0, ticklabelsize=6.5):
    pos = ref_ax.get_position()
    cax = fig.add_axes([pos.x1 + COLORBAR_PAD, pos.y0, COLORBAR_WIDTH, pos.height])
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=ticklabelsize, length=2)
    cb.set_label(label, fontsize=fontsize, labelpad=4, color="#1e293b")
    return cb


# ════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════
def main(out_dir=OUT_DIR, precomputed_matrices_json=None, precomputed_sweep_root=None,
         precomputed_gt_dir=None):
         
    apply_paper_style()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if precomputed_sweep_root:
        global SWEEP_ROOT
        SWEEP_ROOT = precomputed_sweep_root
    if precomputed_gt_dir:
        global GT_DIR
        GT_DIR = precomputed_gt_dir

    ALLOWED_ROWS_A = [300, 400, 500, 600, 700]
    ALLOWED_COLS_A = [0.5, 0.6, 0.7, 0.8, 0.9]

    ALLOWED_ALPHA_ST = [0.1, 0.3, 0.5, 0.7, 0.9]
    ALLOWED_ALPHA_SK = [0.3, 0.5, 0.7, 0.9]

    ALLOWED_LAMBDA_SK = [1.0, 1.3, 1.6, 2.0]
    ALLOWED_LAMBDA_ST = [0.3, 0.8, 1.3, 1.6]

    sweep_profiles = {
        "A": {"fn": discover_diff_sweep, "r_fn": lambda k: k[0], "c_fn": lambda k: k[1],
              "r_all": ALLOWED_ROWS_A, "c_all": ALLOWED_COLS_A, "prefix": "pp_merged"},
        "B": {"fn": discover_diff_split, "r_fn": lambda k: k[0], "c_fn": lambda k: k[1],
              "r_all": ALLOWED_ALPHA_SK, "c_all": ALLOWED_ALPHA_ST, "prefix": "pp_split"},
        "C": {"fn": discover_grad_split, "r_fn": lambda k: k[0], "c_fn": lambda k: k[1],
              "r_all": ALLOWED_LAMBDA_SK, "c_all": ALLOWED_LAMBDA_ST, "prefix": "grad_split"},
    }

    if precomputed_matrices_json:
        import json
        with open(precomputed_matrices_json) as f:
            precomputed = json.load(f)
        for name, spec in sweep_profiles.items():
            entry = precomputed[name]
            rmse_mat = np.array(entry["rmse_mat"]) if entry["rmse_mat"] is not None else None
            spec["mats"] = (entry["row_vals"], entry["col_vals"], rmse_mat)
    else:
        for name, spec in sweep_profiles.items():
            print(f"Aggregating dataset parameters for Sweep {name}...")
            df = sweep_metrics_df(SWEEP_BRAINS, spec["fn"], spec["prefix"])
            spec["mats"] = df_to_matrix(df, spec["r_fn"], spec["c_fn"], spec["r_all"], spec["c_all"]) if not df.empty else None

    fig = plt.figure(figsize=(11.84, 4.2))
    master_gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.18], hspace=0.0, wspace=0.35,
                                  left=0.05, right=0.95, top=0.92, bottom=0.14)

    # ── COLUMN 0: DiffFWI-Refine sweep reconstructions ───────────────────────
    gs_left = master_gs[0, 0]
    ax_a_anchor = fig.add_subplot(gs_left)
    ax_a_anchor.axis("off")
    add_panel_label(ax_a_anchor, "A", x=-0.04, y=1.02, fontsize=14)
    ax_a_anchor.set_title("DiffFWI-Refine Sweep", fontsize=10, fontweight="bold", pad=8, loc="left", color="#0f172a")

    spec_a = sweep_profiles["A"]
    if spec_a["mats"] is not None:
        render_diff_fwi_refine_grid(fig, gs_left, TARGET_VIS_BRAIN, spec_a["fn"], spec_a["mats"][0], spec_a["mats"][1])
    else:
        ax_a_anchor.text(0.5, 0.5, "Data Vector Error\nSweep DF Empty", ha="center", va="center", color="red")

    # ── COLUMN 1: DiffFWI-Refine-C / DiffFWI-Score-C heatmaps ────────────────
    pos_right = master_gs[0, 1].get_position(fig)
    new_width = pos_right.width * BC_WIDTH_SCALE
    gs_right = fig.add_gridspec(2, 1, left=pos_right.x0, right=pos_right.x0 + new_width,
                                 top=pos_right.y1, bottom=pos_right.y0, hspace=0.55)

    ax_b = fig.add_subplot(gs_right[0, 0])
    add_panel_label(ax_b, "B", x=-0.14, y=1.04, fontsize=14)
    ax_b.set_title("DiffFWI-Refine-C Sweep", fontsize=10, fontweight="bold", pad=5, loc="left", color="#0f172a")

    spec_b = sweep_profiles["B"]
    if spec_b["mats"] is not None:
        render_annotated_heatmap(fig, ax_b, spec_b["mats"][0], spec_b["mats"][1], spec_b["mats"][2],
                                  xlabel=r"$\alpha_{\rm ST}$", ylabel=r"$\alpha_{\rm SK}$")
    else:
        ax_b.text(0.5, 0.5, "Matrix Array Missing", ha="center", va="center", color="gray")

    ax_c = fig.add_subplot(gs_right[1, 0])
    add_panel_label(ax_c, "C", x=-0.14, y=1.04, fontsize=14)
    ax_c.set_title("DiffFWI-Score-C Sweep", fontsize=10, fontweight="bold", pad=5, loc="left", color="#0f172a")

    spec_c = sweep_profiles["C"]
    if spec_c["mats"] is not None:
        render_annotated_heatmap(fig, ax_c, spec_c["mats"][0], spec_c["mats"][1], spec_c["mats"][2],
                                  xlabel=r"$\lambda_{\rm ST}$", ylabel=r"$\lambda_{\rm SK}$")
    else:
        ax_c.text(0.5, 0.5, "Matrix Array Missing", ha="center", va="center", color="gray")

    output_path = os.path.join(out_dir, "fig4_hyperparameter_sweep.pdf")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    return fig


if __name__ == "__main__":
    main()
