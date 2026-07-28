import os
import glob
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_closing, binary_fill_holes, label, binary_erosion, zoom
from skimage.morphology import disk

try:
    from torchmetrics.functional.image import (
        multiscale_structural_similarity_index_measure as ms_ssim_fn,
    )
    HAS_MS_SSIM = True
except ImportError:
    HAS_MS_SSIM = False
    print("[warn] torchmetrics not found — falling back to single-scale SSIM for MS-SSIM")

# ═══════════════════════════════════════════════
# DEFAULTS, override with keyword arguments
# ═══════════════════════════════════════════════
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_FILE = os.path.join(REPO_ROOT, "fwi_brains.txt")
GT_DIR = "" # set before running

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
    "missing_low_freq": "Missing low freq.",
}
METHOD_LABELS = {
    "baseline": "FWI only",
    "dg":       "DG",
    "dg_c":     "DG-C",
    "sg":       "SG",
    "sg_c":     "SG-C",
}

DEFAULT_SEARCH_DIRS = [""]  # set before running
FINAL_ITER = 80

# ═══════════════════════════════
# CRITERION + MASK PARAMETERS
# ═══════════════════════════════
TISSUE_TOL = 30     # m/s, pixel-% criterion tissue tolerance
SKULL_TOL = 250     # m/s, pixel-% criterion skull tolerance
TISSUE_BUF_LO = 40.0  # %, below this = confident FAILED
TISSUE_BUF_HI = 60.0  # %, above this = confident CONVERGED
SKULL_BUF_LO = 30.0
SKULL_BUF_HI = 60.0

SKULL_THRESH_MS = 1650.0   # m/s, GT threshold for skull vs tissue
CLOSING_RADIUS = 5         # default skull closing radius (px)
TISSUE_EROSION = 10        # tissue mask erosion (px) — disk(10)
SKULL_EROSION = 3          # skull mask erosion (px) — disk(3)

MANUAL_CLOSING_RADIUS = {
    "vp_182032": 13,
    "vp_268850": 7,
}
MANUAL_PATCHES = {
    "vp_170934": [(slice(225, 240), slice(208, 212))],
}

# Physical velocity range for MS-SSIM data_range and PSNR
VP_MIN = 1480.0   # m/s
VP_MAX = 3000.0   # m/s
VP_RANGE = VP_MAX

# Brains used for hyperparameter sweep
SWEEP_BRAINS = {
    "vp_121416", "vp_177746", "vp_178950", "vp_268749", "vp_385046",
    "vp_395251", "vp_580347", "vp_581450", "vp_732243", "vp_922854",
}

MIN_CONV_FOR_APPENDIX = 5


# ══════════
# MASKING
# ══════════
def make_masks(gt_2d, name):
    """
    Returns (tissue_mask, skull_mask, brain_mask, valid).

    tissue_mask  — eroded soft tissue (disk(TISSUE_EROSION))
    skull_mask   — eroded skull ring (disk(SKULL_EROSION))
    brain_mask   — filled skull ring (uneroded); used for MS-SSIM bbox and PSNR
    valid        — False if either tissue or skull mask is empty
    """
    closing_r = MANUAL_CLOSING_RADIUS.get(name, CLOSING_RADIUS)
    patches = MANUAL_PATCHES.get(name, [])

    skull_raw = (gt_2d >= SKULL_THRESH_MS).copy()
    for rs, cs in patches:
        skull_raw[rs, cs] = True

    closed = binary_closing(skull_raw, structure=disk(closing_r))
    filled = binary_fill_holes(closed)  # = whole brain interior incl. skull

    tissue_raw = filled & ~skull_raw
    labeled, n = label(tissue_raw)
    if n == 0:
        return (np.zeros_like(skull_raw, bool),
                np.zeros_like(skull_raw, bool),
                filled.astype(bool), False)
    elif n > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        tissue_clean = (labeled == sizes.argmax()).astype(bool)
    else:
        tissue_clean = tissue_raw.astype(bool)

    tissue_mask = binary_erosion(tissue_clean, structure=disk(TISSUE_EROSION))
    skull_mask = binary_erosion(closed, structure=disk(SKULL_EROSION))
    brain_mask = filled.astype(bool)  # uneroded, for PSNR / MS-SSIM

    valid = bool(tissue_mask.any()) and bool(skull_mask.any())
    return tissue_mask.astype(bool), skull_mask.astype(bool), brain_mask, valid


# ═════════
# I/O
# ═════════
def load_gt(name, gt_dir):
    path = os.path.join(gt_dir, f"{name}.npy")
    return np.load(path).astype(np.float32) if os.path.isfile(path) else None


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


def align(recon, gt):
    if recon.shape == gt.shape:
        return recon
    zf = (gt.shape[0] / recon.shape[0], gt.shape[1] / recon.shape[1])
    return zoom(recon, zf, order=1)


# ═══════════════════
# METRIC FUNCTIONS
# ═══════════════════
def pct_within(gt_vals, recon_vals, tol):
    if len(gt_vals) == 0:
        return float("nan")
    return float((np.abs(recon_vals - gt_vals) <= tol).mean() * 100)


def rmse(gt_vals, recon_vals):
    if len(gt_vals) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((gt_vals - recon_vals) ** 2)))


def compute_psnr(gt_2d, recon_2d, brain_mask, data_range=VP_RANGE):
    """
    PSNR over the brain mask (tissue | skull, uneroded), data_range=VP_MAX.
    Background (water) is excluded via brain_mask.
    """
    gt_v = gt_2d[brain_mask].astype(np.float64)
    rec_v = recon_2d[brain_mask].astype(np.float64)
    mse_val = np.mean((gt_v - rec_v) ** 2)
    if mse_val == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range / np.sqrt(mse_val)))


def compute_ms_ssim(gt_2d, recon_2d, brain_mask):
    """MS-SSIM over the brain bounding box, background zeroed out."""
    rows, cols = np.where(brain_mask)
    if len(rows) == 0:
        return float("nan")
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1

    gt_crop = gt_2d[r0:r1, c0:c1].copy()
    rec_crop = recon_2d[r0:r1, c0:c1].copy()
    mask_crop = brain_mask[r0:r1, c0:c1]
    gt_crop[~mask_crop] = 0.0
    rec_crop[~mask_crop] = 0.0

    if HAS_MS_SSIM:
        gt_t = torch.from_numpy(gt_crop).unsqueeze(0).unsqueeze(0).float()
        rec_t = torch.from_numpy(rec_crop).unsqueeze(0).unsqueeze(0).float()
        try:
            return float(ms_ssim_fn(rec_t, gt_t, data_range=VP_RANGE,
                                     kernel_size=7,
                                     betas=(0.0448, 0.2856, 0.3001)))
        except Exception:
            pass
    from skimage.metrics import structural_similarity
    return float(structural_similarity(gt_crop, rec_crop, data_range=VP_RANGE))


def classify_pixel_pct(tissue_pct, skull_pct):
    """Provisional classification from pixel-% values alone: converged/failed/buffer."""
    if tissue_pct >= TISSUE_BUF_HI and skull_pct >= SKULL_BUF_HI:
        return "converged"
    if tissue_pct <= TISSUE_BUF_LO or skull_pct <= SKULL_BUF_LO:
        return "failed"
    return "buffer"


def fmt(mean, std, decimals=1):
    """Format mean±std, or '—' if NaN."""
    if np.isnan(mean):
        return "—"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def compute_stats(group, col):
    """Return (mean, std) over a DataFrame group, or (nan, nan) if empty."""
    vals = group[col].dropna()
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


# ═════════════════════
# MAIN ENTRY POINT
# ═════════════════════
def compute(search_dirs=DEFAULT_SEARCH_DIRS, output_dir=None, scenarios=SCENARIOS,
            methods=METHODS, final_iter=FINAL_ITER, models_file=MODELS_FILE,
            gt_dir=GT_DIR, classification_csv=None, reference_scenario="standard",
            verbose=True):
    """
    Run  full evaluation and write results_full.csv / summary_main.csv /
    summary_main.txt / summary_appendix_common.csv / .txt into output_dir.

    search_dirs : list of run-output roots to search, in order (first match
                  wins per brain/scenario/method).
    classification_csv : optional path to a hand-labeled classification CSV
                  (brain,scenario,method,status) resolving buffer-zone cases.
    reference_scenario : which scenario the "common converged subset" (Table B2)
                  is computed over.

    Returns the per-brain results DataFrame.
    """
    if output_dir is None:
        raise ValueError("output_dir is required")
    os.makedirs(output_dir, exist_ok=True)

    final_status_map = {}  # (brain, scenario, method) -> "converged" | "failed"
    if classification_csv is not None and os.path.isfile(classification_csv):
        clf = pd.read_csv(classification_csv)
        for _, row in clf.iterrows():
            final_status_map[(row["brain"], row["scenario"], row["method"])] = row["status"]
        if verbose:
            print(f"Loaded {len(final_status_map)} final classifications from {classification_csv}")
    elif verbose:
        print("No classification_csv — using pixel-% criterion directly "
              "(buffer zone -> converged=0 provisionally).")

    with open(models_file) as f:
        brains = [l.strip() for l in f if l.strip()]
    brains = [b for b in brains if b not in SWEEP_BRAINS]

    if verbose:
        print(f"\nBrains: {len(brains)}  |  Scenarios: {len(scenarios)}  |  Methods: {len(methods)}")
        print(f"Tissue mask: disk({TISSUE_EROSION})  |  Skull mask: disk({SKULL_EROSION})  |  "
              f"Brain mask: filled (for PSNR + MS-SSIM)\n")
        print(f"Search dirs: {search_dirs}")

    rows_out = []
    skipped_gt, skipped_mask, skipped_norun = [], [], []

    for brain in brains:
        gt = load_gt(brain, gt_dir)
        if gt is None:
            skipped_gt.append(brain)
            continue

        tissue_mask, skull_mask, brain_mask, valid = make_masks(gt, brain)
        if not valid:
            skipped_mask.append(brain)
            continue

        for scenario in scenarios:
            for method in methods:
                recon = load_fwi(brain, scenario, method, search_dirs, final_iter)
                if recon is None:
                    skipped_norun.append((brain, scenario, method))
                    continue

                recon = align(recon, gt)

                t_pct = pct_within(gt[tissue_mask], recon[tissue_mask], TISSUE_TOL)
                s_pct = pct_within(gt[skull_mask], recon[skull_mask], SKULL_TOL)

                k = (brain, scenario, method)
                status = final_status_map.get(k) or classify_pixel_pct(t_pct, s_pct)
                converged = 1 if status == "converged" else 0

                t_rmse = rmse(gt[tissue_mask], recon[tissue_mask])
                t_std = float(recon[tissue_mask].std()) if tissue_mask.any() else float("nan")
                t_mean = float(recon[tissue_mask].mean()) if tissue_mask.any() else float("nan")
                s_rmse = rmse(gt[skull_mask], recon[skull_mask])

                ms_ssim_val = compute_ms_ssim(gt, recon, brain_mask)
                psnr_val = compute_psnr(gt, recon, brain_mask)

                rows_out.append({
                    "brain": brain, "scenario": scenario, "method": method,
                    "tissue_pct": round(t_pct, 2), "skull_pct": round(s_pct, 2),
                    "status": status, "converged": converged,
                    "tissue_rmse": round(t_rmse, 3), "tissue_std": round(t_std, 3),
                    "tissue_mean": round(t_mean, 3), "skull_rmse": round(s_rmse, 3),
                    "ms_ssim": round(ms_ssim_val, 4), "psnr": round(psnr_val, 3),
                    "n_tissue_px": int(tissue_mask.sum()), "n_skull_px": int(skull_mask.sum()),
                    "n_brain_px": int(brain_mask.sum()),
                })

    if verbose:
        print(f"\nComputed {len(rows_out)} rows")
        if skipped_gt:
            print(f"GT missing:    {skipped_gt}")
        if skipped_mask:
            print(f"Mask failures: {skipped_mask}")
        if skipped_norun:
            print(f"Missing runs:  {len(skipped_norun)}")

    df = pd.DataFrame(rows_out)
    results_path = os.path.join(output_dir, "results_full.csv")
    df.to_csv(results_path, index=False)
    if verbose:
        print(f"Saved: {results_path}")

    # ── MAIN TABLE: all brains ─────────────────────────────────────
    main_lines = [
        "=" * 110,
        "  MAIN TABLE — mean±std over ALL brains (no selection bias)",
        f"  Tissue RMSE: disk({TISSUE_EROSION}) eroded tissue mask  |  "
        f"MS-SSIM / PSNR: filled brain mask, data_range={VP_RANGE:.0f}m/s",
        "=" * 110,
        f"  {'Scenario':<20} {'Method':<22} {'n':>4} {'Conv%':>7}"
        f" {'RMSE':>12} {'MS-SSIM':>14} {'PSNR':>13}",
        "-" * 110,
    ]
    main_rows = []
    for scenario in scenarios:
        for method in methods:
            sub = df[(df.scenario == scenario) & (df.method == method)] if len(df) else df
            if len(sub) == 0:
                continue
            n = len(sub)
            n_c = int(sub["converged"].sum())
            conv_rate = n_c / n * 100
            rmse_m, rmse_s = compute_stats(sub, "tissue_rmse")
            ssim_m, ssim_s = compute_stats(sub, "ms_ssim")
            psnr_m, psnr_s = compute_stats(sub, "psnr")
            main_lines.append(
                f"  {SCENARIO_LABELS.get(scenario, scenario):<20} "
                f"{METHOD_LABELS.get(method, method):<22} {n:>4} {conv_rate:>6.1f}%"
                f" {fmt(rmse_m, rmse_s):>12} {fmt(ssim_m, ssim_s, decimals=4):>14}"
                f" {fmt(psnr_m, psnr_s):>13}"
            )
            main_rows.append({
                "scenario": scenario, "method": method, "n_total": n, "n_converged": n_c,
                "convergence_rate_pct": round(conv_rate, 2),
                "tissue_rmse_mean_all": round(rmse_m, 3), "tissue_rmse_std_all": round(rmse_s, 3),
                "ms_ssim_mean_all": round(ssim_m, 4), "ms_ssim_std_all": round(ssim_s, 4),
                "psnr_mean_all": round(psnr_m, 3), "psnr_std_all": round(psnr_s, 3),
            })
    main_lines += [
        "-" * 110,
        "  All quality metrics averaged over all N brains regardless of convergence status.",
        "  RMSE in m/s  |  PSNR in dB  |  MS-SSIM dimensionless [0-1]",
    ]
    main_csv = os.path.join(output_dir, "summary_main.csv")
    main_txt = os.path.join(output_dir, "summary_main.txt")
    pd.DataFrame(main_rows).to_csv(main_csv, index=False)
    with open(main_txt, "w") as fh:
        fh.write("\n".join(main_lines))
    if verbose:
        for line in main_lines:
            print(line)
        print(f"\nSaved: {main_csv}\nSaved: {main_txt}")

    # ── APPENDIX: common subset converged in ALL methods ─────────
    common_rows = []
    common_lines = []
    if reference_scenario in scenarios and len(df):
        ref = df[df["scenario"] == reference_scenario]
        per_brain = ref.groupby("brain")["converged"].min()
        common_brains = set(per_brain[per_brain == 1].index)
        if verbose:
            print(f"\nCommon subset ({reference_scenario}, all methods converge): "
                  f"{len(common_brains)} brains")

        common_lines = [
            "=" * 110,
            f"  APPENDIX — common subset: {len(common_brains)} brains converging in ALL "
            f"methods ({reference_scenario})",
            "=" * 110,
            f"  {'Method':<22} {'n':>4} {'RMSE':>14} {'MS-SSIM':>14} {'PSNR':>13}",
            "-" * 110,
        ]
        for method in methods:
            sub = ref[(ref["brain"].isin(common_brains)) & (ref["method"] == method)]
            rmse_m, rmse_s = compute_stats(sub, "tissue_rmse")
            ssim_m, ssim_s = compute_stats(sub, "ms_ssim")
            psnr_m, psnr_s = compute_stats(sub, "psnr")
            common_lines.append(
                f"  {METHOD_LABELS.get(method, method):<22} {len(sub):>4}"
                f" {fmt(rmse_m, rmse_s):>14} {fmt(ssim_m, ssim_s, decimals=4):>14}"
                f" {fmt(psnr_m, psnr_s):>13}"
            )
            if len(sub) >= MIN_CONV_FOR_APPENDIX:
                common_rows.append({
                    "method": method, "n_common": len(sub),
                    "tissue_rmse_mean": round(rmse_m, 3), "tissue_rmse_std": round(rmse_s, 3),
                    "ms_ssim_mean": round(ssim_m, 4), "ms_ssim_std": round(ssim_s, 4),
                    "psnr_mean": round(psnr_m, 3), "psnr_std": round(psnr_s, 3),
                })

    app_txt = os.path.join(output_dir, "summary_appendix_common.txt")
    app_csv = os.path.join(output_dir, "summary_appendix_common.csv")
    with open(app_txt, "w") as fh:
        fh.write("\n".join(common_lines))
    pd.DataFrame(common_rows).to_csv(app_csv, index=False)
    if verbose:
        for line in common_lines:
            print(line)
        print(f"Saved: {app_txt}\nSaved: {app_csv}")

    return df


if __name__ == "__main__":
    compute(
        search_dirs=DEFAULT_SEARCH_DIRS,
        output_dir=os.path.join(REPO_ROOT, "paper-results", "results", "reference"),
        scenarios=SCENARIOS,
        methods=METHODS,
        final_iter=FINAL_ITER,
    )
