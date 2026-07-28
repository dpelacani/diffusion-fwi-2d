import os
import glob
import json
import pandas as pd
import numpy as np

METHOD_SHORT_LABELS = {
    "baseline": "FWI",
    "dg":       "DG",
    "dg_c":     "DG-C",
    "sg":       "SG",
    "sg_c":     "SG-C",
}
METHODS_ORDER = ["baseline", "dg", "dg_c", "sg", "sg_c"]

SCENARIO_SHORT_LABELS = {
    "standard":         "Standard",
    "reduced_compute":  "Reduced",
    "missing_low_freq": "No LF",
}
SCENARIOS_ORDER = ["standard", "reduced_compute", "missing_low_freq"]


def _fmt_pm(mean, std, prec, best_rank=None):
    """Format 'mean \\pm std' with optional bold (rank 1) / underline (rank 2)."""
    if pd.isna(mean):
        return "--"
    m_str = f"{mean:.{prec}f}"
    if pd.isna(std):
        if best_rank == 1:
            return f"$\\mathbf{{{m_str}}}$"
        if best_rank == 2:
            return f"$\\underline{{{m_str}}}$"
        return f"${m_str}$"
    s_str = f"{std:.{prec}f}"
    if best_rank == 1:
        return f"$\\mathbf{{{m_str}}} \\pm \\mathbf{{{s_str}}}$"
    if best_rank == 2:
        return f"$\\underline{{{m_str}}} \\pm \\underline{{{s_str}}}$"
    return f"${m_str} \\pm {s_str}$"


def _fmt_pct(value, best_rank=None):
    if pd.isna(value):
        return "--"
    v_str = f"{value:.1f}\\%"
    if best_rank == 1:
        return f"$\\mathbf{{{v_str}}}$"
    if best_rank == 2:
        return f"$\\underline{{{v_str}}}$"
    return f"${v_str}$"


def _rank_within(values, ascending):
    """
    Given a dict {method: value}, return {method: rank} where rank 1 = best,
    2 = second-best (ties broken by pandas default 'first'-seen order),
    ranks computed only over non-NaN values.
    """
    s = pd.Series(values).dropna().sort_values(ascending=ascending)
    ranks = {}
    for i, method in enumerate(s.index):
        if i == 0:
            ranks[method] = 1
        elif i == 1:
            ranks[method] = 2
    return ranks


def write_fwi_latex_table(summary_csv, output_tex, caption, label="tab:fwi_consolidated_metrics"):
    """
    FWI reconstruction performance across acquisition
    settings, one `\\multirow` block per scenario, methods in the fixed
    order FWI/DG/DG-C/SG/SG-C, bold/underline best/second-best computed
    independently within each scenario block per metric.

    summary_csv : path to a summary_main.csv written by
                  compute_fwi_results.compute() (columns: scenario, method,
                  convergence_rate_pct, tissue_rmse_mean_all/_std_all,
                  ms_ssim_mean_all/_std_all, psnr_mean_all/_std_all).
    caption : full caption text (including any \\label-adjacent citations),
              passed through verbatim.
    """
    df = pd.read_csv(summary_csv)

    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Setting & Method & Conv. rate\\,$\\uparrow$ & ST-RMSE (m/s)\\,$\\downarrow$ & "
        "MS-SSIM\\,$\\uparrow$ & PSNR (dB)\\,$\\uparrow$ \\\\",
        "\\midrule",
    ]

    for i, scenario in enumerate(SCENARIOS_ORDER):
        sc_df = df[df["scenario"] == scenario].set_index("method")
        if sc_df.empty:
            continue
        if i > 0:
            lines.append("\\midrule")

        conv_rank = _rank_within(sc_df["convergence_rate_pct"].to_dict(), ascending=False)
        rmse_rank = _rank_within(sc_df["tissue_rmse_mean_all"].to_dict(), ascending=True)
        ssim_rank = _rank_within(sc_df["ms_ssim_mean_all"].to_dict(), ascending=False)
        psnr_rank = _rank_within(sc_df["psnr_mean_all"].to_dict(), ascending=False)

        present_methods = [m for m in METHODS_ORDER if m in sc_df.index]
        lines.append(f"\\multirow{{{len(present_methods)}}}{{*}}{{\\textit{{{SCENARIO_SHORT_LABELS[scenario]}}}}}")
        for method in present_methods:
            row = sc_df.loc[method]
            cells = [
                "",
                METHOD_SHORT_LABELS[method],
                _fmt_pct(row["convergence_rate_pct"], conv_rank.get(method)),
                _fmt_pm(row["tissue_rmse_mean_all"], row["tissue_rmse_std_all"], 1, rmse_rank.get(method)),
                _fmt_pm(row["ms_ssim_mean_all"], row["ms_ssim_std_all"], 4, ssim_rank.get(method)),
                _fmt_pm(row["psnr_mean_all"], row["psnr_std_all"], 2, psnr_rank.get(method)),
            ]
            lines.append(" & ".join(cells).lstrip() + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]

    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    with open(output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_tex}")


def write_common_converged_latex_table(summary_csv, output_tex, caption,
                                        label="tab:fwi_common_converged_metrics"):
    """
    Table with single scenario, brains converging under every method),
    method rows in the fixed order FWI/DG/DG-C/SG/SG-C, bold/underline
    best/second-best per metric computed on rounded-to-displayed-precision
    values (so near-ties at higher precision don't produce a bold/underline
    pair on numbers that print identically).

    summary_csv : path to a summary_appendix_common.csv written by
                  compute_fwi_results.compute().
    """
    df = pd.read_csv(summary_csv).set_index("method")

    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & ST-RMSE (m/s)\\,$\\downarrow$ & MS-SSIM\\,$\\uparrow$ & PSNR (dB)\\,$\\uparrow$ \\\\",
        "\\midrule",
    ]

    metric_specs = [("tissue_rmse", 1, True), ("ms_ssim", 4, False), ("psnr", 2, False)]
    ranks = {}
    for metric, prec, ascending in metric_specs:
        rounded = df[f"{metric}_mean"].round(prec)
        ranks[metric] = rounded.rank(method="min", ascending=ascending)

    for method in METHODS_ORDER:
        if method not in df.index:
            continue
        row = df.loc[method]
        cells = [METHOD_SHORT_LABELS[method]]
        for metric, prec, _ in metric_specs:
            rank = ranks[metric].loc[method]
            best_rank = 1 if rank == 1 else (2 if rank == 2 else None)
            cells.append(_fmt_pm(row[f"{metric}_mean"], row[f"{metric}_std"], prec, best_rank))
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]

    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    with open(output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_tex}")


# ════════════════════════════════════
# Table for diffusion model ablation
# ════════════════════════════════════
ROW_ORDER = ["best_case", "size_small", "reference", "size_large", "no_split",
             "aug_none", "worst_case", "best_case_aug", "aug_full", "worst_case_aug"]
REAL_MODELS = ["reference", "size_small", "size_large", "no_split", "aug_none"]
AUG_BLOCK_START = "best_case_aug"

ROW_LABELS = {
    "best_case":      "\\textit{Real$\\leftrightarrow$Real}",
    "reference":      "Base ($C=64$)",
    "size_small":     "Small ($C=32$)",
    "size_large":     "Large ($C=128$)",
    "no_split":       "No channel split",
    "aug_none":       "No augmentation",
    "worst_case":     "\\textit{Real$\\leftrightarrow$Blurred}",
    "best_case_aug":  "\\textit{Aug.$\\leftrightarrow$Aug.}",
    "aug_full":       "Full augmentation$^\\dagger$",
    "worst_case_aug": "\\textit{Aug.$\\leftrightarrow$Blurred}",
}

# (json key for mean, json key for std, decimals, ascending-is-better,
#  rank_by_abs: True for the four Delta columns, where "best" = closest to 0)
TABLE2_METRICS = [
    ("fid_in",  "fid_in_std",  2, True,  False),
    ("fid_rad", "fid_rad_std", 4, True,  False),
    ("lpips",   None,          3, False, False),
]

# aug_full is trained on the augmented distribution, so its row is evaluated
# against an augmented reference instead of the clean val+test reference
AUG_REFERENCE_MODELS = {"aug_full"}


def _load_table2_row(base_dir, model_name, seed_glob="*"):
    """
    Aggregate upd_metrics.json across all seeds for one row (mean/std across
    seeds).
    """
    metrics_filename = "upd_metrics_aug_ref.json" if model_name in AUG_REFERENCE_MODELS else "upd_metrics.json"
    paths = sorted(glob.glob(os.path.join(base_dir, f"{model_name}_seed{seed_glob}", metrics_filename)))
    if not paths:
        return None

    records = []
    for p in paths:
        with open(p) as f:
            records.append(json.load(f))

    def mean_std(key):
        vals = [r[key] for r in records if key in r]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    row = {}
    row["fid_in_mean"], row["fid_in_std"] = mean_std("fid_in")
    row["fid_rad_mean"], row["fid_rad_std"] = mean_std("fid_rad")
    row["lpips_mean"], row["lpips_std"] = mean_std("lpips")

    # Delta columns: relative deviation of generated ST/SK velocity mean/std
    # from reference
    for tag, prefix in [("st", "soft_tissue"), ("sk", "skull")]:
        for stat in ("mean", "std"):
            real_key = f"{prefix}_{stat}_real"
            gen_key = f"{prefix}_{stat}_gen"
            vals = []
            for r in records:
                if real_key in r and gen_key in r and r[real_key] != 0:
                    vals.append((r[gen_key] - r[real_key]) / r[real_key] * 100.0)
            m, s = (float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)) \
                if vals else (float("nan"), float("nan"))
            row[f"{tag}_d{stat}_mean"] = m
            row[f"{tag}_d{stat}_std"] = s

    return row


def write_diffusion_ablation_latex_table(base_dir, output_tex, caption,
                                          label="tab:diffusion_metrics",
                                          precomputed_json=None):
    """
    Diffusion model sample quality / ablation. Reads
    `{base_dir}/{model}_seed*/upd_metrics.json` for every model name in
    ROW_ORDER (aggregating mean+-std across seeds).
    """
    if precomputed_json:
        with open(precomputed_json) as f:
            rows = json.load(f)
        missing = [m for m in ROW_ORDER if m not in rows]
        if missing:
            raise KeyError(f"precomputed_json {precomputed_json} is missing rows for: {missing}")
    else:
        rows = {}
        missing = []
        for model in ROW_ORDER:
            row = _load_table2_row(base_dir, model)
            if row is None:
                missing.append(model)
            else:
                rows[model] = row
        if missing:
            raise FileNotFoundError(
                f"No upd_metrics.json found for: {missing} under {base_dir}/<model>_seed*/. "
                "Run paper-pipeline/evaluate.py (and compute_bounds.py for the best_case/"
                "worst_case rows) first."
            )

    delta_cols = ["st_dmean", "st_dstd", "sk_dmean", "sk_dstd"]
    ranks = {}
    for col in ["fid_in_mean", "fid_rad_mean"]:
        ranks[col] = _rank_within({m: rows[m][col] for m in REAL_MODELS}, ascending=True)
    ranks["lpips_mean"] = _rank_within({m: rows[m]["lpips_mean"] for m in REAL_MODELS}, ascending=False)
    for col in delta_cols:
        ranks[f"{col}_mean"] = _rank_within(
            {m: abs(rows[m][f"{col}_mean"]) for m in REAL_MODELS}, ascending=True
        )

    def cell(model, mean_key, std_key, prec, signed=False):
        m, s = rows[model][mean_key], rows[model][std_key]
        if pd.isna(m):
            return "--"
        rank = ranks.get(mean_key, {}).get(model)
        m_str = f"{m:+.{prec}f}" if signed else f"{m:.{prec}f}"
        s_str = f"{s:.{prec}f}"
        if rank == 1:
            return f"$\\mathbf{{{m_str}}} \\pm \\mathbf{{{s_str}}}$"
        if rank == 2:
            return f"$\\underline{{{m_str}}} \\pm \\underline{{{s_str}}}$"
        return f"${m_str} \\pm {s_str}$"

    lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Model & FID-In $\\downarrow$ & FID-Rad $\\downarrow$ & LPIPS $\\uparrow$ & "
        "ST $\\Delta \\mu$ (\\%) & ST $\\Delta \\mathrm{std}$ (\\%) & "
        "SK $\\Delta \\mu$ (\\%) & SK $\\Delta \\mathrm{std}$ (\\%) \\\\",
        "\\midrule",
    ]

    for model in ROW_ORDER:
        if model == AUG_BLOCK_START:
            lines.append("\\midrule")
        cells = [
            ROW_LABELS[model],
            cell(model, "fid_in_mean", "fid_in_std", 2),
            cell(model, "fid_rad_mean", "fid_rad_std", 4),
            cell(model, "lpips_mean", "lpips_std", 3),
            cell(model, "st_dmean_mean", "st_dmean_std", 2, signed=True),
            cell(model, "st_dstd_mean", "st_dstd_std", 2, signed=True),
            cell(model, "sk_dmean_mean", "sk_dmean_std", 2, signed=True),
            cell(model, "sk_dstd_mean", "sk_dstd_std", 2, signed=True),
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\vspace{4pt}",
        "\\begin{minipage}{\\textwidth}",
        "\\tiny",
        "$^\\dagger$Evaluated against the augmented reference distribution, not directly "
        "comparable to the clean-reference rows above.",
        "\\end{minipage}",
        "\\end{table*}",
    ]

    os.makedirs(os.path.dirname(output_tex), exist_ok=True)
    with open(output_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_tex}")
