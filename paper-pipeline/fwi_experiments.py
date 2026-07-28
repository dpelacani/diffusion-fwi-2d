import numpy as np

BASE_FWI = {
    "data_dir":   "", # set before running
    "input_dir":  "", # set before running
    "output_dir": "", # set before running
    "sweep_output_dir":  "", # set before running
    "sweep_results_dir": "", # set before running

    # FWI grid
    "shape": (320, 256),
    "spacing": (0.5e-3, 0.5e-3), # m/pixel
    "extra": (50, 50),
    "absorbing": (40, 40),

    # Time discretization
    "time_step": 0.08e-6, # s
    "num_steps": 2500,

    # Transducer geometry
    "num_transducers": 256,
    "num_shots": 32,

    # Optimizer
    "step_size": 5,

    # Diffusion model setup
    "input_dim": 128,
    "T": 1000,
    "s": 0.001
}   

# Pre-compute stride shots (every) from num_transducers and num_shots
BASE_FWI["every"] = BASE_FWI["num_transducers"] // BASE_FWI["num_shots"]

# -------- Diffusion model checkpoints --------
DIFFUSION_MODELS = {
    "reference": {
        "checkpoint": "", # set before running
        "layers": 4, "channels": 64, "split": True
    },
    "aug_full": {
        "checkpoint": "", # set before running
        "layers": 4, "channels": 64, "split": True
    }
}

# --------------- FWI experiments ---------------

EXPERIMENTS = {
    # Full frequency sweep, extra iterations in lowest band to escape cycle skipping
    "standard": {
        "run_dir_suffix": "reference_v2",
        "block_iters":    [32, 8, 8, 8, 8, 8, 8],
        "max_freqs":      [0.1e6, 0.15e6, 0.2e6, 0.25e6, 0.3e6, 0.35e6, 0.4e6]
    },
    # Compute constrained: full frequency sweep, single FWI iteration per frequency band
    "reduced_compute": {
        "run_dir_suffix": "restricted",
        "block_iters":    [8, 8, 8, 8, 8, 8, 8],
        "max_freqs":      [0.1e6, 0.15e6, 0.2e6, 0.25e6, 0.3e6, 0.35e6, 0.4e6]
    },
    # Low-frequency content missing: same total number of iterations
    "missing_low_freq": {
        "run_dir_suffix": "missing_low_freq_v2",
        "block_iters":    [32, 16, 16, 8, 8],
        "max_freqs":      [0.2e6, 0.25e6, 0.3e6, 0.35e6, 0.4e6]
    }
}

for exp in EXPERIMENTS.values():
    exp["iters_to_run"] = list(np.cumsum(exp["block_iters"]).astype(int))
    exp["total_iters"] = exp["iters_to_run"][-1]

# ---------- FWI guidance strategies ----------
METHODS = {
    "baseline": {
        "run_dir_suffix": "baseline",
        "guidance": None,
        "diffusion_model": None
    },

    # Denoising guidance (DG/DG-C): applied via process_model at end of each
    # freq. block, denoises the FWI output velocity map and blends them together
    "dg": {
        "run_dir_suffix":   "postprocessing_merged",
        "guidance":         "denoising",
        "split":            True, # model training split independent of guidance split
        "diffusion_model":  "reference",
        "t_start":          500,
        "t_end":            100,
        "alpha_skull":      0.9,
        "alpha_tissue":     0.9, # set alpha_skull and alpha_tissue to equal values
        "alpha_end":        0.1
    },
    "dg_c": {
        "run_dir_suffix":   "postprocessing_split",
        "guidance":         "denoising",
        "split":            True,
        "diffusion_model":  "reference",
        "t_start":          500,
        "t_end":            100,
        "alpha_skull":      0.7,
        "alpha_tissue":     0.5,
        "alpha_end":        0.1,  # end value shared across channels
        "warmstart_steps":  1     # provide prior-only tissue warmstart at end of first iteration round
    },

    # Score guidance (SG/SG-C): applied via process_grad at every FWI iteration
    # adds diffusion score to FWI gradient
    "sg": {
        "run_dir_suffix":  "gradient_merged",
        "guidance":        "score",
        "split":           True, # model training split independent of guidance split
        "diffusion_model": "reference",
        "t_start":         700,
        "t_end":           100,
        "lambda_skull":    1.3,
        "lambda_tissue":   1.3, # set lambda_skull and lambda_tissue to equal values
        "lambda_end":      0.01
    },
    "sg_c": {
        "run_dir_suffix":   "gradient_split",
        "guidance":         "score",
        "split":            True,
        "diffusion_model":  "reference",
        "t_start":          700,
        "t_end":            100,
        "lambda_skull":     1.2,
        "lambda_tissue":    0.5,
        "lambda_end":       0.01,
        "warmstart_steps":  1     # provide prior-only tissue warmstart at end of first iteration round
    },
}

FWI_EXPERIMENTS = [
    {"experiment": exp, "method": mt}
    for exp in EXPERIMENTS
    for mt in METHODS
]