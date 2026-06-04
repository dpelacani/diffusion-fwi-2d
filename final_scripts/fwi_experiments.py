import numpy as np

BASE_FWI = {
    "data_dir":   "", # set before running
    "input_dir":  "", # set before running
    "output_dir": "", # set before running

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
        "checkpoint": "/cluster/scratch/fscharitzer/diffusion/final_runs/reference_seed42/checkpoints/ema_final.pth", # set before running
        "layers": 4, "channels": 64, "split": True
    },
    "aug_full": {
        "checkpoint": "", # set before running
        "layers": 4, "channels": 64, "split": True
    }
}

# --------------- FWI scenarios ---------------
SCENARIOS = {
    # Full frequency sweep, extra iterations in lowest band to escape cycle skipping
    "reference": {
        "block_iters": [24, 8, 8, 8, 8, 8, 8],
        "max_freqs":   [0.1e6, 0.15e6, 0.2e6, 0.25e6, 0.3e6, 0.35e6, 0.4e6]
    },
    # Compute constrained: full frequency sweep, single FWI iteration per frequency band 
    "restricted": {
        "block_iters": [8, 8, 8, 8, 8, 8, 8],
        "max_freqs":   [0.1e6, 0.15e6, 0.2e6, 0.25e6, 0.3e6, 0.35e6, 0.4e6]
    },
    # Low-frequency content missing: same total number of iterations
    "missing_low_freq": {
        "block_iters": [24, 16, 16, 8, 8],
        "max_freqs":   [0.2e6, 0.25e6, 0.3e6, 0.35e6, 0.4e6]
    }
}

for sc in SCENARIOS.values():
    sc["iters_to_run"] = list(np.cumsum(sc["block_iters"]).astype(int))
    sc["total_iters"] = sc["iters_to_run"][-1]

# ---------- FWI guidance strategies ----------
METHODS = {
    "baseline": {
        "guidance": None,
        "diffusion_model": None
    },

    # Post-processing guidance: applied via process_model at end of each freq. block
    # denoises the FWI output velocity map and blends them together
    "postprocessing_merged": {
        "guidance":        "postprocessing",
        "split":           True, # model training split independent of guidance split
        "diffusion_model": "reference",
        "t_start":         600,
        "t_end":           100,
        "alpha_skull":     0.9, 
        "alpha_tissue":    0.9, # set alpha_skull and alpha_tissue to equal values
        "alpha_end":       0.1
    },
    "postprocessing_split": {
        "guidance":        "postprocessing",
        "split":           True,
        "diffusion_model": "reference",
        "t_start":         600,
        "t_end":           100,
        "alpha_skull":     0.9,
        "alpha_tissue":    0.7,
        "alpha_end":       0.1  # end value shared across channels
    },

    # Gradient guidance: applied via process_grad at every FWI iteration
    # adds diffusion score to FWI gradient
    "gradient_merged": {
        "guidance":        "gradient",
        "split":           True, # model training split independent of guidance split
        "diffusion_model": "reference",
        "t_start":         700,
        "t_end":           100,
        "lambda_skull":    1.3,
        "lambda_tissue":   1.3, # set lambda_skull and lambda_tissue to equal values
        "lambda_end":      0.01
    },
    "gradient_split": {
        "guidance":        "gradient",
        "split":           True,
        "diffusion_model": "reference",
        "t_start":         700,
        "t_end":           100,
        "lambda_skull":    1.5,
        "lambda_tissue":   0.7,
        "lambda_end":      0.01
    },
}

FWI_EXPERIMENTS = [
    {"scenario": sc, "method": mt}
    for sc in SCENARIOS
    for mt in METHODS
]