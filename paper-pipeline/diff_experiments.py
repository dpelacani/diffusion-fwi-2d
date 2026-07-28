BASE = {
    "work_dir": "", # set before running
    "data_dir": "", # set before running
    "true_model": "vp_996782.npy",
    "rad_model_path": "", # set before running
    "x_dim": 128,
    "batch_size": 16,
    "num_timesteps": 1000,
    "cosine_schedule_s": 0.001,
    "time_emb_dim": 256,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "adam_betas": (0.9, 0.98),
    "num_epochs": 300,
    "checkpoint_interval": 25,
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

EXPERIMENTS = [
    {
        "name": "reference",
        "layers": 4,
        "channels": 64,
        "split": True,
        "augment": "flip"
    },
    {
        "name": "size_small",
        "layers": 4,
        "channels": 32,
        "split": True,
        "augment": "flip"
    },
    {
        "name": "size_large",
        "layers": 4,
        "channels": 128,
        "split": True,
        "augment": "flip"
    },
    {
        "name": "no_split",
        "layers": 4,
        "channels": 64,
        "split": False,
        "augment": "flip"
    },
    {
        "name": "aug_none",
        "layers": 4,
        "channels": 64,
        "split": True,
        "augment": None
    },
    {
        "name": "aug_full",
        "layers": 4,
        "channels": 64,
        "split": True,
        "augment": "full"
    }
]