import os
import sys
import logging
import torch
import json
import math
import numpy as np

from diffusionfwi.models import UNet
from diffusionfwi.diffusion import DiffusionProcess
from diffusionfwi.dataset import build_dataset, get_dataloaders, build_augm_reference_dataset
from diffusionfwi.utils import set_seed
from diffusionfwi.utils.metrics import (
    RadImageNetFeaturesFID,
    pixel_statistics,
    tissue_statistics,
    compute_fid,
    compute_precision_recall,
    compute_lpips,
    compute_mmd
)

from diff_experiments import BASE, EXPERIMENTS

# FID is averaged over N_FID_REPS independent batches of N_FID_BATCH samples for more stable estimate.
N_SAMPLES = 1000
N_FID_BATCH = 200
N_FID_REPS = 5
N_PR_K = 3 
N_AUGM_REPS = 5 # number of augmented image versions per val+test image for full augmentation reference distribution
AUG_REFERENCE_SEED = 0

def run_evaluation(exp, seed):

    # ---------------------------- EXPERIMENT SETUP ----------------------------
    
    run_name = f"{exp['name']}_seed{seed}"
    run_dir = os.path.join(BASE["work_dir"], run_name)
    samples_path = os.path.join(run_dir, "samples.npy")

    # Set up logging for looping over experiments
    logger = logging.getLogger(f"eval_{run_name}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(run_dir, "eval.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.info("Starting evaluation: %s", run_name)

    # Set random seed for reproducibility
    set_seed(seed)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------- DATA SETUP -------------------------------

    # No data augmentation for evaluation, split consistent with training
    train_dataset, val_dataset, test_dataset = build_dataset(
        BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"], augment=None,
    )

    # aug_full is trained to model the augmented distribution, evaluate against augmented reference
    if exp.get("augment") == "full":
        set_seed(AUG_REFERENCE_SEED) # set fixed seed for building augmented reference dataset
        real_images = build_augm_reference_dataset(
            BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"], num_reps=N_AUGM_REPS
        )
        set_seed(seed) # set seed back to current model
        logger.info("Using augmneted reference distribution (ref_seed=%d): %d images",
                    AUG_REFERENCE_SEED, len(real_images))
    else:
        # Use val + test sets as real images for more stable FID estimate
        # Val dataset wraps tensors in TensorDataset tuple (needs indexing with [0])
        val_images = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
        test_images = torch.stack([test_dataset[i] for i in range(len(test_dataset))])
        real_images = torch.cat([val_images, test_images], dim=0) # [N, 1, H, W]
        logger.info("Using clean reference distribution: %d (val %d + test %d)",
                    len(real_images), len(val_images), len(test_images))

    # Construct test loader for DiffusionProcess.sample_diffusion
    _, _, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=BASE["batch_size"]
    )


    # ----------------------------- MODEL SETUP --------------------------------
    
    num_channels = 2 if exp["split"] else 1
    model = UNet(
        in_channels=num_channels,
        out_channels=num_channels,
        time_emb_dim=BASE["time_emb_dim"],
        num_layers=exp["layers"],
        base_channels=exp["channels"]
    ).to(device)

    ema_path = os.path.join(run_dir, "checkpoints", "ema_final.pth")
    model.load_state_dict(torch.load(ema_path, map_location=device))
    model.eval()
    logger.info("Loaded EMA checkpoint from %s", ema_path)

    # -------------------------- SAMPLE GENERATION -----------------------------

    if os.path.exists(samples_path):
        samples = torch.from_numpy(np.load(samples_path)).float()
        logger.info(f"Samples already exist, loaded from {samples_path}.")
    else:
        pipeline = DiffusionProcess(
            device=device, T=BASE["num_timesteps"], s=BASE["cosine_schedule_s"]
        )
        batches = []
        num_batches = math.ceil(N_SAMPLES / BASE["batch_size"])
        for i in range(num_batches):
            batch = pipeline.sample_diffusion(
                model, test_loader, BASE["batch_size"], device, num_channels=num_channels
            )
            batches.append(batch.cpu())
            if (i+1) % 10 == 0:
                logger.info("Generated %d/%d samples", (i+1) * BASE["batch_size"], N_SAMPLES)
        
        samples = torch.cat(batches, dim=0)[:N_SAMPLES]
            
        np.save(samples_path, samples.numpy())
        logger.info("Saved %d samples to %s", N_SAMPLES, samples_path)

        
    # ----------------------------- METRIC COMPUTATION -------------------------
    
    rad_model = RadImageNetFeaturesFID(BASE["rad_model_path"])

    # FID: averaged over N_FID_REPS independent batches for stability
    fid_rad_scores, fid_in_scores = [], []
    for i in range(N_FID_REPS):
        batch = samples[i*N_FID_BATCH : (i+1)*N_FID_BATCH]
        fid_rad_scores.append(compute_fid(real_images, batch, device, rad_model=rad_model))
        fid_in_scores.append(compute_fid(real_images, batch, device))
        logger.info("FID for rep %d: Rad: %.2f  IN: %.2f", i+1, fid_rad_scores[-1], fid_in_scores[-1])

    min_sample_sz = min(len(real_images), len(samples))
    # Precision, Recall, LPIPS, MMD computed on all samples
    precision, recall = compute_precision_recall(real_images, samples, rad_model, device, k=N_PR_K)
    lpips_score       = compute_lpips(samples, device)
    mmd               = compute_mmd(real_images[:min_sample_sz], samples[:min_sample_sz], device)
    mmd_feat          = compute_mmd(real_images[:min_sample_sz], samples[:min_sample_sz], device, feat=True, rad_model=rad_model)
    tissue_stats      = tissue_statistics(real_images, samples)
    real_pixel_stats  = {f"real_{k}": v for k, v in pixel_statistics(real_images).items()}
    gen_pixel_stats   = {f"gen_{k}": v for k, v in pixel_statistics(samples).items()}

    metric_dict = {
        "fid_rad":        float(np.mean(fid_rad_scores)),
        "fid_rad_std":    float(np.std(fid_rad_scores)),
        "fid_in":         float(np.mean(fid_in_scores)),
        "fid_in_std":     float(np.std(fid_in_scores)),
        "precision":      float(precision),
        "recall":         float(recall),
        "lpips":          float(lpips_score),
        "mmd":            float(mmd),
        "mmd_feat":       float(mmd_feat),
        "reference_dist": "augmented" if exp.get("augment") == "full" else "clean",
        **tissue_stats,
        **real_pixel_stats,
        **gen_pixel_stats
    }

    metrics_path = os.path.join(run_dir, "upd_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metric_dict, f, indent=4)

    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Results: %s", {k: f"{v:.4f}" for k, v in metric_dict.items()
                                if isinstance(v, float)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed (default: all seeds in BASE)")

    args, _ = parser.parse_known_args()
        
    exps  = [e for e in EXPERIMENTS if args.name is None or e["name"] == args.name]
    seeds = [args.seed] if args.seed is not None else BASE["seeds"]

    for exp in exps:
        for seed in seeds:
            run_evaluation(exp, seed)