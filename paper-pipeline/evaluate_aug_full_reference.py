import os
import json
import logging

import torch

from diffusionfwi.dataset import build_dataset, build_augm_reference_dataset
from diffusionfwi.utils import set_seed
from diffusionfwi.utils.metrics import (
    RadImageNetFeaturesFID,
    tissue_statistics,
    pixel_statistics,
    compute_fid,
    compute_lpips,
)

from diff_experiments import BASE

N_FID_BATCH = 200
N_FID_REPS = 5

AUG_FULL_NAME = "aug_full"
NUM_REPS = 5          # augmented versions per val+test image
REF_SEED = 0          


def _build_reference(seed, num_reps=NUM_REPS):
    """Build the augmented reference set under a specific construction seed."""
    set_seed(seed)
    return build_augm_reference_dataset(
        BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"], num_reps=num_reps,
    )

def compute_aug_full_row(ref_seed=REF_SEED, num_reps=NUM_REPS, seeds=None):
    """
    Computes aug_full's metrics against the augmented
    reference, for each of its existing model-training seeds, reusing the
    samples.npy each seed's evaluate.py run already generated.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("aug_ref_eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rad_model = RadImageNetFeaturesFID(BASE["rad_model_path"]).to(device)

    seeds = seeds if seeds is not None else BASE["seeds"]

    aug_reference = _build_reference(ref_seed, num_reps)
    logger.info("Augmented reference built: %d images (seed=%d, num_reps=%d)",
                len(aug_reference), ref_seed, num_reps)

    for seed in seeds:
        run_dir = os.path.join(BASE["work_dir"], f"{AUG_FULL_NAME}_seed{seed}")
        samples_path = os.path.join(run_dir, "samples.npy")
        if not os.path.exists(samples_path):
            logger.warning("No samples found for seed %d at %s, skipping.", seed, samples_path)
            continue

        import numpy as np
        samples = torch.from_numpy(np.load(samples_path)).float()

        fid_rad_scores, fid_in_scores = [], []
        for i in range(N_FID_REPS):
            batch = samples[i * N_FID_BATCH:(i + 1) * N_FID_BATCH]
            fid_rad_scores.append(compute_fid(aug_reference, batch, device, rad_model=rad_model))
            fid_in_scores.append(compute_fid(aug_reference, batch, device))

        metrics = {
            "fid_rad": float(torch.tensor(fid_rad_scores).mean()),
            "fid_rad_std": float(torch.tensor(fid_rad_scores).std()),
            "fid_in": float(torch.tensor(fid_in_scores).mean()),
            "fid_in_std": float(torch.tensor(fid_in_scores).std()),
            "lpips": float(compute_lpips(samples, device)),
            **tissue_statistics(aug_reference, samples),
            "aug_reference_seed": ref_seed,
            "aug_reference_n": len(aug_reference),
        }

        out_path = os.path.join(run_dir, "upd_metrics_aug_ref.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Seed %d -> FID-Rad=%.4f FID-In=%.2f -> saved %s",
                    seed, metrics["fid_rad"], metrics["fid_in"], out_path)


if __name__ == "__main__":
    compute_aug_full_row()