import os
import json
import logging

import torch
import torchvision.transforms.functional as TF

from diffusionfwi.dataset import build_dataset, build_augm_reference_dataset
from diffusionfwi.utils import set_seed
from diffusionfwi.utils.metrics import (
    RadImageNetFeaturesFID,
    tissue_statistics,
    compute_fid,
    compute_lpips,
)

from diff_experiments import BASE

N_REPS = 5          
N_EVAL = 200       

AUG_REFERENCE_SEED = 0
N_AUG_REPS = 5

BLUR_KERNEL_SIZE = 19
BLUR_SIGMA = 3.0


def blur_batch(x, kernel_size=None, sigma=None):
    """
    Gaussian-blur a batch of normalized velocity maps [N, 1, H, W].
    """
    kernel_size = BLUR_KERNEL_SIZE if kernel_size is None else kernel_size
    sigma = BLUR_SIGMA if sigma is None else sigma
    return TF.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)


def _save(metrics, name, seed, logger):
    run_dir = os.path.join(BASE["work_dir"], f"{name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "upd_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved %s rep %d -> %s", name, seed, out_path)


def run_bounds():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("bounds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    rad_model = RadImageNetFeaturesFID(BASE["rad_model_path"]).to(device)

    # ------------------------------- DATA SETUP -----------------------------
    # Identical split to evaluate.py: no augmentation, same fixed seed.
    train_dataset, val_dataset, test_dataset = build_dataset(
        BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"], augment=None,
    )
    val_images = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    test_images = torch.stack([test_dataset[i] for i in range(len(test_dataset))])
    real_images = torch.cat([val_images, test_images], dim=0)  # [N, 1, H, W] — fixed reference, same as evaluate.py
    train_images = torch.stack([train_dataset[i] for i in range(len(train_dataset))])
    logger.info("Real reference (val+test): %d | Train pool (independent real brains): %d",
                len(real_images), len(train_images))

    n_eval = min(N_EVAL, len(train_images))

    for rep in range(N_REPS):
        new_rep = rep + 5
        set_seed(new_rep)

        # best case: val+test reference vs independent real (train) subsample
        idx_b = torch.randperm(len(train_images))[:n_eval]
        train_b = train_images[idx_b]

        metrics_best = {
            "fid_in": compute_fid(real_images, train_b, device),
            "fid_rad": compute_fid(real_images, train_b, device, rad_model=rad_model),
            "lpips": compute_lpips(train_b, device),
            **tissue_statistics(real_images, train_b),
        }
        _save(metrics_best, "best_case", new_rep, logger)

        # worst case: val+test reference vs blurred independent real (train) subsample
        idx_w = torch.randperm(len(train_images))[:n_eval]
        blur_w = blur_batch(train_images[idx_w])

        metrics_worst = {
            "fid_in": compute_fid(real_images, blur_w, device),
            "fid_rad": compute_fid(real_images, blur_w, device, rad_model=rad_model),
            "lpips": compute_lpips(blur_w, device),
            **tissue_statistics(real_images, blur_w),
        }
        _save(metrics_worst, "worst_case", new_rep, logger)

        logger.info(
            "Rep %d — best FID-Rad: %.4f | worst FID-Rad: %.4f",
            new_rep, metrics_best["fid_rad"], metrics_worst["fid_rad"],
        )


def run_bounds_aug():
    """
    Run bound computation in augmented-distribution.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("bounds_aug")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rad_model = RadImageNetFeaturesFID(BASE["rad_model_path"]).to(device)

    set_seed(AUG_REFERENCE_SEED)
    aug_reference = build_augm_reference_dataset(
        BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"],
        num_reps=N_AUG_REPS, split="val_test",
    )
    logger.info("Augmented reference (matches aug_full's): %d images", len(aug_reference))

    n_eval = min(N_EVAL, 800) 

    for rep in range(N_REPS):
        new_rep = rep + 5
        set_seed(new_rep)

        # best_case_aug: augmented reference vs independent augmented train sample
        aug_train_pool = build_augm_reference_dataset(
            BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"],
            num_reps=1, split="train",
        )
        idx_b = torch.randperm(len(aug_train_pool))[:n_eval]
        train_b = aug_train_pool[idx_b]

        metrics_best = {
            "fid_in": compute_fid(aug_reference, train_b, device),
            "fid_rad": compute_fid(aug_reference, train_b, device, rad_model=rad_model),
            "lpips": compute_lpips(train_b, device),
            **tissue_statistics(aug_reference, train_b),
        }
        _save(metrics_best, "best_case_aug", new_rep, logger)

        # worst_case_aug: augmented reference vs blurred augmented train sample 
        idx_w = torch.randperm(len(aug_train_pool))[:n_eval]
        blur_w = blur_batch(aug_train_pool[idx_w])

        metrics_worst = {
            "fid_in": compute_fid(aug_reference, blur_w, device),
            "fid_rad": compute_fid(aug_reference, blur_w, device, rad_model=rad_model),
            "lpips": compute_lpips(blur_w, device),
            **tissue_statistics(aug_reference, blur_w),
        }
        _save(metrics_worst, "worst_case_aug", new_rep, logger)

        logger.info(
            "Rep %d — best_aug FID-Rad: %.4f | worst_aug FID-Rad: %.4f",
            new_rep, metrics_best["fid_rad"], metrics_worst["fid_rad"],
        )


def plot_blur_sanity_check(out_path="blur_sanity_check.png", n_show=4):
    """
    Visual check of blur strength, adjust BLUR_KERNEL_SIZE / BLUR_SIGMA.
    """
    import matplotlib.pyplot as plt

    _, val_dataset, test_dataset = build_dataset(
        BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"], augment=None,
    )
    val_images = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    real_images = val_images[:n_show]
    blurred = blur_batch(real_images)

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    for i in range(n_show):
        axes[0, i].imshow(real_images[i, 0], cmap="jet")
        axes[0, i].set_title("Real")
        axes[0, i].axis("off")
        axes[1, i].imshow(blurred[i, 0], cmap="jet")
        axes[1, i].set_title(f"Blurred (k={BLUR_KERNEL_SIZE}, σ={BLUR_SIGMA})")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved sanity check to {out_path}")


def check_blur_strength(n_check=50):
    """
    Print real vs blurred skull/tissue statistics.
    """
    from diffusionfwiclean.utils import SKULL_THRESH_MS

    _, val_dataset, test_dataset = build_dataset(
        BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"], augment=None,
    )
    val_images = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    real_images = val_images[:n_check]
    blurred = blur_batch(real_images)

    stats = tissue_statistics(real_images, blurred)

    # Same denormalization formula used internally by tissue_statistics /
    # postprocess_vp: undo Normalize(0.5, 0.5), then undo the log scaling.
    def _to_ms(x):
        return torch.exp((x * 0.5 + 0.5) - 1.0) * 3000.0

    real_ms = _to_ms(real_images)
    blur_ms = _to_ms(blurred)
    frac_real = (real_ms > SKULL_THRESH_MS).float().mean().item()
    frac_blur = (blur_ms > SKULL_THRESH_MS).float().mean().item()

    print(f"\nBlur config: k={BLUR_KERNEL_SIZE}, sigma={BLUR_SIGMA}  (n={n_check} images)")
    print(f"{'':22s}{'real':>12s}{'blurred':>12s}")
    print(f"{'skull mean (m/s)':22s}{stats['skull_mean_real']:12.1f}{stats['skull_mean_gen']:12.1f}")
    print(f"{'skull std  (m/s)':22s}{stats['skull_std_real']:12.1f}{stats['skull_std_gen']:12.1f}")
    print(f"{'soft mean  (m/s)':22s}{stats['soft_tissue_mean_real']:12.1f}{stats['soft_tissue_mean_gen']:12.1f}")
    print(f"{'soft std   (m/s)':22s}{stats['soft_tissue_std_real']:12.1f}{stats['soft_tissue_std_gen']:12.1f}")
    print(f"{'% px > skull thresh':22s}{100*frac_real:11.2f}%{100*frac_blur:11.2f}%")


if __name__ == "__main__":
    run_bounds()
    run_bounds_aug()