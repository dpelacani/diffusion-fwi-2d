import random
from itertools import combinations

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import MaximumMeanDiscrepancy
from torcheval.metrics.image.fid import FrechetInceptionDistance
from torchvision.models import inception_v3

from diffusionfwi.dataset import postprocess_vp
from diffusionfwi.constants import SKULL_THRESH_MS, LOG_MIN, LOG_MAX
SOFT_TISSUE_CROP = 110

class RadImageNetFeaturesFID(nn.Module):
    """
    InceptionV3 backbone pretrained on RadImageNet for FID computation.
    This model is used to extract features from the input images for FID calculation.
    Loads the model weights from the specified path for future FID calculations.

    Args:
        w_path: Path to the model weights file.
    """
 
    def __init__(self, weights_path: str):
        super().__init__()
        base_model = inception_v3(aux_logits=False, transform_input=False, pretrained=False)
        # Remove the final classification layers, extract the 2048-dim features
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.load_state_dict(torch.load(weights_path))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.backbone(x), 1)

# -------------------------------- HELPERS ---------------------------------

def _normalize_images(images, reshape_size=(224, 224), rescale=False):
    """
    Expand images to 3 channels, resize to target size and optionally rescale to [0,1] range.
    Use as preprocessing for Inception feature extraction.

    Args:
        images: tensor of images to normalize.
        reshape_size: size to resize the images to.
        rescale: whether to reshape the images to the specified size.

    Returns:
        Normalized and resized images.
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    if rescale:
        # Map from training range [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1).contiguous()
    return F.interpolate(images, size=reshape_size, mode="bilinear", align_corners=False)
 
def _select_pairs(length, device, n_pairs=300):
    """
    Select random pairs of indices from the range [0, len) for computing LPIPS.

    Args:
        length: Length of the range to select pairs from.
        n_pairs: Number of pairs to select.

    Returns:
        Two tensors containing the indices of the selected pairs
    """
    # select random pairs of indices for computing LPIPS
    all_pairs = list(combinations(range(length), 2))
    selected_pairs = random.sample(all_pairs, min(n_pairs, len(all_pairs)))

    idx1, idx2 = zip(*selected_pairs)
    idx1 = torch.tensor(idx1, dtype=torch.long, device=device)
    idx2 = torch.tensor(idx2, dtype=torch.long, device=device)
    return idx1, idx2

def _pairwise_distances(U, V):
    """ Squared Euclidean pairwise distances. See batch_pairwise_distances from Kynkäänniemi et al. (2019) """
    norm_u = np.sum(U ** 2, axis=1, keepdims=True)
    norm_v = np.sum(V ** 2, axis=1, keepdims=True)
    return np.maximum(norm_u - 2 * U @ V.T + norm_v.T, 0.0)

def _manifold_radii(features, k):
    """ k-NN radius for each point. See ManifoldEstimator from Kynkäänniemi et al. (2019) """
    D = _pairwise_distances(features, features)
    return np.partition(D, k, axis=1)[:, k]


# -------------------------------- METRICS ---------------------------------

def pixel_statistics(x):
    """
    Compute the mean, standard deviation, minimum, and maximum of pixels in normalized training space.

    Args:
        x: Input tensor.

    Returns:
        Dict with mean, std, min, and max of the tensor.
    """

    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "min": x.min().item(),
        "max": x.max().item(),
    }

def tissue_statistics(real, gen):
    """
    Per-tissue mean and std in velocity space (m/s).
    
    Args:
        real: Real images in normalized training space.
        gen: Generated images in normalized training space.
    
    Returns:
        Dict with mean / std for soft tissue and skull for real and generated images.
    """
    real_ms = postprocess_vp(real)
    gen_ms = postprocess_vp(gen)

    def _center_crop(x, size):
        _, _, H, W = x.shape
        sy, sx = (H - size) // 2, (W - size) // 2
        return x[:, :, sy : sy+size, sx : sx+size]
    
    soft_real = _center_crop(real_ms, SOFT_TISSUE_CROP)
    soft_gen = _center_crop(gen_ms, SOFT_TISSUE_CROP)
        
    skull_real = real_ms[real_ms > SKULL_THRESH_MS]
    skull_gen = gen_ms[gen_ms > SKULL_THRESH_MS]

    return {
        "soft_tissue_mean_real": soft_real.mean().item(), 
        "soft_tissue_std_real":  soft_real.std().item(),
        "soft_tissue_mean_gen":  soft_gen.mean().item(),
        "soft_tissue_std_gen":   soft_gen.std().item(),
        "skull_mean_real":       skull_real.mean().item(),
        "skull_std_real":        skull_real.std().item(),
        "skull_mean_gen":        skull_gen.mean().item(),
        "skull_std_gen":         skull_gen.std().item()
    }

def compute_fid(real, gen, device, rad_model=None):
    """
    Compute the Frechet Inception Distance (FID) between two sets of images.
    Pass rad_model to use RadImageNet features (FID-Rad), else standard InceptionV3 (FID-IN) is used.

    Args:
        real: tensor of real images.
        gen: tensor of generated images.
        rad_model: RadImageNet feature extractor. If None, uses torchvision InceptionV3

    Returns:
        FID score (float, clipped to >= 0).
    """
    if rad_model is not None:
        fid = FrechetInceptionDistance(model=rad_model, feature_dim=2048).to(device)
        preproc = lambda x: _normalize_images(x, rescale=True).to(device)
    else:
        fid = FrechetInceptionDistance(feature_dim=2048).to(device)
        preproc = lambda x: _normalize_images(x, reshape_size=(299, 299), rescale=True).to(device)

    with torch.no_grad():
        fid.update(preproc(real), is_real=True)
        fid.update(preproc(gen), is_real=False)
    
    return max(float(fid.compute()), 0.0)

def compute_precision_recall(real, gen, rad_model, device, k=3):
    """
    Precision and Recall computation with k-NN manifold estimation (Kynkäänniemi et al. 2019).

    Args:
        real: tensor of real images in normalized training space.
        gen: tensor of generated images in normalized training space.
        rad_model: RadImageNet feature extractor for embedding tensors.
        k: number of nearest neighbors for manifold radius estimation.
    
    Returns:
        (precision, recall) as floating point values in [0, 1]
    """
    rad_model = rad_model.to(device)
    rad_model.eval()
    with torch.no_grad():
        real_feats = rad_model(_normalize_images(real).to(device)).cpu().float().numpy()
        gen_feats = rad_model(_normalize_images(gen).to(device)).cpu().float().numpy()

    real_radii = _manifold_radii(real_feats, k)
    gen_radii = _manifold_radii(gen_feats, k)
    cross_dists = _pairwise_distances(gen_feats, real_feats)
    
    precision = float((cross_dists <= real_radii[None, :]).any(axis=1).mean())
    recall = float((cross_dists.T <= gen_radii[None, :]).any(axis=1).mean())

    return precision, recall

def compute_lpips(samples, device, num_pairs=300): # TODO: consider if we keep averaging?
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS) over random pairs of generated images (diversity evaluation).

    Args:
        samples: tensor of generated images to compute LPIPS on, in normalized training space
        num_pairs: number of random pairs to average over.

    Returns:
        Mean LPIPS distance between pairs of images.
    """
    samples = samples.to(device)
    lo, hi = samples.min(), samples.max()
    samples_rescaled = (samples - lo) / (hi - lo + 1e-8) # normalize to [0,1]

    # Select pairs of images
    idx1, idx2 = _select_pairs(samples.shape[0], device, num_pairs)
    loss_fn = lpips.LPIPS(net="alex").to(device)

    with torch.no_grad():
        dists = loss_fn(samples_rescaled[idx1], samples_rescaled[idx2], normalize=True)  # [M, 1, 1, 1] or [M, 1], normalize to [-1,1]
    
    return dists.mean().item()

def compute_mmd(real, gen, device, feat=False, rad_model=None, var=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of images.
    In pixel space (feat=False) or RadImageNet feature space (feat=True).

    Args:
        real: tensor of real images [N, 1, H, W]
        gen: tensor of generated images [N, 1, H, W]
        feat: if True, use features from rad_model for MMD computation.
        rad_model: pretrained model for feature extraction, required when feat=True.
        var: variance for the MMD computation.

    Returns:
        MMD score.
    """
    mmd = MaximumMeanDiscrepancy(var=var)
    real, gen = real.to(device), gen.to(device)

    if feat:
        # Extract features using the pretrained model
        rad_model = rad_model.to(device)
        rad_model.eval()
        with torch.no_grad():
            real = rad_model(_normalize_images(real))
            gen = rad_model(_normalize_images(gen))
    else:
        real = real.flatten(1)
        gen = gen.flatten(1)

    mmd.reset()
    mmd.update((real, gen))
    value = mmd.compute()
    return float(value.item() if isinstance(value, torch.Tensor) else value)