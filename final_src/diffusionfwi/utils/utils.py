import torch
import random
import numpy as np

from diffusionfwi.constants import SKULL_THRESH_MS, LOG_MIN, LOG_MAX

def set_seed(seed):
    """ Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
    torch.backends.cudnn.enabled = False

    return True

def velocity_to_normalized(v_ms):
    """
    Convert velocity value (m/s) to normalized space.
    Applies AcousticNormalization and LogMinMaxNormalization.
    """
    log_val = np.log(np.asarray(v_ms, dtype=np.float32) / 3000.0) + 1.0
    return (log_val - LOG_MIN) / (LOG_MAX - LOG_MIN) * 2.0 - 1.0

def split_channels(images):
    """ Split soft tissue and skull velocities into two separate channels through masking based on threshold value. """ 
    skull_thresh_norm = velocity_to_normalized(SKULL_THRESH_MS)

    tissue_mask = (images <= skull_thresh_norm).float()
    skull_mask = (images > skull_thresh_norm).float()
    return torch.cat([images * tissue_mask, images * skull_mask], dim=1) # [N, 2, H, W]

def merge_channels(images):
    """ Combine soft tissue and skull channels to generate output. """
    return images[:, 0:1] + images[:, 1:2] # [N, 1, H, W]

def cosine_schedule(start: float, end: float, K: int) -> list:
    """ Cosine annealing from start to end in K steps. """
    return [
        end + (start - end) * 0.5 * (1.0 + np.cos(np.pi * k / max(K-1, 1)))
        for k in range(K)
    ]