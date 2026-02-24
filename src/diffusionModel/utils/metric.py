import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torcheval.metrics.image.fid import FrechetInceptionDistance
from ignite.metrics import MaximumMeanDiscrepancy
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy.linalg import sqrtm
import lpips
from typing import Dict
import random
from itertools import combinations
import matplotlib.pyplot as plt  
    
class GetBackbone(nn.Module):
    """
    Get the backbone of the Inception V3 model pretrained on RadImageNet.
    It uses the Inception V3 architecture without the final classification layers.
    """
    def __init__(self):
        super().__init__()
        base_model = inception_v3(aux_logits=False, transform_input=False, pretrained=False)
        encoder_layers = list(base_model.children())
        # remove the last two layers, extract the 2048-dim features
        self.backbone = nn.Sequential(*encoder_layers[:-2])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x
    
class RadImageNetFeaturesFID(nn.Module):
    """
    Inception V3 model pretrained on RadImageNet for FID computation.
    This model is used to extract features from the input images for FID calculation.
    Loads the model weights from the specified path for future FID calculations.

    Args:
        w_path: Path to the model weights file.
    """
    def __init__(self, w_path: str):
        super().__init__()
        self.model = GetBackbone()
        # load the RadImageNet model weights
        checkpoint = torch.load(w_path)
        self.model.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.model(x)
        return x

class EvaluationMetrics:
    """
    A class to compute various evaluation metrics for evaluating the performance of diffusion models.
    It includes methods to normalize images, compute statistics, plot histograms,
    and compute FID, MMD, and LPIPS metrics.
    """
    def __init__(self, real, fake, model, device):
        """
        Initialize the EvaluationMetrics class with real and fake images, model, and device.
        Args:
            real: Tensor of real images.
            fake: Tensor of generated images.
            model: Pretrained RadImageNet model for feature extraction.
            device: Device to run the computations on (e.g., 'cuda' or 'cpu).
        """
        self.device = device
        # store the real and fake images, here is the normalized version
        self.real_norm = real.to(device)
        self.fake_norm = fake.to(device)
        # find the min and max values for later normalization
        all_vals = torch.cat([self.real_norm, self.fake_norm], dim=0)
        self.rev_min = float(all_vals.min())
        self.rev_max = float(all_vals.max())
        # postprocess the images back to original pixel values (velocity)
        self.real = self.postprocess_vp(real)
        self.fake = self.postprocess_vp(fake)
    
    def normalize_images(self, images, reshape_size=(224, 224), normalize=False, reshape=True):
        """
        Normalize images to the range [0, 1] and resize them to the specified shape.

        Args:
            images: Tensor of images to normalize.
            reshape_size: Size to resize the images to.
            normalize: Whether to normalize the images to the range [0, 1].
            reshape: Whether to reshape the images to the specified size.
        
        Returns:
            Normalized and resized images.
        """
        # Normalize to 0-1 if needed
        if normalize:
            images = (images - self.rev_min) / (self.rev_max - self.rev_min)

        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if reshape:
            # Resize to the desired shape
            images = F.interpolate(images, size=reshape_size, mode='bilinear', align_corners=False)
        return images

    def postprocess_vp(self, tensor):
        """
        Post-process the tensor to reverse the normalization and log transformation, back to the original pixel values (velocity).

        Args:
            tensor: Input tensor to post-process.
        
        Returns:
            Post-processed tensor.
        """
        # Denormalize from -1 to 1 back to 0 to 1
        tensor = tensor * 0.5 + 0.5
        # Reverse the log transformation
        tensor = torch.exp(tensor - 1.0)
        tensor = tensor * 3000.0
        # Resize back to original shape
        tensor = F.interpolate(tensor, size=(320, 256), mode='bilinear', align_corners=True)
        return tensor
    
    def statistics(self, x):
        """
        Compute the mean, standard deviation, minimum, and maximum of the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Tuple containing mean, std, min, and max of the tensor.
        """
        # compute mean and std of the images
        mean = x.mean().item()
        std = x.std().item()
        min_val = x.min().item()
        max_val = x.max().item()
        return mean, std, min_val, max_val

    def plot_hist(self, label1="Real", label2="Generated", bins=100, figsize=(6,5), color1='skyblue', color2='salmon', ylim=0.03, log=False, save_path=None):
        """
        Plot histograms of the real and generated images.

        Args:
            label1: Label for the first batch of images.
            label2: Label for the second batch of images.
            bins: Number of bins for the histogram.
            figsize: Size of the figure.
            color1: Color for the first batch of images histogram.
            color2: Color for the second batch of images histogram.
            ylim: Y-axis limit for the histogram.
            log: Whether to use logarithmic scale for the y-axis.
        """
        plt.figure(figsize=figsize)
        # plot histograms on the denormalized images
        real_vals = self.real.detach().cpu().view(-1).numpy()
        fake_vals = self.fake.detach().cpu().view(-1).numpy()

        lo, hi = min(real_vals.min(), fake_vals.min()), max(real_vals.max(), fake_vals.max())
        bin_edges = np.linspace(lo, hi, bins + 1)
        plt.hist(real_vals, bins=bin_edges, density=True, color=color1, edgecolor='black', alpha=0.3, label=label1, log=log)
        plt.hist(fake_vals, bins=bin_edges, density=True, color=color2, edgecolor='black', alpha=0.3, label=label2, log=log)

        plt.title("Histogram for pixel values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Density")
        plt.ylim(top=ylim)
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def center_crop_batch(self, images, crop_size):
        """
        Center crop a batch of images to evaluate only the soft tissues.

        Args:
            images: Tensor of images to crop.
            crop_size: Size to crop the images to.

        Returns:
            Cropped images.
        """
        _, _, H, W = images.shape
        start_y = (H - crop_size) // 2
        start_x = (W - crop_size) // 2
        return images[:, :, start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    def plot_hist_soft_tissue(self, ylim=2e5, save_path=None):
        """
        Plot histograms of the soft tissue pixel values from the real and generated images.

        Args:
            ylim: Y-axis limit for the histogram.
        """
        # center crop the images to evaluate only the soft tissues
        cropped_real = self.center_crop_batch(self.real, 110)
        cropped_fake = self.center_crop_batch(self.fake, 110)
        real_vals = cropped_real.detach().cpu().numpy().flatten()
        fake_vals = cropped_fake.detach().cpu().numpy().flatten()
        lo, hi = min(real_vals.min(), fake_vals.min()), max(real_vals.max(), fake_vals.max())
        bin_edges = np.linspace(lo, hi, 100 + 1)
        plt.figure(figsize=(6, 5))
        plt.hist(real_vals, bins=bin_edges, density=True, color='skyblue', edgecolor='black', alpha=0.3, label='Real')
        plt.hist(fake_vals, bins=bin_edges, density=True, color='salmon', edgecolor='black', alpha=0.3, label='Generated')
        plt.title("Histogram for soft tissue pixel values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Density")
        plt.ylim(top=ylim)
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def select_pairs(self, len, n_pairs=300):
        """
        Select random pairs of indices from the range [0, len) for computing LPIPS.

        Args:
            len: Length of the range to select pairs from.
            n_pairs: Number of pairs to select.

        Returns:
            Two tensors containing the indices of the selected pairs
        """
        # select random pairs of indices for computing LPIPS
        all_pairs = list(combinations(range(len), 2))

        selected_pairs = random.sample(all_pairs, n_pairs)
        idx1, idx2 = zip(*selected_pairs)
        idx1 = torch.tensor(idx1, dtype=torch.long, device=self.device)
        idx2 = torch.tensor(idx2, dtype=torch.long, device=self.device)
        return idx1, idx2
    
    def blur_batch(self, batch, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian blur to a batch of images. This will be used to calculate a bad value for metrics.

        Args:
            batch: Tensor of images to blur.
            kernel_size: Size of the Gaussian kernel.
            sigma: Standard deviation for the Gaussian kernel.

        Returns:
            Blurred images.
        """
        blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return torch.stack([blur(img) for img in batch])
    
    def compute_fid(self, b1 = None, b2 = None, use_radinet=True, model = None):
        """
        Compute the Frechet Inception Distance (FID) between two sets of images.

        Args:
            b1: Tensor of the first batch of images. If None, use self.real_norm.
            b2: Tensor of the second batch of images. If None, use self.fake_norm.
            use_radinet: Whether to use the RadImageNet model for FID computation.
            model: Pretrained RadImageNet model.

        Returns:
            FID score.
        """
        if b1 is None and b2 is None:
            b1, b2 = self.real_norm, self.fake_norm
        # if use_radinet is True, use the RadImageNet model for FID computation, and we don't need to resize the images
        if use_radinet:
            fid = FrechetInceptionDistance(model = model, feature_dim=2048).to(self.device)
            with torch.no_grad():
                fid.update(self.normalize_images(b1).to(self.device), is_real=True)
                fid.update(self.normalize_images(b2).to(self.device), is_real=False)
        else:
            # we use the default inceptionv3 model for FID computation, and we need to resize the images to 299x299
            fid = FrechetInceptionDistance(feature_dim=2048).to(self.device)
            with torch.no_grad():
                fid.update(self.normalize_images(b1, reshape_size=(299, 299), normalize=True).to(self.device), is_real=True)
                fid.update(self.normalize_images(b2, reshape_size=(299, 299), normalize=True).to(self.device), is_real=False)

        return max(float(fid.compute()), 0.0)
    
    def compute_mmd(self, b1 = None, b2 = None, feat = False, model = None, var=1.0):
        """
        Compute the Maximum Mean Discrepancy (MMD) between two sets of images.

        Args:
            b1: Tensor of the first batch of images. If None, use self.real_norm.
            b2: Tensor of the second batch of images. If None, use self.fake_norm.
            feat: Whether to use features from a model for MMD computation.
            model: Pretrained model for feature extraction.
            var: Variance for the MMD computation.

        Returns:
            MMD score.
        """
        mmd = MaximumMeanDiscrepancy(var=var)

        if b1 is None and b2 is None:
            b1, b2 = self.real_norm, self.fake_norm
        b1 = b1.to(self.device)
        b2 = b2.to(self.device)
    
        if feat:
            # extract features using the given model
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                real = model(self.normalize_images(b1))
                fake = model(self.normalize_images(b2))
        else:
            real = b1.flatten(1).detach()
            fake = b2.flatten(1).detach()
        mmd.reset()
        mmd.update((real, fake))
        value = mmd.compute()
        if isinstance(value, torch.Tensor):
            value = value.item()
        return float(value)
    
    def compute_lpips(self, img):
        """
        Compute the Learned Perceptual Image Patch Similarity (LPIPS) between two sets of images.

        Args:
            img: Tensor of images to compute LPIPS on.

        Returns:
            Mean LPIPS distance between pairs of images.
        """
        img = img.to(self.device)
        img = self.normalize_images(img, reshape=False)
        # select pairs of images
        idx1, idx2 = self.select_pairs(img.shape[0])
        loss_fn = lpips.LPIPS(net='alex').to(self.device)
        with torch.no_grad():
            imgs_i = img[idx1]
            imgs_j = img[idx2]
            dists = loss_fn(imgs_i, imgs_j)  # [M, 1, 1, 1] or [M, 1]
            mean_dist = dists.mean().item()
        return mean_dist

