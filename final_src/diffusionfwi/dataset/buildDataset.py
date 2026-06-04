import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from diffusionfwi.constants import VMIN, VMAX, WATER_VELOCITY, SKULL_THRESH_MS, LOG_MIN, LOG_MAX

# Augmentation parameters
AUG_SOFT_TISSUE_SHIFT = (-50.0, 50.0) # m/s, applied before normalization
AUG_SKULL_SHIFT       = (-300.0, 300.0) # m/s, applied before normalization
AUG_ROTATION_DEGREES  = 5.0
AUG_SCALE_X           = (1.0, 1.1)
AUG_SCALE_Y           = (1.0, 1.1)

# Normalized water velocity, used as border fill for rotation and scaling augmentations
WATER_FILL = -1.0   # VMIN maps to -1.0 by construction of LogMinMaxNormalization

class UltrasoundDataset(Dataset):
    """
    A PyTorch Dataset for loading ultrasound images from a directory.
    Each image is expected to be stored as a .npy file.

    Args:
        data_dir: Directory where the dataset is located.
        x_dim: Dimension to which images will be resized (x_dim x x_dim).
        true_model_name: Filename of the true model to exclude from the dataset (e.g., 'vp_1234.npy').
    """
    def __init__(self, data_dir, x_dim, transform=None, true_model_name=None):
        self.data_dir = data_dir
        self.file_list = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".npy")
            and f != true_model_name
        ])

        self.resize = transforms.Resize(
            (x_dim, x_dim),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        #  Additional transformations (normalization and augmentation)
        self.transform = transform
       
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item to retrieve.
        """
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        image = np.load(file_path)
        image = torch.from_numpy(image).float().unsqueeze(0)
        image = self.resize(image)
        if self.transform:
            image = self.transform(image)
        return image

    def plot_histogram(self, bins=100, figsize=(5, 5), color="skyblue"):
        """
        Plot a histogram of the pixel values in the dataset.

        Args:
            bins: Number of bins for the histogram.
            figsize: Size of the figure.
            color: Color of the histogram bars.
        """
        all_data = []
        for fname in self.file_list:
            file_path = os.path.join(self.data_dir, fname)
            arr = np.load(file_path)
            tensor = torch.from_numpy(arr).float().unsqueeze(0)
            if self.transform:
                tensor = self.transform(tensor)
            all_data.append(tensor.flatten().numpy())
        all_data = np.concatenate(all_data)

        plt.figure(figsize=figsize)
        plt.hist(all_data, bins=bins, density=True, color=color, edgecolor="black")
        plt.title("Histogram of Transformed Pixel Values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

class LogMinMaxNormalization(object):
    """
    Normalize log-space velocities to [-1, 1] using physical data bounds.
    Maps water (1480 m/s) to -1.0, skull (3000 m/s) to +1.0.
    """
    def __call__(self, tensor):
        return (tensor - LOG_MIN) / (LOG_MAX - LOG_MIN) * 2.0 - 1.0 

class ReverseLogMinMaxNormalization(object):
    """Reverse LogMinMaxNormalization back to log-space."""
    def __call__(self, tensor):
        return (tensor + 1.0) / 2.0 * (LOG_MAX - LOG_MIN) + LOG_MIN 

class AcousticNormalization(object):
    """
    Normalize the tensor using a logarithmic transformation.

    Args:
        tensor: Input tensor to be normalized.
    """
    def __call__(self, tensor):
        # Clamp to prevent artifacts from bilinear interpolation
        tensor = torch.clamp(tensor, min=VMIN)
        # Normalize the tensor to a range of about [0.5, 1]
        tensor = tensor / 3000.0
        # Apply a logarithmic transformation to compress the range
        # to help with the skewed distribution and help the training more stable
        # Also add 1 to make sure values are positive
        tensor = torch.clamp(tensor, min=1e-6)
        tensor = torch.log(tensor) + 1.0
        return tensor

class ReverseAcousticNormalization(object):
    """
    Reverse the acoustic normalization.

    Args:
        tensor: Input tensor to be reversed.
    """

    def __call__(self, tensor):
        # Reverse the logarithmic transformation
        tensor = torch.exp(tensor - 1.0)
        # Reverse the normalization
        tensor = tensor * 3000.0
        return tensor


class VelocityShift(object):
    """
    Apply independent random shifts to soft tissue and skull regions.
    Shifts are uniformly sampled and applied before normalization (in m/s).
    
    Args:
        tensor: Input tensor with velocities in m/s. 
        soft_tissue_range: (min, max) shift range for soft tissue in m/s.
        skull_range: (min, max) shift range for skull in m/s.
        skull_threshold: Velocity threshold in m/s to separate skull from soft tissue.
    """
    def __init__(self, soft_tissue_range=(-30.0, 30.0), skull_range=(-300.0, 300.0),
                 skull_threshold = 1650.0):
        self.soft_tissue_range = soft_tissue_range
        self.skull_range = skull_range
        self.skull_threshold = skull_threshold

    def __call__(self, tensor):
        # Tensor shape [1, H, W] before normalization
        water_mask = tensor == 1480.0
        skull_mask = tensor > self.skull_threshold
        soft_tissue_mask = (~skull_mask) & (~water_mask)

        skull_shift = torch.FloatTensor(1).uniform_(*self.skull_range).item()
        soft_tissue_shift = torch.FloatTensor(1).uniform_(*self.soft_tissue_range).item()
        
        shifted = tensor.clone()
        shifted[skull_mask] += skull_shift
        shifted[soft_tissue_mask] += soft_tissue_shift
        return shifted


class RandomRotation(object):
    """
    Apply random rotation and fill borders with water value.
    """
    def __init__(self, degrees=20.0, fill=0.0):
        self.degrees = degrees
        self.fill = fill

    def __call__(self, tensor):
        angle = torch.FloatTensor(1).uniform_(-self.degrees, self.degrees).item()
        
        rotated = transforms.functional.rotate(
            tensor, angle, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        
        ones = torch.ones_like(tensor)
        border_mask = transforms.functional.rotate(
            ones, angle, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        
        rotated[border_mask < 0.999] = self.fill
        return rotated

class RandomScaling(object):
    """
    Apply independent random scaling along x and y axes. 
    Scaling factors are uniformly sampled.

    Args:
        tensor: Input tensor with normalized velocities.
        scale_x: (min, max) scale range for horizontal axis.
        scale_y: (min, max) scale range for vertical axis.
        fill: Value to fill in border regions (normalized water velocity)
    """
    def __init__(self, scale_x=(1.0, 1.15), scale_y=(1.0, 1.15), fill=0.0):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.fill = fill 

    def __call__(self, tensor):
        # Tensor shape [1, H, W] with normalized velocities
        sx = torch.FloatTensor(1).uniform_(*self.scale_x).item()
        sy = torch.FloatTensor(1).uniform_(*self.scale_y).item()
        
        # Affine scaling matrix for sample grid 
        scaling_mat = torch.tensor([
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0] 
        ], dtype=torch.float32).unsqueeze(0)

        grid = F.affine_grid(scaling_mat, tensor.unsqueeze(0).shape, align_corners=False)
        scaled = F.grid_sample(tensor.unsqueeze(0), grid, mode="bilinear", 
                            padding_mode="zeros", align_corners=False).squeeze(0)
        
        # Create mask for border pixels to fill with constant value
        ones = torch.ones_like(tensor)
        border_mask = F.grid_sample(ones.unsqueeze(0), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False).squeeze(0)

        scaled[border_mask < 0.999] = self.fill
        return scaled 


def load_true_model(data_dir, true_model, transform=None):
    """Load the true model image from the specified directory.

    Args:
        data_dir: Directory where the true model file is located.
        true_model: Filename of the true model (e.g., 'vp_1234.npy').
        transform: Optional transform to be applied on the image.

    Returns:
        A tensor representing the true model image.
    """
    # Load the true model image from the specified directory
    # It is the model we used in FWI so it should not be in the train set
    path = os.path.join(data_dir, true_model)
    image = np.load(path)
    image = torch.from_numpy(image).float().unsqueeze(0)
    if transform:
        image = transform(image)
    return image


def add_true_model_to_val(val_dataset, true_model_tensor):
    """
    Add the true model image to the validation dataset.
    This is done to ensure that the true model is included in the validation set.

    Args:
        val_dataset: The original validation dataset.
        true_model_tensor: The tensor representing the true model image.

    Returns:
        A new validation dataset that includes the true model image.
    """
    val_tensors = [val_dataset[i] for i in range(len(val_dataset))]
    # Add the true model to the validation dataset
    val_tensors.append(true_model_tensor)

    new_val_tensor = torch.stack(val_tensors)
    new_val_dataset = TensorDataset(new_val_tensor)

    return new_val_dataset

def _save_split(file_list, indices, train_len, val_len, true_model, save_dir):
    """ Write train/val/test filenames to .txt files """
    os.makedirs(save_dir, exist_ok=True)
    splits = {
        "train": indices[:train_len],
        "val":   indices[train_len:train_len + val_len],
        "test":  indices[train_len + val_len:],
    }
    for name, idxs in splits.items():
        with open(os.path.join(save_dir, f"{name}_split.txt"), "w") as f:
            f.write(f"# {name}: {len(idxs)} samples\n")
            if true_model:
                f.write(f"# excluded (true model): {true_model}\n")
            for i in idxs:
                f.write(file_list[i] + "\n")

def build_dataset(data_dir, true_model=None, x_dim=128, 
                  augment=None, save_split_dir=None):
    """
    Build the dataset for training, validation, and testing.

    Args:
        data_dir: Directory where the dataset is located.
        true_model: Filename of the true model (e.g., 'vp_1234.npy').
        x_dim: Dimension to which images will be resized (x_dim x x_dim).
        augment: Data augmentation mode. 
                 None (no augmentation), "flip" (horizontal flip only) 
                 or "full" (flip + rotation + scaling + velocity shift)

    Returns:
        A tuple containing the training, validation, and test datasets.
    """

    transform = transforms.Compose([
        # Apply the log transformation
        AcousticNormalization(),
        # Normalize to -1 to 1 range
        LogMinMaxNormalization()
    ])

    if augment == "flip":
        train_transform = transforms.Compose([
            AcousticNormalization(),
            LogMinMaxNormalization(),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
    elif augment == "full":
        train_transform = transforms.Compose([
            VelocityShift(
               soft_tissue_range=AUG_SOFT_TISSUE_SHIFT,
               skull_range=AUG_SKULL_SHIFT,
               skull_threshold=SKULL_THRESH_MS
            ),
            AcousticNormalization(),
            LogMinMaxNormalization(),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=AUG_ROTATION_DEGREES, fill=WATER_FILL),
            RandomScaling(scale_x=AUG_SCALE_X, scale_y=AUG_SCALE_Y, fill=WATER_FILL)
        ])
    else:
        train_transform = transform

    train_data = UltrasoundDataset(
        data_dir=data_dir, x_dim=x_dim, transform=train_transform, 
        true_model_name=true_model
    )

    val_test_data = UltrasoundDataset(
        data_dir=data_dir, x_dim=x_dim, transform=transform, 
        true_model_name=true_model
    )
    
    # Use fixed index split for reproducibility across global seeds
    n = len(train_data)
    train_len = int(n * 0.8) + 1
    val_len = int(n * 0.1)
    test_len = n - train_len - val_len
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    
    train_dataset = torch.utils.data.Subset(train_data, indices[:train_len])
    val_dataset = torch.utils.data.Subset(val_test_data, indices[train_len:train_len+val_len])
    test_dataset = torch.utils.data.Subset(val_test_data, indices[train_len+val_len:])
    
    if save_split_dir:
        _save_split(train_data.file_list, indices, train_len, val_len,
                    true_model, save_split_dir)

    # Load the true model and add it to the validation dataset
    true_model_tensor = load_true_model(
        data_dir=data_dir, true_model=true_model, transform=transforms.Compose([train_data.resize, transform])
    )
    val_dataset = add_true_model_to_val(val_dataset, true_model_tensor)
    
    return train_dataset, val_dataset, test_dataset

def build_augm_reference_dataset(data_dir, true_model=None, x_dim=128, num_reps=5):
    """
    Build augmented reference images for evaluating the model learning the fully augmented distribution.
    Applies the full augmentation transform num_reps times to each val and test image,
    to produce more robust reference distribution that matches what aug_full was trained on.

    Args:
        data_dir: data directory path.
        true_model: filename of the true model to exclude.
        x_dim: image dimension.
        num_reps: number of augmented versions per image.

    Returns:
        Tensor of shape [len(val_test_indices) * num_reps, 1, H, W] in normalized training space.
    """
    augm_transform = transforms.Compose([
        VelocityShift(
            soft_tissue_range=AUG_SOFT_TISSUE_SHIFT,
            skull_range=AUG_SKULL_SHIFT,
            skull_threshold=SKULL_THRESH_MS,
        ),
        AcousticNormalization(),
        LogMinMaxNormalization(),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=AUG_ROTATION_DEGREES, fill=WATER_FILL),
        RandomScaling(scale_x=AUG_SCALE_X, scale_y=AUG_SCALE_Y, fill=WATER_FILL),
    ])

    augm_data = UltrasoundDataset(
        data_dir=data_dir, x_dim=x_dim, transform=augm_transform, true_model_name=true_model,
    )

    # Replicate deterministic split from training dataset creation
    n = len(augm_data)
    train_len = int(n * 0.8) + 1
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    val_test_indices = indices[train_len:]   # val and test data combined

    # Sample num_reps augmented versions of each val and test image
    # Each access to augm_data[idx] draws fresh random transforms
    all_images = []
    for _ in range(num_reps):
        for idx in val_test_indices:
            all_images.append(augm_data[idx])

    return torch.stack(all_images)

def postprocess_vp(tensor):
    """
    Post-process the tensor to reverse the normalization and log transformation, 
        back to the original pixel values (velocity).

    Args:
        tensor: Input tensor to post-process.

    Returns:
        Post-processed tensor.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).float()

    tensor = ReverseLogMinMaxNormalization()(tensor)
    tensor = ReverseAcousticNormalization()(tensor)
    return tensor