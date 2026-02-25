import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, TensorDataset
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import logging

class UltrasoundDataset(Dataset):
    """
    A PyTorch Dataset for loading ultrasound images from a directory.
    Each image is expected to be stored as a .npy file.

    Args:
        data_dir: Directory where the dataset is located.
        x_dim: Dimension to which images will be resized (x_dim x x_dim).
        true_model_name: Filename of the true model to exclude from the dataset (e.g., 'vp_1234.npy').
    """
    def __init__(self, data_dir, x_dim, true_model_name=None):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy') and f != true_model_name])
        self.transform = transforms.Compose([
            # use anitialias = True to avoid aliasing artifacts
            transforms.Resize((x_dim, x_dim), interpolation=InterpolationMode.BILINEAR, antialias=True),
            # apply the log transformation
            AcousticNormalization(),
            # normalize to -1 to 1 range
            transforms.Normalize(mean=[0.5], std=[0.5])])

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
        if self.transform:
            image = self.transform(image)
        return image
    
    def plot_histogram(self, bins=100, figsize=(5,5), color='skyblue'):
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
        plt.hist(all_data, bins=bins, density=True, color=color, edgecolor='black')
        plt.title('Histogram of Transformed Pixel Values')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

class AcousticNormalization(object):
    """
    Normalize the tensor using a logarithmic transformation.

    Args:
        tensor: Input tensor to be normalized.
    """
    def __call__(self, tensor):
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


def load_true_model(data_dir, true_model, transform=None):
    """ Load the true model image from the specified directory.

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
    # Add the true model image to the validation dataset
    val_tensors = [val_dataset[i] for i in range(len(val_dataset))]
    # add the true model to the validation dataset
    val_tensors.append(true_model_tensor) 

    new_val_tensor = torch.stack(val_tensors)
    new_val_dataset = TensorDataset(new_val_tensor)

    return new_val_dataset

def split_dataset(dataset, train_size=0.8, val_size=0.1):
    """
    Split the dataset into training, validation, and test sets.
    By default, the training set will contain 80% of the data, validation set 10%, and test set 10%.

    Args:
        dataset: The full dataset to be split.
        train_size: Proportion of the dataset to include in the training set.
        val_size: Proportion of the dataset to include in the validation set.

    Returns:
        A tuple containing the training, validation, and test datasets.
    """
    total_size = len(dataset)
    train_len = int(total_size * train_size) + 1  # add 1 because we abandon the true model
    val_len = int(total_size * val_size)
    test_len = total_size - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])

def build_dataset(data_dir, true_model=None, x_dim=256):
    """
    Build the dataset for training, validation, and testing.

    Args:
        data_dir: Directory where the dataset is located.
        true_model: Filename of the true model (e.g., 'vp_1234.npy').
        x_dim: Dimension to which images will be resized (x_dim x x_dim).

    Returns:
        A tuple containing the training, validation, and test datasets.
    """
    # Build the dataset
    dataset = UltrasoundDataset(data_dir=data_dir, x_dim=x_dim, true_model_name=true_model)
    # Split the dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    # Load the true model and add it to the validation dataset
    true_model_tensor = load_true_model(data_dir=data_dir, true_model=true_model, transform=dataset.transform)
    val_dataset = add_true_model_to_val(val_dataset, true_model_tensor)
    return train_dataset, val_dataset, test_dataset

