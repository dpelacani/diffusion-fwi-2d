from matplotlib import pyplot as plt
import torch

def visualize_data(x):
    """
    Visualize the first 10 images.

    Args:
        x: Tensor of shape (N, C, H, W) to plot.
    """
    # Visualize the first 10 images in the training set
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    for i, ax in enumerate(axes.flat):
        img = x[i].detach().squeeze().numpy()
        ax.imshow(img, cmap = 'terrain', origin='lower')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_sample_vs_real(real_img, generate_img):
    """
    Plot the real images and the generated images side by side.

    Args:
        real_img: Tensor of real images.
        generate_img: Tensor of generated images.
    """
    print(f'Real Images(Test Set):')
    visualize_data(real_img.cpu())
    print(f'Sample Images(Model predict):')
    visualize_data(generate_img.cpu())

def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plot the training and validation losses.

    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        save_path: Path to save the loss plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_batch(batch, nrow=4, title=None, save_path=None, cmap='terrain', vmin=None, vmax=None):
    """
    Plot a batch of images.

    Args:
        batch: Tensor of shape (N, C, H, W) to plot.
        title: Optional title for the plot.
    """
    from torchvision.utils import make_grid
    ncols = batch.size(0) // nrow + (batch.size(0) % nrow > 0)
    grid = make_grid(batch.cpu(), nrow=nrow, normalize=True)
    plt.figure(figsize=(5 * nrow, 5 * ncols))
    plt.imshow(grid[0], cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()