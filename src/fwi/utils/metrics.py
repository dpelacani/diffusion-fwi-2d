import torch
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
import matplotlib.animation as animation
import numpy as np
from IPython.display import Image, display

def preprocess_vp(tensor):
    """
    Preprocess a velocity model tensor for metrics evaluation.

    Args:
        tensor: Input tensor to preprocess.

    Returns:
        Preprocessed tensor.
    """
    if tensor.dim() > 2:
        tensor = tensor.squeeze()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    # normalize to [0, 1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # ensure the tensor is in float format
    if not tensor.is_floating_point():
        tensor = tensor.float()
    # ensure the tensor is in the range [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor

def get_vp_from_file(file_dir, end_frame='00096', file_suffix='.h5', true_model=False):
    """
    Load a velocity model from a file and preprocess it.

    Args:
        file_path (str): Path to the file containing the velocity model.

    Returns:
        torch.Tensor: Preprocessed velocity model tensor.
    """
    # True Model was not named with anastasio, so we need to deal with it separately
    if true_model:
        vp_h5 = f"{file_dir}/BrainTrueModel{file_suffix}"
    else:
        vp_h5 = f"{file_dir}/anastasio2D-Vp-{end_frame}{file_suffix}"

    with h5py.File(vp_h5, 'r') as f:
            vp = f['data'][()]
    vp = torch.tensor(vp, dtype=torch.float32)
    return vp

def compute_ms_ssim(file_dir1, file_dir2, data_range=1.0, suffix=None, end_frame='00096'):
    """
    Compute the Multi-Scale Structural Similarity Index (MS-SSIM) between two images.

    Args:
        file_dir1 (str): Path to the first image file.
        file_dir2 (str): Path to the second image file.
        data_range (float): The data range of the input images. Default is 1.0.

    Returns:
        torch.Tensor: The MS-SSIM score.
    """
    if suffix:
        image1 = get_vp_from_file(file_dir1, file_suffix=suffix, true_model=True, end_frame=end_frame)
    else:
        image1 = get_vp_from_file(file_dir1, true_model=True, end_frame=end_frame)
    # the data need to be preprocessed to [0, 1]
    image1 = preprocess_vp(image1)
    image2 = get_vp_from_file(file_dir2, end_frame=end_frame)
    image2 = preprocess_vp(image2)
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape for MS-SSIM computation.")
    ms_ssim = MS_SSIM(data_range=data_range)
    return ms_ssim(image1, image2)

def compute_psnr(file_dir1, file_dir2, data_range=1.0, suffix=None, end_frame='00096'):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        file_dir1 (str): Path to the first image file.
        file_dir2 (str): Path to the second image file.
        data_range (float): The data range of the input images. Default is 1.0.

    Returns:
        float: The PSNR value.
    """
    if suffix:
        image1 = get_vp_from_file(file_dir1, file_suffix=suffix, true_model=True, end_frame=end_frame)
    else:
        image1 = get_vp_from_file(file_dir1, true_model=True, end_frame=end_frame)
    # the data need to be preprocessed to [0, 1]
    image1 = preprocess_vp(image1)
    image2 = get_vp_from_file(file_dir2, end_frame=end_frame)
    image2 = preprocess_vp(image2)
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape for PSNR computation.")
    
    mse = F.mse_loss(image1, image2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if there is no noise
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()

def plot_comparison(raw_dir, true_dir, diff_dir, suffix=None, end_frame='00096'):
    """
    Plot the comparison of raw, with diffusion, and true velocity models. I did this plot here instead of the plotting.py file to avoid using Stride package.

    Args:
        raw_dir (str): Directory containing the raw velocity model.
        diff_dir (str): Directory containing the velocity model with diffusion prior.
        true_dir (str): Directory containing the true velocity model.
    """


    raw_vp = get_vp_from_file(raw_dir, end_frame=end_frame)
    diff_vp = get_vp_from_file(diff_dir, end_frame=end_frame)
    if suffix:
        true_vp = get_vp_from_file(true_dir, file_suffix=suffix, true_model=True, end_frame=end_frame)
    else:
        true_vp = get_vp_from_file(true_dir, true_model=True, end_frame=end_frame)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Raw Velocity Model')
    # plot result: fwi only
    plt.imshow(raw_vp.squeeze().cpu().numpy(), cmap='terrain', origin='lower', vmin=1480, vmax=3000)
    plt.xlabel('X (mm)', fontsize=14)
    plt.ylabel('Y (mm)', fontsize=14)

    plt.subplot(1, 3, 2)
    plt.title('True Velocity Model')
    # plot ground truth
    plt.imshow(true_vp.squeeze().cpu().numpy(), cmap='terrain', origin='lower', vmin=1480, vmax=3000)
    plt.xlabel('X (mm)', fontsize=14)
    plt.ylabel('Y (mm)', fontsize=14)

    plt.subplot(1, 3, 3)
    plt.title('With Diffusion Velocity Model')
    # plot result: fwi + diffusion
    plt.imshow(diff_vp.squeeze().cpu().numpy(), cmap='terrain', origin='lower', vmin=1480, vmax=3000)
    plt.xlabel('X (mm)', fontsize=14)
    plt.ylabel('Y (mm)', fontsize=14)

    cbar = plt.colorbar()

    # label the colorbar
    cbar.set_label('Velocity (m/s)', fontsize=14)
    cbar.ax.tick_params(labelsize=8)

    plt.suptitle('Comparison of Velocity Models')

    plt.tight_layout()
    plt.show()

def plot_diff(true_dir, inverse_dir, raw_dir, suffix=None, end_frame='00096'):
    """
    Plot the difference between the true velocity model and the inverse velocity model.

    Args:
        true_dir (str): Directory containing the true velocity model.
        inverse_dir (str): Directory containing the inverse velocity model.
    """
    if suffix:
        true_vp = get_vp_from_file(true_dir, file_suffix=suffix, true_model=True, end_frame=end_frame)
    else:
        true_vp = get_vp_from_file(true_dir, true_model=True, end_frame=end_frame)
    inverse_vp = get_vp_from_file(inverse_dir, true_model=False, end_frame=end_frame)
    raw_vp = get_vp_from_file(raw_dir, true_model=False, end_frame=end_frame)
    diff_vp = true_vp - inverse_vp
    raw_diff_vp = true_vp - raw_vp
    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(raw_diff_vp.squeeze().cpu().numpy(), cmap='terrain', origin='lower')
    ax1.set_title('Diff: True vs Raw FWI')

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(diff_vp.squeeze().cpu().numpy(), cmap='terrain', origin='lower')
    ax2.set_title('Diff: True vs FWI+Diffusion')

    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', pad=0.02)

    plt.show()

def visualize_gradient_heatmap(gradient, title="Gradient Heatmap"):
    """
    Visualize the gradient as a heatmap.

    Args:
        gradient: numpy array representing the gradient.
        title: Title of the plot.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(gradient, cmap='terrain', aspect='auto', origin='lower',)
    plt.colorbar(label='Gradient Value')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def plot_loss(all_loss_diff = None, all_loss_fwi = None, raw_fwi = False, freqs=None, iters=None):
    """
    Plot the loss curves for FWI and Diffusion models.

    Args:
        all_loss_diff: List of loss values for the diffusion model.
        all_loss_fwi: List of loss values for the FWI model.
        raw_fwi: Boolean indicating if the FWI is raw (without diffusion).
        freqs: List of max frequency values for vertical lines.
        iters: List of iteration number for vertical lines.
    """
    plt.figure(figsize=(10, 4))
    if raw_fwi:
        plt.plot(all_loss_fwi, color='orange', label='Loss FWI Raw', alpha=0.6)
    else:
        plt.plot(all_loss_diff, color='blue', label='Loss with Diffusion', alpha=0.6)
        plt.plot(all_loss_fwi, color='orange', label='Loss FWI', alpha=0.6)
    # add vertical lines for different frequencies
    plt.axvline(x=iters[0], color='red', linestyle='--', label=f'max_freq = {freqs[0]} MHz')
    plt.axvline(x=iters[1], color='green', linestyle='--', label=f'max_freq = {freqs[1]} MHz')
    plt.axvline(x=iters[2], color='orange', linestyle='--', label=f'max_freq = {freqs[2]} MHz')
    plt.axvline(x=iters[3], color='purple', linestyle='--', label=f'max_freq = {freqs[3]} MHz')
    plt.title('Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def load_grad_sequence(prefix, start_frame, end_frame, file_suffix='.h5'):
    """
    Load a sequence of gradient files into a numpy array.

    Args:
        prefix: Prefix of the gradient file names.
        start_frame: Starting frame number.
        end_frame: Ending frame number.
        file_suffix: Suffix of the gradient file names.
    """
    grad_list = []
    for i in range(start_frame, end_frame + 1):
        filename = f"{prefix}{i:05d}{file_suffix}"
        with h5py.File(filename, 'r') as f:
            grad = f['data'][()]
            grad_list.append(grad)
    return np.array(grad_list)

def load_vp_sequence(prefix, start_frame, end_frame, file_suffix='.h5'):
    """
    Load a sequence of velocity model files into a numpy array.

    Args:
        prefix: Prefix of the velocity model file names.
        start_frame: Starting frame number.
        end_frame: Ending frame number.
        file_suffix: Suffix of the velocity model file names.
    """
    vp_list = []
    for i in range(start_frame, end_frame + 1):
        filename = f"{prefix}{i:05d}{file_suffix}"
        with h5py.File(filename, 'r') as f:
            vp = f['data'][()]
            vp_list.append(vp)
    return np.array(vp_list)

def plot_all_gifs(file_dir, start_frame=1, end_frame=96, file_suffix='.h5'):
    """
    Plot all GIFs for the velocity model and the wavelet with the shots locations.

    Args:
        file_dir: Directory containing the gradient and velocity model files.
        start_frame: Starting frame number.
        end_frame: Ending frame number.
        file_suffix: Suffix of the file names.
    """
    grad_array = load_grad_sequence(f"{file_dir}/anastasio2D-VpGrad-", start_frame, end_frame, file_suffix)
    processed_grad_array = load_grad_sequence(f"{file_dir}/anastasio2D-ProcessedVpGrad-", start_frame, end_frame, file_suffix)
    vp_array = load_vp_sequence(f"{file_dir}/anastasio2D-Vp-", start_frame, end_frame, file_suffix)

    n_frames, ny, nx = grad_array.shape

    vmin_vp = vp_array.min()
    vmax_vp = vp_array.max()
    vmin_grad = grad_array.min()
    vmax_grad = grad_array.max()
    vmin_pro = processed_grad_array.min()
    vmax_pro = processed_grad_array.max()

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))

    im1 = axs[0].imshow(vp_array[0], cmap='terrain', origin='lower',
                        aspect='auto', vmin=vmin_vp, vmax=vmax_vp)
    im2 = axs[1].imshow(grad_array[0], cmap='terrain', origin='lower',
                        aspect='auto', vmin=vmin_grad, vmax=vmax_grad)
    im3 = axs[2].imshow(processed_grad_array[0], cmap='terrain', origin='lower',
                        aspect='auto', vmin=vmin_pro, vmax=vmax_pro)

    cb1 = fig.colorbar(im1, ax=axs[0], label='Velocity (m/s)')
    cb2 = fig.colorbar(im2, ax=axs[1], label='Gradient Value')
    cb3 = fig.colorbar(im3, ax=axs[2], label='Processed Gradient Value')

    axs[0].set_title('Velocity')
    axs[1].set_title('Gradient')
    axs[2].set_title('Processed Gradient')

    for ax in axs:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # update
    def update(frame):
        im1.set_data(vp_array[frame])
        im2.set_data(grad_array[frame])
        im3.set_data(processed_grad_array[frame])

        axs[0].set_title(f'Velocity - Iteration {frame + start_frame}')
        axs[1].set_title(f'Gradient - Iteration {frame + start_frame}')
        axs[2].set_title(f'Processed Gradient - Iteration {frame + start_frame}')
        return [im1, im2, im3]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=300, blit=False)
    # save the animation
    ani.save('gradient_comparison.gif', writer='pillow', fps=3)
    plt.close(fig)

    # show the animation in Jupyter Notebook
    display(Image(filename='gradient_comparison.gif'))

def plot_observation(acquisition):
    """
    Plot the observation data from an acquisition HDF5 file.

    Args:
        acquisition: Path to the acquisition HDF5 file.
    """

    with h5py.File(acquisition, 'r') as f:
        shot = f['shots']['0']
        data = shot['observed']['data'][()]
    # clip data to increase contrast to see the waveform
    data_clipped = np.clip(data, -0.1, 0.1)
    n_receivers, n_timesteps = data_clipped.shape
    dt = 0.08
    t_max = n_timesteps * dt
    plt.figure(figsize=(8, 6))
    # Transpose the data for correct orientation
    im = plt.imshow(data_clipped.T, aspect='auto', cmap='seismic', extent=[0, n_receivers-1, t_max, 0])
    cbar = plt.colorbar(im, pad=0.04, fraction=0.02)
    cbar.set_label('Amplitude', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel('Receiver index', fontsize=14)
    plt.ylabel('Time (\u03bcs)', fontsize=14)
    plt.title('Data Recorded from One Shot', fontsize=16)
    plt.tight_layout()
    plt.show()