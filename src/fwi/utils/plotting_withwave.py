from stride import *
from stride.utils import wavelets, fetch

import matplotlib.animation as animation
import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display


def plot_vp_and_wavelet(data_dir, max_freqs=None, max_iter=96):
    """
    Plot the velocity model and the wavelet with the shots locations.

    Args:
        data_dir: Directory containing the acquisition and model files.
        max_freqs: String of the maximum frequencies for the inverse process.
        max_iter: Iteration number for the velocity model file.
    """
    acquisition_raw = f"{data_dir}/anastasio2D-Acquisitions.h5"

    with h5py.File(acquisition_raw, 'r') as f:
        source_wavelets = f['shots']['0']['wavelets']['data'][0]
    shape = (320, 256)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = Space(shape=shape,
                    extra=extra,
                    absorbing=absorbing,
                    spacing=spacing)

    start = 0.
    step = 0.08e-6
    num = 2500

    time = Time(start=start,
                step=step,
                num=num)

    # Create problem
    problem = Problem(name='anastasio2D',
                        space=space, time=time)

    # Create medium
    # this is the speed of sound of the region of interest
    # vp contains a Numpy array with the velocity for every point on the grid
    vp = ScalarField(name='vp', grid=problem.grid)
    vp.load(f"{data_dir}/anastasio2D-Vp-000{max_iter}.h5")

    problem.medium.add(vp)

    # Create transducers
    # we generally assume point transducers in simulation, but
    # in reality more complex transducer geometries are used
    problem.transducers.default()

    # Create geometry
    # this will make a ring of 512 transducers around the region of interest
    num_locations = 256
    problem.geometry.default('elliptical', num_locations, radius=((space.limit[0] - 7.e-3) / 2, (space.limit[1] - 5.e-3) / 2))

    # Create acquisitions
    # this will link every transducer in the region of interest with each other:
    # they will each fire a pulse in turns, and then record the data for every pulse
    # we call each of these turns a shot
    problem.acquisitions.default()

    vmin = vp.data.min()
    vmax = vp.data.max()
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), 
                            gridspec_kw={'height_ratios': [2, 1]})
    axs[0, 1].axis('off')
    # Create a meshgrid for the x and y coordinates
    x = np.arange(problem.space.shape[1]) * problem.space.spacing[1] * 1e3  # Convert to mm
    y = np.arange(problem.space.shape[0]) * problem.space.spacing[0] * 1e3  # Convert to mm

    im = axs[0, 0].imshow(
        vp.data,
        cmap='terrain',
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto',
        vmin=1480,
        vmax=3000,
        # norm=PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)
    )
    trans_coords = np.array([c for c in problem.geometry.coordinates])
    # Convert coordinates to mm
    # Change the order of coordinates to match (y, x) for plotting
    x_coords = trans_coords[:, 1] * 1e3 
    y_coords = trans_coords[:, 0] * 1e3

    # Plot all transducers
    transducer_label = axs[0, 0].scatter(x_coords, y_coords, color='orange', s=10, label='Receivers')

    shot_ids = np.load(f'{data_dir}/shotids-iter-{max_iter}.npy')
    for idx, id in enumerate(shot_ids):
        source_coord = problem.geometry.coordinates[id]
        source_x = source_coord[1] * 1e3
        source_y = source_coord[0] * 1e3
        if idx == 0:
            source_label = axs[0, 0].scatter(source_x, source_y, color='red', s=50, marker='*', label='Sources')
        else:
            axs[0, 0].scatter(source_x, source_y, color='red', s=50, marker='*')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axs[0, 0])
    cbar.set_label('Velocity (m/s)')

    # Labels and title
    axs[0, 0].set_xlabel(f'X (mm)')
    axs[0, 0].set_ylabel(f'Y (mm)')
    axs[0, 0].set_title(f'Velocity Model (Vp) f_center = 0.25 MHz \n max_freqs = {max_freqs} MHz')
    # Optional grid
    # axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend(handles=[source_label, transducer_label], loc='lower left', bbox_to_anchor=(1, -0.2))

    # wavelet
    time_axis = np.arange(source_wavelets.size) * 0.08e-6 * 1e6  # Convert to microseconds
    axs[1, 0].plot(time_axis, source_wavelets, color='black')
    axs[1, 0].set_title('Source Wavelet', fontsize=16)
    axs[1, 0].set_xlabel('Time (μs)', fontsize=14)
    axs[1, 0].set_ylabel('Amplitude', fontsize=14)
    axs[1, 0].tick_params(axis='both', labelsize=12)
    axs[1, 0].set_xlim(0, 100)  # Limit x-axis to 100 microseconds
    # Wavelet Spectrum
    spectrum_clean = np.abs(np.fft.rfft(source_wavelets))

    # add noise to the wavelet
    # noise = 0.01*np.random.normal(loc=0.0, scale=0.9, size=source_wavelets.shape)
    # noisy_wavelets = source_wavelets + noise
    # spectrum_noisy = np.abs(np.fft.rfft(noisy_wavelets))

    freqs = np.fft.rfftfreq(source_wavelets.size, d=problem.time.step)

    axs[1, 1].plot(freqs * 1e-6, spectrum_clean, color='black')
    # axs[1, 1].plot(freqs * 1e-6, spectrum_noisy, color='orange', linestyle='--', label='Noisy')
    axs[1, 1].set_title('Wavelet Spectrum', fontsize=16)
    axs[1, 1].set_xlabel('Frequency (MHz)', fontsize=14)
    axs[1, 1].set_ylabel('Magnitude', fontsize=14)
    axs[1, 1].tick_params(axis='both', labelsize=12)
    axs[1, 1].set_xlim(0, 0.5)

    
    plt.tight_layout()
    plt.show()

def plot_vp_gif_wavelet(data_dir, max_freqs=None, max_iter=96):
    """
    Plot the gif plot for velocity model and the wavelet with the shots locations.

    Args:
        data_dir: Directory containing the acquisition and model files.
        max_freqs: String of the maximum frequencies for the inverse process.
        max_iter: Iteration number for the velocity model file.
    """
    acquisition_raw = f"{data_dir}/anastasio2D-Acquisitions.h5"

    with h5py.File(acquisition_raw, 'r') as f:
        source_wavelets = f['shots']['0']['wavelets']['data'][0]
    file_prefix = 'anastasio2D-Vp-'
    file_suffix = '.h5'
    start_frame = 1
    end_frame = max_iter

    shape = (320, 256)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = Space(shape=shape,
                    extra=extra,
                    absorbing=absorbing,
                    spacing=spacing)

    start = 0.
    step = 0.08e-6
    num = 2500

    time = Time(start=start,
                step=step,
                num=num)

    # Create problem
    problem = Problem(name='anastasio2D',
                        space=space, time=time)

    # Create transducers
    # we generally assume point transducers in simulation, but
    # in reality more complex transducer geometries are used
    problem.transducers.default()

    # Create geometry
    # this will make a ring of 512 transducers around the region of interest
    num_locations = 256
    problem.geometry.default('elliptical', num_locations, radius=((space.limit[0] - 7.e-3) / 2, (space.limit[1] - 5.e-3) / 2))

    # Create acquisitions
    # this will link every transducer in the region of interest with each other:
    # they will each fire a pulse in turns, and then record the data for every pulse
    # we call each of these turns a shot
    problem.acquisitions.default()

    trans_coords = np.array([c for c in problem.geometry.coordinates])
    # Convert coordinates to mm
    # Change the order of coordinates to match (y, x) for plotting
    x_coords = trans_coords[:, 1] * 1e3 
    y_coords = trans_coords[:, 0] * 1e3

    vp_list = []
    for frame in range(start_frame, end_frame + 1):
        filename = f"{data_dir}/{file_prefix}{frame:05d}{file_suffix}"
        vp = ScalarField(name='vp', grid=problem.grid)
        vp.load(filename)
        vp_list.append(vp.data)

    vp_array = np.array(vp_list)
    vmin = np.min(vp_array)
    vmax = np.max(vp_array)

    x = np.arange(problem.space.shape[1]) * problem.space.spacing[1] * 1e3  # Convert to mm
    y = np.arange(problem.space.shape[0]) * problem.space.spacing[0] * 1e3  # Convert to mm

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), 
                            gridspec_kw={'height_ratios': [2, 1]})
    axs[0, 1].axis('off')
    im = axs[0, 0].imshow(
        vp_list[0],
        cmap='terrain',
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto',
        # norm=PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)
        vmin=1480,
        vmax=3000,
    )
    id_file = f"{data_dir}/shotids-iter-{start_frame}.npy"
    shot_ids = np.load(id_file)
    # Plot all transducers
    axs[0, 0].scatter(x_coords, y_coords, color='yellow', s=10, label='Transducers')

    source_coords = [problem.geometry.coordinates[id] for id in shot_ids]
    source_x = [coord[1] * 1e3 for coord in source_coords]
    source_y = [coord[0] * 1e3 for coord in source_coords]

    source_scatter = axs[0, 0].scatter(source_x, source_y, color='red', s=50, marker='*', label='Sources')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axs[0, 0])
    cbar.set_label('Velocity (m/s)')

    # Labels and title
    axs[0, 0].set_xlabel(f'X (mm)')
    axs[0, 0].set_ylabel(f'Y (mm)')
    title = axs[0, 0].set_title('Velocity Model (Vp) - Iteration 0')

    # Optional grid
    axs[0, 0].grid(alpha=0.3)
    # wavelet
    time_axis = np.arange(source_wavelets.size) * 0.08e-6 * 1e6  # Convert to microseconds
    axs[1, 0].plot(time_axis, source_wavelets, color='black')
    axs[1, 0].set_title('Source Wavelet', fontsize=16)
    axs[1, 0].set_xlabel('Time (μs)', fontsize=14)
    axs[1, 0].set_ylabel('Amplitude', fontsize=14)
    axs[1, 0].tick_params(axis='both', labelsize=12)
    axs[1, 0].grid(True)
    # Wavelet Spectrum
    spectrum = np.abs(np.fft.rfft(source_wavelets))
    freqs = np.fft.rfftfreq(source_wavelets.size, d=problem.time.step)
    axs[1, 1].plot(freqs * 1e-6, spectrum, color='blue')
    axs[1, 1].set_title('Wavelet Spectrum', fontsize=16)
    axs[1, 1].set_xlabel('Frequency (MHz)', fontsize=14)
    axs[1, 1].set_ylabel('Magnitude', fontsize=14)
    axs[1, 1].tick_params(axis='both', labelsize=12)
    axs[1, 1].grid(True)
    axs[1, 1].set_xlim(0, 0.5)

    plt.tight_layout()

    def update(frame):
        im.set_data(vp_array[frame])
        title.set_text(f'Velocity Model (Vp) - Iteration {frame}')
        if frame > 0:
            # Load the shot IDs for the current frame
            id_file = f"{data_dir}/shotids-iter-{frame}.npy"
            shot_ids = np.load(id_file)

            source_coords = [problem.geometry.coordinates[id] for id in shot_ids]
            source_x = [coord[1] * 1e3 for coord in source_coords]
            source_y = [coord[0] * 1e3 for coord in source_coords]

            source_scatter.set_offsets(np.column_stack((source_x, source_y)))
        return [im, title, source_scatter]

    ani = animation.FuncAnimation(fig, update, frames=vp_array.shape[0], interval=300, blit=False)
    ani.save('velocity_model.gif', writer='pillow', fps=3)
    plt.close(fig)

    # Display the animation in Jupyter Notebook

    display(Image(filename='velocity_model.gif'))