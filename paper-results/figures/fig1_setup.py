import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import (
    apply_paper_style, masked_brain_imshow, add_colorbar, add_panel_label,
    fix_panel_aspect, RING_COLOR, VMAX, save_figure,
)

import h5py
import numpy as np
import matplotlib.pyplot as plt

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HELMET_IMAGE = os.path.join(PAPER_DIR, "assets", "Head_TransducerPositions.png")


def _find_top_transducer(coords_yx):
    """
    coords_yx : (N, 2) array of (y, x) transducer positions

    Returns the index of the transducer geometrically closest to the top
    of the ring (0 degrees / 12 o'clock), found from coordinates.
    """
    coords_yx = np.asarray(coords_yx)
    center = coords_yx.mean(axis=0)
    dy = coords_yx[:, 0] - center[0]
    dx = coords_yx[:, 1] - center[1]
    angle = np.arctan2(dy, dx)   # 0 = +x (right), +pi/2 = +y ("up", since imshow uses origin='lower')
    target = np.pi / 2
    wrapped_diff = np.angle(np.exp(1j * (angle - target)))  # wrap-safe angular distance
    return int(np.argmin(np.abs(wrapped_diff)))


def _angle_from_top_deg(transducer_xy_mm, idx):
    """
    Angle of transducer `idx`, in degrees clockwise from the top of the
    ring (0 deg = top, matching _find_top_transducer's convention).
    """
    transducer_xy_mm = np.asarray(transducer_xy_mm)
    coords_yx = transducer_xy_mm[:, ::-1]
    center = coords_yx.mean(axis=0)
    dy = coords_yx[idx, 0] - center[0]
    dx = coords_yx[idx, 1] - center[1]
    # angle measured from +y ("up"), clockwise positive
    angle_rad = np.arctan2(dx, dy)
    return round(np.degrees(angle_rad)) % 360


def plot_observation_panel(ax, acquisition_h5_path, shot_id="0", clip=0.1,
                            dt_us=0.08, cmap="seismic", title=None):
    """
    Draw the recorded-data waterfall plot for one shot:
    receiver index on x, time on y, amplitude as color.

    acquisition_h5_path : path to the *-Acquisitions.h5 file (the same
                          file that has the source wavelet).
    shot_id : which shot to show, as the literal key into the h5 file's
              "shots" group
    clip : amplitude clip range (+/-).
    dt_us : time step in microseconds, for the y-axis scale.

    Returns the AxesImage.
    """
    shot_id = str(shot_id)
    with h5py.File(acquisition_h5_path, "r") as f:
        data = f["shots"][shot_id]["observed"]["data"][()]
    data_clipped = np.clip(data, -clip, clip)
    n_receivers, n_timesteps = data_clipped.shape
    t_max = n_timesteps * dt_us

    im = ax.imshow(
        data_clipped.T, aspect="auto", cmap=cmap,
        extent=[0, n_receivers - 1, t_max, 0],
    )
    add_colorbar(im, ax, label="Amplitude")
    ax.set_xlabel("Receiver index", fontsize=14)
    ax.set_ylabel("Time (μs)", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    return im


def plot_acquisition_panel(ax, vp_data, x_mm, y_mm, transducer_xy_mm,
                            active_idx=None, cmap=None, mask=None,
                            vmin=None, vmax=VMAX, mask_color="white",
                            ring_color=RING_COLOR, title=None):
    """
    Draw the velocity model with the transducer ring and the single active
    source highlighted.

    vp_data           : 2D array (rows=y, cols=x), velocity in m/s.
    x_mm, y_mm        : 1D coordinate arrays for the imshow extent, in mm.
    transducer_xy_mm  : (N, 2) array of (x_mm, y_mm) transducer positions.
    active_idx        : index into transducer_xy_mm to highlight as firing.
                        None = auto-detect the top (0 deg) transducer.

    Returns (im, active_idx).
    """
    im = masked_brain_imshow(
        ax, vp_data, mask=mask, vmin=vmin, vmax=vmax, cmap=cmap,
        mask_color=mask_color,
        extent=[x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()],
        origin="lower", aspect="auto",
    )

    transducer_xy_mm = np.asarray(transducer_xy_mm)
    ring_handle = ax.scatter(
        transducer_xy_mm[:, 0], transducer_xy_mm[:, 1],
        facecolors="none", edgecolors=ring_color, linewidths=0.8, s=14,
        label="Transducers",
    )

    if active_idx is None:
        active_idx = _find_top_transducer(transducer_xy_mm[:, ::-1])  # (x,y) -> (y,x)
    src_x, src_y = transducer_xy_mm[active_idx]
    source_handle = ax.scatter(
        [src_x], [src_y], color=ring_color, edgecolors="white",
        linewidths=0.6, s=150, marker="*", zorder=5, label="Active source (0°)",
    )

    add_colorbar(im, ax, label="Velocity (m/s)")
    ax.set_xlabel("X (mm)", fontsize=14)
    ax.set_ylabel("Y (mm)", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    ax.legend(handles=[ring_handle, source_handle], loc="lower left",
              bbox_to_anchor=(1.038, -0.20), frameon=False, fontsize=10)
    return im, active_idx


def build_ring_geometry(shape=(300, 256), extra=(50, 50), absorbing=(40, 40),
                         spacing=(0.5e-3, 0.5e-3), num_locations=256,
                         precomputed_path=None):
    """
    Build just the acquisition geometry (transducer ring positions + the
    physical x/y axes) via Stride, with no velocity data file needed.

    precomputed_path : path to a frozen .npz (x_mm, y_mm, transducer_xy_mm)

    Returns (x_mm, y_mm, transducer_xy_mm).
    """
    if precomputed_path:
        data = np.load(precomputed_path)
        return data["x_mm"], data["y_mm"], data["transducer_xy_mm"]

    from stride import Space, Time, Problem  # lazy: only needed here

    space = Space(shape=shape, extra=extra, absorbing=absorbing, spacing=spacing)
    time = Time(start=0., step=0.08e-6, num=2500)  # arbitrary — Problem just needs *a* Time to construct
    problem = Problem(name="geometry_only", space=space, time=time)
    problem.transducers.default()
    problem.geometry.default(
        "elliptical", num_locations,
        radius=((space.limit[0] - 7.e-3) / 2, (space.limit[1] - 5.e-3) / 2),
    )
    problem.acquisitions.default()

    x_mm = np.arange(problem.space.shape[1]) * problem.space.spacing[1] * 1e3
    y_mm = np.arange(problem.space.shape[0]) * problem.space.spacing[0] * 1e3
    trans_coords = np.array([c for c in problem.geometry.coordinates])  # (y, x), meters
    transducer_xy_mm = np.column_stack([trans_coords[:, 1] * 1e3, trans_coords[:, 0] * 1e3])
    return x_mm, y_mm, transducer_xy_mm


def plot_setup_figure_groundtruth(brain_id, gt_dir, forward_dir, shape=(320, 256),
                                   extra=(50, 50), absorbing=(40, 40),
                                   spacing=(0.5e-3, 0.5e-3), num_locations=256,
                                   shot_id=None, preserve_aspect=True, aspect_anchor="W",
                                   save_dir=None, formats=("pdf", "svg"),
                                   fig_name="fig1_setup_panel", mask_color="white",
                                   helmet_image_path=DEFAULT_HELMET_IMAGE,
                                   helmet_citation="",
                                   helmet_width_in=3.0, helmet_gap_in=0.4,
                                   show=True, precomputed_ring_geometry=None):
    """
    Plot setup panel (A: helmet render, B: velocity model +
    transducer ring + active source, C: recorded shot gather, D: source
    wavelet, E: wavelet spectrum) from a ground-truth velocity .npy file.

    Returns (fig, axs, active_idx).
    """
    apply_paper_style()

    gt_path = f"{gt_dir}/vp_{brain_id}.npy"
    acquisition_raw = f"{forward_dir}/VP_{brain_id}/VP_{brain_id}-Acquisitions.h5"

    vp_data = np.load(gt_path).astype(np.float32)

    expected_shape = tuple(shape)
    if vp_data.shape != expected_shape:
        print(f"[warning] {gt_path} has shape {vp_data.shape}, but the "
              f"acquisition geometry expects {expected_shape}. The "
              "transducer ring is positioned for that grid, so if these "
              "don't match it won't line up with the anatomy — resample "
              "vp_data first, or pass matching shape/spacing.")

    x_mm, y_mm, transducer_xy_mm = build_ring_geometry(
        shape=shape, extra=extra, absorbing=absorbing, spacing=spacing,
        num_locations=num_locations, precomputed_path=precomputed_ring_geometry,
    )

    dt_us = 0.08

    fig, axs = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.5, "wspace": 0.5})

    axs[0, 0].set_box_aspect(320 / 256)

    im_a, active_idx = plot_acquisition_panel(
        axs[0, 0], vp_data, x_mm, y_mm, transducer_xy_mm,
        mask_color=mask_color, title="",
    )

    if shot_id is None:
        shot_id = str(active_idx)
    shot_angle = _angle_from_top_deg(transducer_xy_mm, int(shot_id)) if str(shot_id).isdigit() else None
    panel_b_title = f"Shot {shot_id}" + (f" ({shot_angle:.0f}°)" if shot_angle is not None else "")
    plot_observation_panel(axs[0, 1], acquisition_raw, shot_id=shot_id, dt_us=dt_us,
                            title=panel_b_title)

    with h5py.File(acquisition_raw, "r") as f:
        source_wavelets = f["shots"]["0"]["wavelets"]["data"][0]

    time_axis = np.arange(source_wavelets.size) * dt_us
    axs[1, 0].plot(time_axis, source_wavelets, color="black")
    axs[1, 0].set_title("Source Wavelet", fontsize=16)
    axs[1, 0].set_xlabel("Time (μs)", fontsize=14)
    axs[1, 0].set_ylabel("Amplitude", fontsize=14)
    axs[1, 0].set_xlim(0, 100)

    spectrum = np.abs(np.fft.rfft(source_wavelets))
    freqs = np.fft.rfftfreq(source_wavelets.size, d=dt_us * 1e-6)
    axs[1, 1].plot(freqs * 1e-6, spectrum, color="black")
    axs[1, 1].set_title("Wavelet Spectrum", fontsize=16)
    axs[1, 1].set_xlabel("Frequency (MHz)", fontsize=14)
    axs[1, 1].set_ylabel("Magnitude", fontsize=14)
    axs[1, 1].set_xlim(0, 0.5)

    plt.tight_layout()

    if preserve_aspect:
        data_aspect = (y_mm.max() - y_mm.min()) / (x_mm.max() - x_mm.min())
        fix_panel_aspect(axs[0, 0], im_a, data_aspect, anchor=aspect_anchor)

    fig.canvas.draw()

    pos_b = axs[0, 1].get_position()
    pos_c = axs[1, 0].get_position()
    pos_d = axs[1, 1].get_position()
    pos_a_shrunk = axs[0, 0].get_position()

    if preserve_aspect and pos_a_shrunk.height > 0:
        scale_factor = pos_b.height / pos_a_shrunk.height
        new_width = pos_a_shrunk.width * scale_factor
        new_height = pos_b.height

        axs[0, 0].set_aspect('auto')
        axs[0, 0].set_position([pos_a_shrunk.x0, pos_b.y0, new_width, new_height])
        axs[0, 0].set_axes_locator(None)

    pos_a = axs[0, 0].get_position()

    axs[1, 0].set_position([pos_a.x0, pos_c.y0, pos_c.width, pos_c.height])
    axs[1, 0].set_axes_locator(None)

    extra_axes = [ax for ax in fig.axes if ax not in axs.flat]
    if len(extra_axes) >= 2:
        colorbars = sorted(extra_axes, key=lambda ax: ax.get_position().x0)
        cax_a = colorbars[0]
        cax_b = colorbars[1]

        target_colorbar_width = cax_b.get_position().width

        cax_a.set_position([pos_a.x1 + 0.02, pos_a.y0, target_colorbar_width, pos_a.height])
        cax_a.set_axes_locator(None)

        cax_b.set_position([pos_b.x1 + 0.02, pos_b.y0, target_colorbar_width, pos_b.height])
        cax_b.set_axes_locator(None)

    elif len(extra_axes) == 1:
        cax_a = extra_axes[0]
        pos_cbar_a = cax_a.get_position()
        cax_a.set_position([pos_a.x1 + 0.02, pos_a.y0, pos_cbar_a.width, pos_a.height])
        cax_a.set_axes_locator(None)

    old_width_in, height_in = fig.get_size_inches()
    new_width_in = old_width_in + helmet_width_in + helmet_gap_in

    for ax in fig.axes:
        p = ax.get_position()
        old_x0_in = p.x0 * old_width_in
        old_w_in = p.width * old_width_in
        new_x0 = (old_x0_in + helmet_width_in + helmet_gap_in) / new_width_in
        new_w = old_w_in / new_width_in
        ax.set_position([new_x0, p.y0, new_w, p.height])
        ax.set_axes_locator(None)

    fig.set_size_inches(new_width_in, height_in)

    pos_a = axs[0, 0].get_position()
    pos_b = axs[0, 1].get_position()
    pos_c = axs[1, 0].get_position()
    pos_d = axs[1, 1].get_position()

    e_x0 = 0.0 + (helmet_gap_in * 0.3) / new_width_in
    e_width = helmet_width_in / new_width_in
    ax_e = fig.add_axes([e_x0, pos_a.y0, e_width, pos_a.height])
    if helmet_image_path:
        helmet_img = plt.imread(helmet_image_path)
        ax_e.imshow(helmet_img, aspect='auto')
    ax_e.axis("off")
    ax_e.text(0.5, -0.05, helmet_citation, ha="center", va="top",
              fontsize=14, style="italic", transform=ax_e.transAxes)

    y_row1_baseline = max(pos_a.y1, pos_b.y1) + 0.015
    y_row2_baseline = max(pos_c.y1, pos_d.y1) + 0.015

    fig.text(e_x0 - 0.005, y_row1_baseline, "A", fontsize=18, fontweight="bold", va="bottom", ha="right")
    fig.text(pos_a.x0 - 0.02, y_row1_baseline, "B", fontsize=18, fontweight="bold", va="bottom", ha="right")
    fig.text(pos_b.x0 - 0.02, y_row1_baseline, "C", fontsize=18, fontweight="bold", va="bottom", ha="right")
    fig.text(pos_a.x0 - 0.02, y_row2_baseline, "D", fontsize=18, fontweight="bold", va="bottom", ha="right")
    fig.text(pos_d.x0 - 0.02, y_row2_baseline, "E", fontsize=18, fontweight="bold", va="bottom", ha="right")

    fig.align_ylabels([axs[0, 0], axs[1, 0]])
    fig.align_ylabels([axs[0, 1], axs[1, 1]])

    if save_dir:
        save_figure(fig, fig_name, save_dir, close=False, formats=formats)
    if show:
        plt.show()
    print(f"  transducer index: {active_idx}. Panel C shows "
          f"shot_id={shot_id!r} — by default this is str(active_idx), which "
          f"assumes the acquisition's shot numbering matches the geometry "
          f"ring's ordering. Verify that holds for your data, or pass "
          f"shot_id explicitly if it doesn't.")
    return fig, axs, active_idx


if __name__ == "__main__":
    fig, axs, active_idx = plot_setup_figure_groundtruth(
        brain_id="654754",
        gt_dir="/cluster/scratch/fscharitzer/fwi-data/Ultrasound-Vp-axial-models",
        forward_dir="/cluster/scratch/fscharitzer/exps/forward",
        save_dir=os.path.join(PAPER_DIR, "results", "figures"),
        formats=("pdf", "svg"),
        show=False,
    )
