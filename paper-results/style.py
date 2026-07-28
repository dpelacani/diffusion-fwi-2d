import copy
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# ===============================================
# 1. COLORMAP — Guasch et al. (Nature) RGB table   
# ===============================================

def load_guasch_cmap(path="brain.txt", name="guasch"):
    try:
        rgb = np.loadtxt(path, delimiter=",")
    except OSError:
        if path == "brain.txt":
            fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain.txt")
            rgb = np.loadtxt(fallback, delimiter=",")
        else:
            raise
    assert rgb.shape[1] == 3 and rgb.min() >= 0 and rgb.max() <= 1, \
        "expected an Nx3 RGB table in [0,1]"
    return ListedColormap(rgb, name=name)

GUASCH_CMAP = None

def _require_cmap(cmap):
    if cmap is None:
        raise RuntimeError(
            "No colormap available."
        )
    return cmap

VMIN, VMAX = 1480.0, 3000.0
SKULL_THRESH = 1650.0
MASK_TOL = 5.0
DISPLAY_VMIN = 1400.0

TISSUE_VMIN = 1450.0
TISSUE_VMAX = 1873.0

# ==================
# 2. METHOD PALETTE             
# ==================

METHOD_META = {
    "baseline":              {"label": "FWI only",         "color": "#666666"},
    "postprocessing_merged": {"label": "Postproc. merged", "color": "#88CCEE"},
    "postprocessing_split":  {"label": "Postproc. split",  "color": "#2255AA"},
    "gradient_merged":       {"label": "Grad. merged",     "color": "#EE7733"},
    "gradient_split":        {"label": "Grad. split",      "color": "#CC3311"},
}
METHODS_ORDER = list(METHOD_META.keys())

# =================================================
# 2b. ACQUISITION GEOMETRY
#     setup figures that draw the transducer ring               
# =================================================

# Sampled from the ring highlighted on the 3D head render in Fig 1
RING_COLOR = "#A54B22"

# ==================
# 3. GLOBAL STYLE                                                
# ==================

_LIBERATION_SEARCH_DIRS = [
    "/usr/share/fonts/truetype/liberation",  
    "/usr/share/fonts/liberation",            
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
]


def _register_liberation_serif():
    """Find and register Liberation Serif wherever it actually lives.

    Returns True if at least the Regular weight was registered.
    """
    found_regular = False
    for d in _LIBERATION_SEARCH_DIRS:
        for f in glob.glob(os.path.join(d, "LiberationSerif-*.ttf")):
            try:
                fm.fontManager.addfont(f)
                if "Regular" in os.path.basename(f):
                    found_regular = True
            except (FileNotFoundError, RuntimeError):
                pass
    return found_regular


def apply_paper_style(cmap_path="brain.txt"):
    """Call once at the top of every figure script."""
    global GUASCH_CMAP

    liberation_ok = _register_liberation_serif()
    font_family = (["Liberation Serif"] if liberation_ok else []) + ["serif"]

    sns.set_theme(style="white", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": font_family,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    try:
        GUASCH_CMAP = load_guasch_cmap(cmap_path)
    except OSError as e:
        GUASCH_CMAP = None
        print(f"[style] WARNING: couldn't load the Guasch colormap.")

    return GUASCH_CMAP


# ===============
# 4. COLORBARS
# ===============

def add_colorbar(im, ax, label="velocity (m/s)", ticks=None, size="4%", pad=0.18):
    """
    One colorbar per panel, detached.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = plt.colorbar(im, cax=cax, ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=2)
    cb.set_label(label, fontsize=10, labelpad=4)
    return cb


def add_shared_colorbar(fig, im, axes, label="velocity (m/s)", ticks=None, pad=0.015, width=0.012):
    """
    One colorbar for multiple panels that share the same vmin/vmax.
    """
    fig.canvas.draw()
    pos = [a.get_position() for a in np.ravel(axes)]
    right = max(p.x1 for p in pos)
    top = max(p.y1 for p in pos)
    bottom = min(p.y0 for p in pos)
    cax = fig.add_axes([right + pad, bottom, width, top - bottom])
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label(label, fontsize=8, labelpad=4)
    return cb


# =========================================
# 4b. PANEL LABELS for multi-panel figures          
# =========================================

def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=14, fontweight="bold",
                     ha="left", va="bottom", **text_kwargs):
    """
    Add a bold panel label (A, B, C...) at the top-left of an axes.
    """
    return ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
                   fontweight=fontweight, ha=ha, va=va, **text_kwargs)


def fix_panel_aspect(ax, im, data_aspect, anchor="W", colorbar_label="velocity (m/s)"):
    fig = ax.figure
    fig.canvas.draw()
    fig_w_in, fig_h_in = fig.get_size_inches()

    def _shrink(pos):
        width_in = pos.width * fig_w_in
        height_in = pos.height * fig_h_in
        target_height_in = width_in * data_aspect
        if target_height_in <= height_in:
            nw_in, nh_in = width_in, target_height_in
        else:
            nw_in, nh_in = height_in / data_aspect, height_in
        nw, nh = nw_in / fig_w_in, nh_in / fig_h_in
        if anchor == "W":
            x0, y0 = pos.x0, pos.y0 + (pos.height - nh) / 2
        elif anchor == "SW":
            x0, y0 = pos.x0, pos.y0
        elif anchor == "C":
            x0, y0 = pos.x0 + (pos.width - nw) / 2, pos.y0 + (pos.height - nh) / 2
        else:
            raise ValueError(f"unsupported anchor {anchor!r}")
        return x0, y0, nw, nh

    cb = im.colorbar
    cb.ax.remove()
    ax.set_axes_locator(None)
    x0, y0, nw, nh = _shrink(ax.get_position())
    ax.set_position([x0, y0, nw, nh])
    pos0 = ax.get_position()

    add_colorbar(im, ax, label=colorbar_label)
    fig.canvas.draw()
    pos1 = ax.get_position()
    actual_aspect = (pos1.height * fig_h_in) / (pos1.width * fig_w_in)
    correction = actual_aspect / data_aspect

    im.colorbar.ax.remove()
    ax.set_axes_locator(None)
    new_width = pos0.width * correction
    ax.set_position([pos0.x0, pos0.y0, new_width, pos0.height])
    add_colorbar(im, ax, label=colorbar_label)


def guasch_mask(vp, background_value=VMIN, tol=MASK_TOL):
    """
    Simple flat-threshold mask: True = keep (vp > background_value + tol),
    False = mask out as background. 
    """
    return vp > (background_value + tol)


def masked_brain_imshow(ax, vp, mask=None, vmin=None, vmax=VMAX,
                         cmap=None, mask_color="white", **imshow_kwargs):
    
    cmap = _require_cmap(cmap or GUASCH_CMAP)

    if mask is False:
        vmin = VMIN if vmin is None else vmin
        return ax.imshow(vp, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)

    if mask is None:
        mask = guasch_mask(vp)

    vmin = DISPLAY_VMIN if vmin is None else vmin
    cmap = copy.copy(cmap)  # don't mutate the shared global GUASCH_CMAP
    cmap.set_bad(mask_color)
    masked_vp = np.ma.masked_array(vp, mask=~mask)
    return ax.imshow(masked_vp, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)


# Sampled from the dark navy-blue skull/bone ring in the reference figure
SKULL_COLOR = "#33588D"


def tissue_skull_imshow(ax, vp, mask=None, tissue_vmin=TISSUE_VMIN, tissue_vmax=TISSUE_VMAX,
                         skull_thresh=SKULL_THRESH, skull_color=SKULL_COLOR,
                         cmap=None, mask_color="white", **imshow_kwargs):
    
    cmap = cmap or GUASCH_CMAP
    if mask is None:
        mask = guasch_mask(vp)

    skull = mask & (vp > skull_thresh)
    tissue = mask & ~skull

    cmap = copy.copy(cmap)
    cmap.set_bad((1, 1, 1, 0))   # transparent where not tissue, not mask_color
                                 # background gets its flat color via ax.set_facecolor below
    masked_vp = np.ma.masked_array(vp, mask=~tissue)
    im = ax.imshow(masked_vp, cmap=cmap, vmin=tissue_vmin, vmax=tissue_vmax, **imshow_kwargs)

    overlay = np.zeros((*vp.shape, 4))
    overlay[skull] = plt.matplotlib.colors.to_rgba(skull_color)
    ax.imshow(overlay, extent=imshow_kwargs.get("extent"),
              origin=imshow_kwargs.get("origin", "upper"),
              aspect=imshow_kwargs.get("aspect", "equal"))

    ax.set_facecolor(mask_color)
    return im


def head_mask(vp, background_value=VMIN, tol=5.0, opening_iterations=1):
    """
    Boolean mask for masked_brain_imshow(): True = inside the head/anatomy,
    False = outside (the water-coupling background).
    """
    from scipy import ndimage
    structure = np.ones((3, 3))
    foreground = vp > (background_value + tol)
    foreground = ndimage.binary_opening(
        foreground, structure=structure, iterations=opening_iterations,
    )
    foreground = ndimage.binary_fill_holes(foreground)

    labeled, n_features = ndimage.label(foreground)
    if n_features == 0:
        return np.zeros_like(vp, dtype=bool)
    sizes = ndimage.sum(foreground, labeled, index=range(1, n_features + 1))
    largest_label = int(np.argmax(sizes)) + 1
    return labeled == largest_label


# =============
# 6. HELPERS 
# =============

def save_figure(fig, name, output_dir, close=True, formats=("png",)):
    
    if isinstance(formats, str):
        formats = (formats,)
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for ext in formats:
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
        paths.append(path)
    if close:
        plt.close(fig)
    return paths[0] if len(paths) == 1 else paths
