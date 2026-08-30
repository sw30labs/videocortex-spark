"""Turn a (n_timesteps x n_vertices) prediction matrix into brain-map stills.

Built on nilearn alone — no torch, no VTK. That keeps two promises: you can
re-render a saved prediction on a machine that never had the model installed,
and the import graph stays small enough that the renderer is testable in CI.

The opinion baked in here: colour limits are computed **once over the whole
run**, never per frame. Per-frame normalisation makes a resting moment look
exactly as vivid as a startling one, which is the easiest way to mislead with
a brain map.
"""

from __future__ import annotations

import io
import logging
import typing as tp
from functools import lru_cache
from pathlib import Path

import numpy as np

from videocortex_spark.config import (
    FSAVERAGE5_VERTICES,
    FSAVERAGE5_VERTICES_PER_HEMI,
    RenderConfig,
)

logger = logging.getLogger(__name__)

_BG_ON_DARK = "#101014"
_FG_ON_DARK = "#e8e8ee"
_BG_ON_LIGHT = "#ffffff"
_FG_ON_LIGHT = "#1a1a1a"

#: How far to zoom into each 3-D axes to squeeze out matplotlib's dead margin.
_AXES_ZOOM = 1.65
#: Width allotted per surface panel, inches.
_PANEL_W_IN = 2.6
#: Downsample factor for the copies kept around to build the contact sheet.
_TILE_DECIMATE = 3
#: Horizontal gutter between contact-sheet columns, as a fraction of cell width.
_SHEET_GUTTER = 0.018


class MeshMismatch(ValueError):
    """Prediction vector doesn't match the fsaverage5 vertex count."""


class RenderOutput(tp.NamedTuple):
    frames: list[Path]
    contact_sheet: Path | None
    vmax: float
    threshold: float


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_fsaverage5() -> dict[str, tp.Any]:
    """Load (once, then cache) the fsaverage5 surfaces and sulcal maps.

    nilearn bundles fsaverage5 as package data, so unlike the higher-resolution
    meshes this never touches the network. Cached anyway because the gifti
    parse is not free when you're drawing sixty frames.
    """
    from nilearn import datasets

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    return {
        "left": {"mesh": fs["infl_left"], "bg": fs["sulc_left"]},
        "right": {"mesh": fs["infl_right"], "bg": fs["sulc_right"]},
    }


def split_hemispheres(values: np.ndarray) -> dict[str, np.ndarray]:
    """Split a flat vertex vector into left/right, fsaverage5 ordering.

    TRIBE concatenates left then right, which is also what nilearn expects
    once you hand it one hemisphere at a time.
    """
    values = np.asarray(values).squeeze()
    if values.ndim != 1:
        raise MeshMismatch(f"expected a 1-D vertex vector, got shape {values.shape}")
    if values.shape[0] != FSAVERAGE5_VERTICES:
        raise MeshMismatch(
            f"expected {FSAVERAGE5_VERTICES} fsaverage5 vertices "
            f"(2 x {FSAVERAGE5_VERTICES_PER_HEMI}), got {values.shape[0]}. "
            "Is this really a TRIBE v2 cortical prediction?"
        )
    return {
        "left": values[:FSAVERAGE5_VERTICES_PER_HEMI],
        "right": values[FSAVERAGE5_VERTICES_PER_HEMI:],
    }


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------


def compute_limits(
    preds: np.ndarray, percentile: float = 99.0, threshold_frac: float = 0.25
) -> tuple[float, float]:
    """Symmetric colour limit and threshold, computed across the whole run.

    Returns ``(vmax, threshold)``. Uses a robust percentile of |x| so one
    berserk vertex can't flatten every other frame.
    """
    arr = np.asarray(preds)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("prediction array contains no finite values")
    vmax = float(np.percentile(np.abs(finite), percentile))
    if vmax <= 0:
        vmax = float(np.abs(finite).max()) or 1.0
    return vmax, vmax * threshold_frac


def select_frames(n_timesteps: int, stride: int, max_frames: int) -> list[int]:
    """Pick which timepoints to draw, honouring stride then the hard cap.

    If stride alone still overshoots ``max_frames``, widen it rather than
    truncating — you want the whole clip sampled evenly, not its first minute.
    """
    if n_timesteps <= 0:
        return []
    stride = max(1, int(stride))
    idx = list(range(0, n_timesteps, stride))
    if max_frames and len(idx) > max_frames:
        widened = max(1, int(np.ceil(n_timesteps / max_frames)))
        idx = list(range(0, n_timesteps, widened))[:max_frames]
    return idx


def auto_columns(n_tiles: int, tile_w: float, tile_h: float) -> int:
    """Choose a column count that lands the sheet near 16:9.

    A four-panel tile is roughly 4:1, so the obvious "6 across" produces a
    thirty-inch-wide strip nobody can read.
    """
    if n_tiles <= 1:
        return 1
    best, best_err = 1, float("inf")
    for cols in range(1, n_tiles + 1):
        rows = int(np.ceil(n_tiles / cols))
        aspect = (cols * tile_w) / (rows * tile_h)
        err = abs(np.log(aspect / (16 / 9)))
        if err < best_err:
            best, best_err = cols, err
    return best


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def soft_threshold_cmap(
    cmap: str, *, vmax: float, threshold: float, ramp_frac: float
):
    """Colormap whose alpha ramps 0→1 across [thr·(1-ramp), thr].

    Nilearn draws with a hard ``threshold=``: a blob pops on at one grey level
    and off at the next, which strobes during a spin. Pushing the ramp into
    the colormap's alpha channel (and passing ``threshold=None``) fades blobs
    in over a band instead of snapping, at zero extra cost.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from nilearn.plotting.cm import cold_hot as _nilearn_cold_hot

    if cmap == "cold_hot":
        base = _nilearn_cold_hot
    else:
        import matplotlib.pyplot as plt

        base = plt.get_cmap(cmap)
    n = 512
    xs = np.linspace(-1.0, 1.0, n)
    rgba = np.asarray(base((xs + 1.0) / 2.0), dtype=np.float64).copy()
    lo = abs(threshold) * (1.0 - max(0.0, min(1.0, ramp_frac)))
    hi = abs(threshold)
    if hi > 0 and ramp_frac > 0:
        a = np.clip((np.abs(xs) * vmax - lo) / (hi - lo), 0.0, 1.0)
    else:
        a = (np.abs(xs) * vmax >= hi).astype(float)
    rgba[..., 3] = a
    return LinearSegmentedColormap.from_list(f"{cmap}-softramp", rgba, N=n)


def energy_curve(preds: np.ndarray) -> np.ndarray:
    """Mean |signal| per TR — the 'is the brain busy right now' trace."""
    arr = np.asarray(preds, dtype=np.float32)
    return np.abs(arr).mean(axis=1).astype(np.float32)


def draw_energy_ribbon(
    energy: np.ndarray,
    *,
    width_px: int,
    height_px: int,
    darkbg: bool = True,
) -> np.ndarray:
    """RGBA strip: the whole-run energy curve, log-ish compressed.

    One shared scale is the house rule, but a 10x peak-to-trough swing puts
    quiet moments on the axis line. A sqrt on the normalised curve keeps the
    honesty (monotone, shared scale) and keeps the trace readable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = np.asarray(energy, dtype=np.float32)
    peak = float(e.max()) if e.size else 1.0
    y = np.sqrt(np.clip(e / (peak or 1.0), 0.0, 1.0))
    fig = plt.figure(figsize=(width_px / 100.0, height_px / 100.0), dpi=100)
    fig.patch.set_facecolor("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("none")
    ax.plot(y, color="#2ee66b", lw=1.4, solid_capstyle="round")
    ax.fill_between(np.arange(y.size), y, color="#2ee66b", alpha=0.22)
    ax.set_xlim(-0.5, y.size - 0.5)
    ax.set_ylim(0, 1.05)
    for s in ax.spines.values():
        s.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, transparent=True)
    plt.close(fig)
    buf.seek(0)
    import matplotlib.image as mpimg

    img = mpimg.imread(buf)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def blit_ribbon(
    frame: np.ndarray,
    ribbon: np.ndarray,
    *,
    playhead: float | None,
    margin_px: int = 10,
    height_px: int = 26,
    bottom_px: int = 30,
) -> np.ndarray:
    """Alpha-composite the energy ribbon near the bottom of ``frame``.

    ``playhead`` is 0..1 across the clip; drawn as a bright vertical with a
    glow dot on the curve. Doubles as a progress bar — the PIP previously
    showed neither.
    """
    h, w = frame.shape[:2]
    if ribbon is None or h < 60:
        return frame
    target_w = w - 2 * margin_px
    if target_w < 40:
        return frame
    try:
        from PIL import Image
    except ImportError:
        return frame

    rib = Image.fromarray(ribbon).resize((target_w, height_px), Image.LANCZOS)
    rib = np.asarray(rib)
    y1 = h - bottom_px
    y0 = y1 - height_px
    frame = frame.copy()
    region = frame[y0:y1, margin_px : margin_px + target_w]
    a = rib[..., 3:4].astype(np.float32) / 255.0 * 0.9
    region[..., :3] = (
        (rib[..., :3].astype(np.float32) * a
         + region[..., :3].astype(np.float32) * (1.0 - a)).astype(np.uint8)
    )
    region[..., 3:] = np.maximum(
        region[..., 3:], (a * 255).astype(np.uint8)
    )
    if playhead is not None:
        x = margin_px + int(np.clip(playhead, 0.0, 1.0) * (target_w - 1))
        frame[y0 - 2 : y1 + 2, x, :3] = 235
        frame[y0 - 2 : y1 + 2, x, 3] = 235
    return frame


def _autocrop(img: np.ndarray, bg_rgb: tuple[float, float, float], tol: float = 0.02):
    """Trim uniform-background rows and columns from a rendered figure.

    matplotlib's 3-D axes reserve a cube of empty space around the mesh, and
    ``bbox_inches="tight"`` cannot crop whitespace that lives *inside* an axes.
    So we render the panels alone, crop the actual ink, and compose the plate
    ourselves.
    """
    rgb = img[..., :3].astype(np.float32)
    if rgb.max() > 1.5:  # 0-255 PNG
        rgb = rgb / 255.0
    ink = np.abs(rgb - np.asarray(bg_rgb, dtype=np.float32)).max(axis=-1) > tol
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return img
    pad = 2
    return img[
        max(rows[0] - pad, 0) : min(rows[-1] + 1 + pad, img.shape[0]),
        max(cols[0] - pad, 0) : min(cols[-1] + 1 + pad, img.shape[1]),
    ]


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("no views")
    if n <= 2:
        return 1, n
    if n <= 4:
        return 2, 2
    return 2, int(np.ceil(n / 2))


def _render_panels(values, cfg: RenderConfig, *, vmax, threshold, bg) -> np.ndarray:
    """Draw just the surface panels; return them as a tightly cropped array."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from nilearn import plotting

    fs = load_fsaverage5()
    hemis = split_hemispheres(values)
    pairs = cfg.view_pairs()
    ramped = cfg.ramp_frac > 0
    cmap_obj = (
        soft_threshold_cmap(cfg.cmap, vmax=vmax, threshold=threshold,
                            ramp_frac=cfg.ramp_frac)
        if ramped else cfg.cmap
    )

    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(_PANEL_W_IN * len(pairs), 2.6),
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": -0.02},
    )
    axes = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor(bg)

    for ax, (hemi, view) in zip(axes, pairs):
        ax.set_facecolor(bg)
        ax.set_box_aspect(None, zoom=_AXES_ZOOM)
        plotting.plot_surf_stat_map(
            surf_mesh=fs[hemi]["mesh"],
            stat_map=hemis[hemi],
            bg_map=fs[hemi]["bg"],
            hemi=hemi,
            view=view,
            cmap=cmap_obj,
            threshold=None if ramped else threshold,
            vmax=vmax,
            colorbar=False,
            bg_on_data=True,
            axes=ax,
            figure=fig,
        )
        # No matplotlib title: at this zoom it either collides with the mesh or
        # gets trimmed by the crop. View labels go on the composed plate.

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=cfg.dpi, facecolor=bg, format="png")
    plt.close(fig)
    buf.seek(0)
    return _autocrop(mpimg.imread(buf), to_rgb(bg))


def _colorbar(fig, rect, cfg, *, vmax, threshold, fg, label_size=7):
    import matplotlib as mpl

    cax = fig.add_axes(rect)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cfg.cmap
        ),
        cax=cax,
        orientation="horizontal",
    )
    cb.set_label(
        f"predicted BOLD (a.u.)  ·  sulcal grey = |x| < {threshold:.2g}",
        color=fg,
        fontsize=label_size,
        labelpad=3,
    )
    cb.ax.tick_params(colors=fg, labelsize=label_size - 1, length=2, pad=1)
    cb.outline.set_edgecolor(fg)
    cb.outline.set_linewidth(0.5)
    return cb


def _compose_plate(
    panels: np.ndarray,
    out_path: Path,
    cfg: RenderConfig,
    *,
    vmax: float,
    threshold: float,
    title: str | None,
    bg: str,
    fg: str,
) -> Path:
    """Label strip + cropped panels + one shared colourbar."""
    import matplotlib.pyplot as plt

    pairs = cfg.view_pairs()
    ph, pw = panels.shape[:2]
    width = _PANEL_W_IN * len(pairs)
    panel_h = width * ph / pw
    top_h, bar_h = 0.32, 0.72
    total_h = top_h + panel_h + bar_h

    fig = plt.figure(figsize=(width, total_h))
    fig.patch.set_facecolor(bg)

    ax = fig.add_axes([0, bar_h / total_h, 1, panel_h / total_h])
    ax.imshow(panels, interpolation="lanczos")
    ax.axis("off")

    label_y = (bar_h + panel_h + 0.07) / total_h
    for i, (hemi, view) in enumerate(pairs):
        fig.text(
            (i + 0.5) / len(pairs), label_y, f"{hemi[0].upper()} {view}",
            color=fg, fontsize=8.5, ha="center", va="bottom",
        )
    if title:
        fig.text(0.012, label_y, title, color=fg, fontsize=10.5, ha="left", va="bottom")

    _colorbar(
        fig,
        [0.36, (0.46 * bar_h) / total_h, 0.28, 0.18 * bar_h / total_h],
        cfg, vmax=vmax, threshold=threshold, fg=fg,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi, facecolor=bg)
    plt.close(fig)
    return out_path


def rounded_rect_alpha(h: int, w: int, radius: float) -> np.ndarray:
    """Antialiased alpha mask: 1 inside a rounded rect, 0 outside."""
    r = max(1.0, float(radius))
    yy, xx = np.ogrid[:h, :w]
    cy0, cy1 = r - 0.5, h - r - 0.5
    cx0, cx1 = r - 0.5, w - r - 0.5
    # Interior (including the straight sides) starts as opaque.
    alpha = np.ones((h, w), dtype=np.float32)

    def _corner(mask, cy, cx):
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        alpha[mask] = np.clip(r + 0.65 - d[mask], 0.0, 1.0)

    _corner((yy < r) & (xx < r), cy0, cx0)
    _corner((yy < r) & (xx >= w - r), cy0, cx1)
    _corner((yy >= h - r) & (xx < r), cy1, cx0)
    _corner((yy >= h - r) & (xx >= w - r), cy1, cx1)
    return alpha


# Phosphor / PACS-monitor green. Sits between the film and the brain so the
# globe reads on a bright cartoon as well as on a dark plate.
_MONITOR_GREEN = np.array([0.18, 0.95, 0.42], dtype=np.float32)
_MONITOR_BLACK = np.array([0.03, 0.04, 0.04], dtype=np.float32)


def apply_monitor_frame(
    rgba: np.ndarray,
    *,
    radius_frac: float = 0.07,
    border_px: float = 5.0,
) -> np.ndarray:
    """Black rounded plate + green bezel, then the brain alpha-over it.

    ``rgba`` is uint8 or float. Corners stay transparent so the bezel is the
    only hard edge on the film.
    """
    img = np.asarray(rgba)
    uint8 = img.dtype == np.uint8
    fg = img.astype(np.float32)
    if uint8:
        fg = fg / 255.0
    if fg.shape[-1] == 3:
        fg = np.concatenate([fg, np.ones(fg.shape[:2] + (1,), np.float32)], axis=-1)
    h, w = fg.shape[:2]
    inset = max(2, int(round(border_px)))
    r_out = radius_frac * min(h, w)
    outer = rounded_rect_alpha(h, w, r_out)
    ih, iw = h - 2 * inset, w - 2 * inset
    inner = np.zeros((h, w), dtype=np.float32)
    if ih > 8 and iw > 8:
        inner[inset : inset + ih, inset : inset + iw] = rounded_rect_alpha(
            ih, iw, max(1.0, r_out - inset)
        )
    fill = inner
    bezel = np.clip(outer - inner, 0.0, 1.0)

    plate = np.zeros((h, w, 4), dtype=np.float32)
    plate[..., :3] = _MONITOR_BLACK
    plate[..., 3] = fill
    # Green bezel on top of the plate edge.
    for c in range(3):
        plate[..., c] = plate[..., c] * (1.0 - bezel) + _MONITOR_GREEN[c] * bezel
    plate[..., 3] = np.maximum(plate[..., 3], bezel)

    a = fg[..., 3:4]
    plate[..., :3] = fg[..., :3] * a + plate[..., :3] * (1.0 - a)
    plate[..., 3:4] = a + plate[..., 3:4] * (1.0 - a)
    plate = np.clip(plate, 0.0, 1.0)
    if uint8:
        return (plate * 255.0 + 0.5).astype(np.uint8)
    return plate


def format_pip_label(kind: str, index: int, timestamp: float | None) -> str | None:
    if kind == "none":
        return None
    if kind == "tr":
        return f"TR {index}"
    t = 0.0 if timestamp is None else max(0.0, float(timestamp))
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def _render_pip_grid(values, cfg: RenderConfig, *, vmax, threshold, bg) -> np.ndarray:
    """2×2 (or 1×N) surface grid, no colourbar, no titles — a corner card."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from nilearn import plotting

    fs = load_fsaverage5()
    hemis = split_hemispheres(values)
    pairs = cfg.view_pairs()
    nrows, ncols = _grid_shape(len(pairs))
    ramped = cfg.ramp_frac > 0
    cmap_obj = (
        soft_threshold_cmap(cfg.cmap, vmax=vmax, threshold=threshold,
                            ramp_frac=cfg.ramp_frac)
        if ramped else cfg.cmap
    )
    cell = 2.15
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(cell * ncols, cell * nrows * 0.92),
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": -0.08, "hspace": -0.06},
    )
    axes = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor(bg)
    for i, ax in enumerate(axes):
        ax.set_facecolor(bg)
        if i >= len(pairs):
            ax.set_axis_off()
            continue
        hemi, view = pairs[i]
        ax.set_box_aspect(None, zoom=_AXES_ZOOM)
        plotting.plot_surf_stat_map(
            surf_mesh=fs[hemi]["mesh"],
            stat_map=hemis[hemi],
            bg_map=fs[hemi]["bg"],
            hemi=hemi,
            view=view,
            cmap=cmap_obj,
            threshold=None if ramped else threshold,
            vmax=vmax,
            colorbar=False,
            bg_on_data=True,
            axes=ax,
            figure=fig,
        )
        ax.text2D(
            0.04 if hemi == "left" else 0.96,
            0.96,
            hemi[0].upper(),
            transform=ax.transAxes,
            color="#c8c8d0",
            fontsize=8,
            ha="left" if hemi == "left" else "right",
            va="top",
            alpha=0.7,
        )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=cfg.dpi, facecolor=bg, format="png")
    plt.close(fig)
    buf.seek(0)
    return _autocrop(mpimg.imread(buf), to_rgb(bg))


def render_pip_card(
    values: np.ndarray,
    out_path: Path,
    cfg: RenderConfig,
    *,
    vmax: float,
    threshold: float,
    label: str | None = None,
    corner_radius_frac: float = 0.055,
) -> Path:
    """One TR as a rounded RGBA card for the PIP overlay.

    Colour limits are passed in — never computed per card — so a quiet second
    stays quieter than a loud one once this is composited onto film.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = _BG_ON_DARK if cfg.darkbg else _BG_ON_LIGHT
    fg = _FG_ON_DARK if cfg.darkbg else _FG_ON_LIGHT
    grid = _render_pip_grid(values, cfg, vmax=vmax, threshold=threshold, bg=bg)
    gh, gw = grid.shape[:2]
    footer = 0.32
    width_in = 4.6
    grid_h_in = width_in * gh / gw
    total_h = grid_h_in + footer

    fig = plt.figure(figsize=(width_in, total_h))
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.03, footer / total_h, 0.94, grid_h_in / total_h])
    ax.imshow(grid, interpolation="lanczos")
    ax.axis("off")
    if label:
        fig.text(
            0.06, 0.045 * footer / max(footer, 0.2) + 0.02,
            label, color=fg, fontsize=9, ha="left", va="center", alpha=0.85,
        )

    buf = io.BytesIO()
    fig.savefig(buf, dpi=cfg.dpi, facecolor=bg, format="png")
    plt.close(fig)
    buf.seek(0)

    import matplotlib.image as mpimg

    img = mpimg.imread(buf)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.shape[-1] == 3:
        rgba = np.concatenate([img, np.ones(img.shape[:2] + (1,), dtype=img.dtype)], axis=-1)
    else:
        rgba = img.copy()
        rgba[..., :3] = np.clip(rgba[..., :3], 0, 1)

    h, w = rgba.shape[:2]
    mask = rounded_rect_alpha(h, w, corner_radius_frac * w)
    rgba[..., 3] = rgba[..., 3] * mask
    rgba = apply_monitor_frame(rgba, radius_frac=corner_radius_frac, border_px=4.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mpimg.imsave writes 0-1 float RGBA as PNG.
    mpimg.imsave(out_path, np.clip(rgba, 0, 1))
    return out_path


def render_pip_series(
    preds: np.ndarray,
    out_dir: Path,
    cfg: RenderConfig,
    *,
    timestamps: tp.Sequence[float] | None = None,
    label_kind: str = "time",
    stride: int = 1,
    progress: tp.Callable[[int, int], None] | None = None,
) -> RenderOutput:
    """Every TR (optionally strided), never widen-stride. Overlay's clock."""
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds[None, :]
    if preds.ndim != 2:
        raise MeshMismatch(
            f"expected a 2-D (timesteps x vertices) array, got {preds.shape}"
        )
    vmax, threshold = compute_limits(preds, cfg.percentile, cfg.threshold_frac)
    logger.info(
        "PIP colour limits from full run: vmax=%.4g threshold=%.4g", vmax, threshold
    )
    stride = max(1, int(stride))
    idx = list(range(0, preds.shape[0], stride))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for n, i in enumerate(idx):
        ts = timestamps[i] if timestamps is not None and i < len(timestamps) else None
        label = format_pip_label(label_kind, i, ts)
        paths.append(
            render_pip_card(
                preds[i],
                out_dir / f"frame_{i:05d}.png",
                cfg,
                vmax=vmax,
                threshold=threshold,
                label=label,
            )
        )
        if progress:
            progress(n + 1, len(idx))
    return RenderOutput(
        frames=paths, contact_sheet=None, vmax=vmax, threshold=threshold
    )


def render_frame(
    values: np.ndarray,
    out_path: Path,
    cfg: RenderConfig,
    *,
    vmax: float,
    threshold: float,
    title: str | None = None,
) -> Path:
    """Render one timepoint as a single PNG containing every requested view."""
    bg = _BG_ON_DARK if cfg.darkbg else _BG_ON_LIGHT
    fg = _FG_ON_DARK if cfg.darkbg else _FG_ON_LIGHT
    panels = _render_panels(values, cfg, vmax=vmax, threshold=threshold, bg=bg)
    return _compose_plate(
        panels, out_path, cfg,
        vmax=vmax, threshold=threshold, title=title, bg=bg, fg=fg,
    )


def build_contact_sheet(
    tiles: tp.Sequence[np.ndarray],
    labels: tp.Sequence[str],
    out_path: Path,
    cfg: RenderConfig,
    *,
    vmax: float,
    threshold: float,
    stims: tp.Mapping[int, np.ndarray] | None = None,
) -> Path | None:
    """Tile the panel renders into one overview — the thing you actually share.

    Composed from the panel arrays rather than by re-reading the frame PNGs,
    so the sheet carries a single colourbar instead of repeating it in every
    cell. When ``stims`` is given, each tile grows a filmstrip frame of the
    stimulus at that moment: stimulus above, brain below, in reading order.
    """
    if not len(tiles):
        return None

    import matplotlib.pyplot as plt

    bg = _BG_ON_DARK if cfg.darkbg else _BG_ON_LIGHT
    fg = _FG_ON_DARK if cfg.darkbg else _FG_ON_LIGHT

    th, tw = tiles[0].shape[:2]
    tile_w = _PANEL_W_IN * len(cfg.view_pairs())
    tile_h = tile_w * th / tw

    label_h, bar_h = 0.26, 0.7
    stims = stims or {}
    # The strip shares the tile width; give it the stimulus' own aspect
    # (capped so a 16:9 grab doesn't dwarf the brain).
    strip_h = 0.0
    if stims:
        any_img = next(iter(stims.values()))
        strip_h = min(0.55 * tile_h, tile_w * any_img.shape[0] / any_img.shape[1])
    # The label strip belongs to the cell, so it counts toward the aspect.
    cols = cfg.contact_sheet_cols or auto_columns(
        len(tiles), tile_w, tile_h + label_h + strip_h
    )
    cols = max(1, min(cols, len(tiles)))
    rows = int(np.ceil(len(tiles) / cols))

    cell_h = tile_h + label_h + strip_h
    grid_h = rows * cell_h
    total_h = grid_h + bar_h
    total_w = cols * tile_w

    fig = plt.figure(figsize=(total_w, total_h))
    fig.patch.set_facecolor(bg)

    for i, (tile, label) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, cols)
        x0 = c / cols
        y0 = (bar_h + (rows - 1 - r) * cell_h) / total_h
        # A gutter between columns: without it two adjacent four-panel tiles
        # read as one eight-panel row.
        gut = _SHEET_GUTTER / cols if cols > 1 else 0.0
        tile_y = y0 + strip_h / total_h
        ax = fig.add_axes([x0 + gut, tile_y, 1 / cols - 2 * gut, tile_h / total_h])
        ax.imshow(tile, interpolation="lanczos")
        ax.axis("off")
        if i in stims and strip_h > 0:
            sx = fig.add_axes([x0 + gut, y0, 1 / cols - 2 * gut, strip_h / total_h])
            sx.imshow(stims[i], interpolation="lanczos")
            sx.axis("off")
            for spine in sx.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(fg)
                spine.set_linewidth(0.5)
                spine.set_alpha(0.35)
        fig.text(
            x0 + gut, y0 + cell_h / total_h + 0.004, label,
            color=fg, fontsize=9, ha="left", va="bottom",
        )

    _colorbar(
        fig, [0.40, (0.42 * bar_h) / total_h, 0.20, 0.16 * bar_h / total_h],
        cfg, vmax=vmax, threshold=threshold, fg=fg, label_size=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=bg)
    plt.close(fig)
    return out_path


def render_series(
    preds: np.ndarray,
    out_dir: Path,
    cfg: RenderConfig,
    *,
    timestamps: tp.Sequence[float] | None = None,
    stimulus_video: Path | None = None,
    progress: tp.Callable[[int, int], None] | None = None,
) -> RenderOutput:
    """Render the selected timepoints of ``preds`` into ``out_dir``.

    ``stimulus_video`` (when given, and ``cfg.filmstrip``) grows a filmstrip
    above every contact-sheet tile. Filmstrip failures never kill a render —
    the brains are the deliverable, the strip is garnish.
    """
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds[None, :]
    if preds.ndim != 2:
        raise MeshMismatch(
            f"expected a 2-D (timesteps x vertices) array, got {preds.shape}"
        )

    vmax, threshold = compute_limits(preds, cfg.percentile, cfg.threshold_frac)
    logger.info(
        "colour limits fixed across run: vmax=%.4g threshold=%.4g", vmax, threshold
    )

    bg = _BG_ON_DARK if cfg.darkbg else _BG_ON_LIGHT
    fg = _FG_ON_DARK if cfg.darkbg else _FG_ON_LIGHT
    idx = select_frames(preds.shape[0], cfg.stride, cfg.max_frames)
    out_dir.mkdir(parents=True, exist_ok=True)

    region_lines: list[str] = [""] * preds.shape[0]
    if cfg.regions:
        try:
            from videocortex_spark.regions import load_region_labels, top_regions_per_tr

            rid, rnames = load_region_labels()
            computed = top_regions_per_tr(preds[idx], rid, rnames, k=3)
            for n, i in enumerate(idx):
                region_lines[i] = computed[n]
        except Exception as exc:  # atlas fetch can fail offline — not fatal
            logger.warning("region labels skipped: %s", exc)

    paths: list[Path] = []
    tiles: list[np.ndarray] = []
    labels: list[str] = []

    for n, i in enumerate(idx):
        if timestamps is not None and i < len(timestamps):
            label = f"t = {timestamps[i]:.1f}s"
        else:
            label = f"TR {i}"
        if region_lines[i]:
            label = f"{label}   {region_lines[i]}"

        # Panels are rendered once and reused for both the plate and the sheet.
        panels = _render_panels(preds[i], cfg, vmax=vmax, threshold=threshold, bg=bg)
        paths.append(
            _compose_plate(
                panels, out_dir / f"frame_{i:05d}.png", cfg,
                vmax=vmax, threshold=threshold, title=label, bg=bg, fg=fg,
            )
        )
        if cfg.contact_sheet:
            tiles.append(panels[::_TILE_DECIMATE, ::_TILE_DECIMATE])
            labels.append(label)
        if progress:
            progress(n + 1, len(idx))

    sheet = None
    if cfg.contact_sheet and tiles:
        stims: dict[int, np.ndarray] = {}
        if cfg.filmstrip and stimulus_video is not None:
            try:
                from videocortex_spark.stimulus import extract_stimulus_frames

                want = [
                    (n, timestamps[i])
                    for n, i in enumerate(idx)
                    if timestamps is not None and i < len(timestamps)
                ]
                got = extract_stimulus_frames(
                    stimulus_video,
                    [t for _, t in want],
                    out_dir.parent / "stim",
                )
                stims = {want[j][0]: img for j, img in got.items()}
            except Exception as exc:  # noqa: BLE001 — garnish must not sink dinner
                logger.warning("filmstrip skipped: %s", exc)
        sheet = build_contact_sheet(
            tiles, labels, out_dir.parent / "contact_sheet.png", cfg,
            vmax=vmax, threshold=threshold, stims=stims,
        )

    return RenderOutput(
        frames=paths, contact_sheet=sheet, vmax=vmax, threshold=threshold
    )
