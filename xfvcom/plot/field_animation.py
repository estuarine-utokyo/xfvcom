"""Frame-by-frame animation of ``plot_field_panels`` maps (mp4 / gif).

Shared by the TB-FVCOM dye (``dye/analysis/dye_anim.py``) and ERSEM
(``ersem/analysis/ersem_anim.py``) pipelines. It owns ONLY the render fan-out and
the video assembly -- exactly as ``plot_field_panels`` owns the static map
render: one ``plot_field_panels`` figure is rendered per time record IN PARALLEL
(joblib), then the numbered PNG frames are assembled into an mp4 (ffmpeg) and/or
gif (PIL) and the scratch frames removed.

The per-frame field assembly (reading NCs, weighting, ratios), the time axis and
the product/layer logic stay app-side; they pass in the finished
``fields[panel, frame, node]`` array plus the per-panel and per-frame label
pieces. Every panel's info-text is ``join(panel_prefixes[p], frame_stamps[i],
panel_suffix)`` so both callers' label conventions are expressible:

- dye grid: prefix = source name, stamp = ``"YYYY-MM-DD\\nHH:MM JST"``,
  suffix = ``"Surface (k=1)"``.
- ERSEM: prefix = ``"<case>  <var>\\n<Layer>"``, stamp = ``"YYYY-MM-DD"``,
  suffix = ``None``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from joblib import Parallel, delayed

from .panel_grid import plot_field_panels


def _panel_label(prefix: str, stamp: str, suffix: str | None) -> str:
    parts = [prefix, stamp] + ([suffix] if suffix else [])
    return "\n".join(p for p in parts if p)


def _render_anim_frame(
    i,
    frames_dir,
    fields,
    panel_prefixes,
    stamp,
    panel_suffix,
    markers,
    layout,
    lon,
    lat,
    nv,
    vmin,
    vmax,
    log_scale,
    cmap,
    cbar_label,
    boundary_paths,
    n_levels,
    cbar_ticks,
    cbar_ticklabels,
    info_fontsize,
    dpi,
):
    """Render one animation frame (top-level for joblib). Fixed canvas
    (``bbox_inches=None``) so every frame shares pixel dimensions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        _panel_label(panel_prefixes[p], stamp, panel_suffix)
        for p in range(fields.shape[0])
    ]
    res = plot_field_panels(
        [fields[p, i] for p in range(fields.shape[0])],
        lon,
        lat,
        nv,
        labels=labels,
        marker_nodes=markers,
        layout=layout,
        vmin=vmin,
        vmax=vmax,
        log_scale=log_scale,
        cmap=cmap,
        cbar_label=cbar_label,
        land="mesh",
        boundary_paths=boundary_paths,
        info_fontsize=info_fontsize,
        dpi=dpi,
        bbox_inches=None,
        n_levels=n_levels,
        cbar_ticks=cbar_ticks,
        cbar_ticklabels=cbar_ticklabels,
        output=str(Path(frames_dir) / f"{i:05d}.png"),
    )
    plt.close(res["fig"])
    return i


def _assemble(
    frames_dir: Path, out_base: Path, fps: int, formats: Sequence[str]
) -> None:
    """Assemble the numbered PNG frames into mp4 (ffmpeg) and/or gif (PIL)."""
    if "mp4" in formats:
        mp4 = out_base.with_suffix(".mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                str(mp4),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"    wrote {mp4.name}", flush=True)
    if "gif" in formats:
        from PIL import Image

        gif = out_base.with_suffix(".gif")
        frames = sorted(frames_dir.glob("*.png"))
        first = Image.open(frames[0]).convert("RGB").quantize(colors=256)
        rest = [
            Image.open(f).convert("RGB").quantize(palette=first) for f in frames[1:]
        ]
        first.save(
            gif,
            save_all=True,
            append_images=rest,
            duration=int(1000 / fps),
            loop=0,
            optimize=False,
        )
        print(f"    wrote {gif.name}", flush=True)


def animate_field_panels(
    out_base: str | Path,
    fields: Any,
    panel_prefixes: Sequence[str],
    frame_stamps: Sequence[str],
    lon: np.ndarray,
    lat: np.ndarray,
    nv: np.ndarray,
    *,
    panel_suffix: str | None = None,
    markers: Sequence[Sequence[int]] | None = None,
    layout: list[int] | None = None,
    vmin: float = 0.1,
    vmax: float = 500.0,
    log_scale: bool = True,
    cmap: str = "turbo",
    cbar_label: str = "",
    boundary_paths: list[list[int]] | None = None,
    n_levels: int | None = None,
    cbar_ticks: Any = None,
    cbar_ticklabels: Any = None,
    info_fontsize: float = 14,
    dpi: int = 96,
    fps: int = 10,
    formats: Sequence[str] = ("mp4",),
    n_jobs: int = 1,
) -> None:
    """Render one ``plot_field_panels`` frame per time record (in parallel) and
    assemble to mp4/gif.

    Parameters
    ----------
    out_base
        Output path WITHOUT a suffix; ``.mp4`` / ``.gif`` is appended per format.
    fields
        ``(npanel, nframes, node)`` -- panel ``p`` at frame ``i`` is
        ``fields[p, i]``.
    panel_prefixes, frame_stamps, panel_suffix
        Per-panel info-text is ``join(panel_prefixes[p], frame_stamps[i],
        panel_suffix)``.
    markers, layout
        Per-panel 1-based marker nodes (default none) and panels-per-row layout
        (default a single row of all panels).
    vmin..n_jobs
        Forwarded to ``plot_field_panels`` (vmin/vmax/log_scale/cmap/cbar_label/
        n_levels/cbar_ticks/cbar_ticklabels/info_fontsize/dpi) and the
        ffmpeg/PIL assembly (fps/formats); ``n_jobs`` is the parallel frame-render
        worker count.
    """
    out_base = Path(out_base)
    fields = np.asarray(fields)
    npanel, nframes = fields.shape[0], fields.shape[1]
    if markers is None:
        markers = [[] for _ in range(npanel)]
    if layout is None:
        layout = [npanel]
    frames_dir = out_base.parent / f"_frames_{out_base.name}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    Parallel(n_jobs=n_jobs, temp_folder=str(frames_dir))(
        delayed(_render_anim_frame)(
            i,
            str(frames_dir),
            fields,
            panel_prefixes,
            frame_stamps[i],
            panel_suffix,
            markers,
            layout,
            lon,
            lat,
            nv,
            vmin,
            vmax,
            log_scale,
            cmap,
            cbar_label,
            boundary_paths,
            n_levels,
            cbar_ticks,
            cbar_ticklabels,
            info_fontsize,
            dpi,
        )
        for i in range(nframes)
    )
    _assemble(frames_dir, out_base, fps, list(formats))
    shutil.rmtree(frames_dir, ignore_errors=True)  # PNG frames + joblib temp
