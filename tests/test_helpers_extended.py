import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from xfvcom.utils.helpers import (
    FrameGenerator,
    _cleanup_files,
    convert_gif_to_mp4,
    create_mp4,
    ensure_dir,
    generate_frames_and_collect,
)


class DummyPlotter:
    """
    Dummy plotter to simulate plot_2d behavior by writing an empty file.
    """

    def plot_2d(self, da=None, save_path=None, **kwargs):
        Path(save_path).write_text("ok")
        return None


def test_ensure_dir(tmp_path):
    # Create directory without cleaning existing contents
    d = tmp_path / "nested" / "dir"
    ensure_dir(d, clean=False)
    assert d.exists() and d.is_dir()
    # Clean existing contents when clean=True
    (d / "foo.txt").write_text("content")
    assert (d / "foo.txt").exists()
    ensure_dir(d, clean=True)
    assert not (d / "foo.txt").exists()


def test_cleanup_files(tmp_path):
    # Create dummy files
    files = []
    for name in ["a.png", "b.png", "c.txt"]:
        p = tmp_path / name
        p.write_text("content")
        files.append(p)
    # Only PNG files should be deleted
    png_files = [p for p in files if p.suffix == ".png"]
    _cleanup_files(png_files)
    for p in png_files:
        assert not p.exists()
    # Non-PNG file should remain
    assert (tmp_path / "c.txt").exists()


def test_generate_frames_and_collect(tmp_path, monkeypatch):
    # Dummy DataArray with a time dimension of size 3
    class DummyDA:
        sizes = {"time": 3}

        def isel(self, **kwargs):
            return self

        def load(self):
            return self

    DummyDA()

    # Mock FrameGenerator.generate_frames to generate and write dummy frame files
    def dummy_generate_frames(**kwargs):
        paths = []
        for i in range(3):
            p = tmp_path / f"{i}.png"
            p.write_text("dummy")
            paths.append(str(p))
        return paths

    monkeypatch.setattr(FrameGenerator, "generate_frames", dummy_generate_frames)
    out = generate_frames_and_collect(
        plotter=DummyPlotter(), processes=1, var_name="x", output_dir=tmp_path
    )
    assert len(out) == 3
    for p in out:
        assert Path(p).exists()


def test_create_mp4_and_cleanup(tmp_path, monkeypatch):
    # Create two dummy PNG frames
    frames = []
    for i in range(2):
        p = tmp_path / f"{i}.png"
        imageio.imwrite(str(p), np.ones((1, 1), dtype=np.uint8))
        frames.append(str(p))
    out = tmp_path / "out.mp4"

    # Mock moviepy.editor.ImageSequenceClip
    class DummyClip:
        def __init__(self, seq, fps):
            pass

        def write_videofile(self, filename, **kwargs):
            Path(filename).write_text("mp4")

    sys.modules["moviepy.editor"] = type("m", (), {"ImageSequenceClip": DummyClip})
    create_mp4(frames, output_mp4=out, fps=1, cleanup=True)
    assert out.exists()
    # Ensure source frames are deleted when cleanup=True
    for f in frames:
        assert not Path(f).exists()


def test_convert_gif_to_mp4_error(monkeypatch):
    # Simulate ffmpeg failure
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "ffmpeg")
        ),
    )
    with pytest.raises(subprocess.CalledProcessError):
        convert_gif_to_mp4("nonexistent.gif", "o.mp4")
