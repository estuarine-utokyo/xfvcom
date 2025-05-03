import os
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from xfvcom.utils.helpers import _cleanup_files, create_gif, list_png_files


@pytest.fixture
def tmp_pngs(tmp_path):
    # テスト用に 3 つの PNG ファイルを作成
    names = ["a.png", "b.png", "c.txt", "1.png"]
    paths = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"")  # 空ファイル
        paths.append(p)
    return tmp_path, paths


def test_list_png_files(tmp_pngs):
    tmp_dir, all_paths = tmp_pngs
    pngs = list_png_files(tmp_dir)
    # .txt を除いてソートされた Path のリストが返ること
    expected = sorted([p for p in all_paths if p.suffix == ".png"])
    assert pngs == expected


def test_cleanup_files(tmp_pngs):
    tmp_dir, all_paths = tmp_pngs
    png_paths = [p for p in all_paths if p.suffix == ".png"]
    # _cleanup_files で PNG だけ削除
    _cleanup_files(png_paths)
    for p in png_paths:
        assert not p.exists()
    # .txt ファイルは残る
    assert (tmp_dir / "c.txt").exists()


def test_create_gif(tmp_path, monkeypatch):
    # 2 枚のダミー PNG を用意
    frames = []
    # Write two 1×1 pixel images with uint8 dtype
    for i in range(2):
        p = tmp_path / f"f{i}.png"
        arr = np.array([[i]], dtype=np.uint8)
        imageio.imwrite(str(p), arr)
        frames.append(str(p))
    out = tmp_path / "out.gif"
    # 実際にファイルが作成されることを確認
    create_gif(frames, output_gif=out, fps=1, cleanup=False)
    assert out.exists()
    # 中身は GIF フォーマットであるはず
    with open(out, "rb") as f:
        header = f.read(6)
    assert header in (b"GIF87a", b"GIF89a")
