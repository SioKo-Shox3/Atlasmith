"""スタンドアロン画像ファイルの入出力(Pillow バックエンド)。

IR の画像配列契約: `float32 (H, W, C) [0, 1]`、channels last、8bit 精度
(判断2: EXR/16bit 需要が出た時点で OpenImageIO を再評価)。
`_array_to_pil`/`_pil_to_array` は `io/mesh.py` からも再利用する
(埋め込みテクスチャの変換ロジックを単一ソースに保つため)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_MODE_BY_CHANNELS = {1: "L", 3: "RGB", 4: "RGBA"}


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    """float32 [0, 1] の HxWxC 配列を 8bit Pillow Image へ変換する。"""
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected an (H, W, C) array, got shape {arr.shape}")
    channels = arr.shape[2]
    mode = _MODE_BY_CHANNELS.get(channels)
    if mode is None:
        raise ValueError(f"Unsupported channel count: {channels} (expected 1, 3, or 4)")
    u8 = np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if channels == 1:
        return Image.fromarray(u8[:, :, 0], mode=mode)
    return Image.fromarray(u8, mode=mode)


def _pil_to_array(img: Image.Image) -> np.ndarray:
    """8bit Pillow Image を float32 [0, 1] の HxWxC 配列へ変換する。"""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return (arr.astype(np.float32) / 255.0).astype(np.float32)


def load_image(path: str | Path) -> np.ndarray:
    """画像ファイルを読み、`float32 (H, W, C) [0, 1]` 配列として返す。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    with Image.open(path) as img:
        img.load()
        return _pil_to_array(img)


def save_image(arr: np.ndarray, path: str | Path) -> None:
    """`float32 (H, W, C) [0, 1]` 配列を画像ファイルへ書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _array_to_pil(arr).save(path)
