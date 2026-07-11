"""ローカルデモ用のテクスチャ付き GLB(`examples/demo.glb`)を生成するスクリプト。

README のサンプルコマンド(`uv run atlasmith` の入力素材)を用意するためだけの
スクリプトであり、`atlasmith` パッケージ内では `io`/`types` のみを使う(numpy/
trimesh はメッシュ生成の道具として使う。pack/bake/cli 等の再展開・焼き直しロジック
は使わない — それは `atlasmith rebake` 側の仕事)。

実行:
    uv run python examples/make_demo_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from atlasmith.io import save_mesh
from atlasmith.types import MeshData

_OUTPUT_PATH = Path(__file__).parent / "demo.glb"
_TEXTURE_SIZE = 64
_CHECKER_CELLS = 8


def _analytic_sphere_uv(vertices: np.ndarray) -> np.ndarray:
    """頂点座標(球面上)から経度→U・緯度→V の球面 UV を解析的に作る。"""
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    r = np.linalg.norm(vertices, axis=1)
    u = 0.5 + np.arctan2(y, x) / (2.0 * np.pi)
    v = 0.5 - np.arcsin(np.clip(z / r, -1.0, 1.0)) / np.pi
    return np.stack([u, v], axis=1).astype(np.float32)


def _checker_basecolor(size: int, cells: int) -> np.ndarray:
    """再展開・焼き直しの結果が目視で判別しやすい市松模様の basecolor テクスチャ。

    戻り値は横断規約どおり `float32 (H, W, 3) [0, 1]`、row 0 = 画像上端 = V=0。
    """
    cols = (np.arange(size) + 0.5) / size
    rows = (np.arange(size) + 0.5) / size
    u_grid, v_grid = np.meshgrid(cols, rows, indexing="xy")
    u_cell = (u_grid * cells).astype(np.int64)
    v_cell = (v_grid * cells).astype(np.int64)
    checker = (u_cell + v_cell) % 2
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[..., 0] = np.where(checker == 0, 0.85, 0.10)
    img[..., 1] = np.where(checker == 0, 0.20, 0.55)
    img[..., 2] = v_grid.astype(np.float32)
    return img


def build_demo_mesh() -> MeshData:
    """UV+basecolor テクスチャ付きの球メッシュを組み立てて返す。"""
    ico = trimesh.creation.icosphere(subdivisions=2)
    vertices = np.asarray(ico.vertices, dtype=np.float64)
    faces = np.asarray(ico.faces, dtype=np.int64)
    uv = _analytic_sphere_uv(vertices)
    texture = _checker_basecolor(_TEXTURE_SIZE, _CHECKER_CELLS)
    return MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps={"basecolor": texture},
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )


def main() -> None:
    mesh = build_demo_mesh()
    save_mesh(mesh, _OUTPUT_PATH)
    print(f"Wrote demo asset: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
