"""Atlasmith の内部表現(IR)。全パイプライン段(io/pack/bake/metrics)が共有する型。

このモジュールは numpy 以外に依存しない(横断規約の依存方向: types は無依存)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MeshData:
    """単一メッシュ・単一マテリアル・単一UVセットのテクスチャ付きメッシュ表現。

    公開だが **provisional**(pre-1.0 の間はフィールド追加があり得る)。
    値渡し前提で扱うこと(このデータクラス自身はミュータブルだが、パイプライン各段は
    引数を破壊的に変更しない — 横断規約のスレッド/並行性節を参照)。

    フィールド:
        vertices: 頂点座標 `(N, 3) float64`。
        faces: 三角形の頂点 index `(M, 3) int64`。
        uv: 頂点 UV `(N, 2) float32`(テクスチャなしメッシュは `None`)。
        maps: テクスチャ画像の辞書。値は `float32 (H, W, C) [0, 1]` の ndarray、
            channels last。正規名は `"basecolor"` / `"normal"` /
            `"metallic_roughness"`。V方向規約: row 0 = 画像上端 = V=0(glTF 規約。
            OBJ 由来の場合も io 層でこの規約に揃える)。
        source_vertex: render 頂点 → 元頂点(weld 前)の index `(N,) int64`。
            io では恒等写像(`np.arange(N)`)または trimesh の weld 由来。
            `None` は「元頂点との対応を追跡していない」ことを表す。
    """

    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None = None
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    source_vertex: np.ndarray | None = None
