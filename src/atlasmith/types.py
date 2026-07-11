"""Atlasmith の内部表現(IR)。全パイプライン段(io/pack/bake/metrics)が共有する型。

このモジュールは numpy のみに依存する(横断規約の依存方向:
trimesh/PIL/xatlas は import しない)。
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
        vertices: 頂点座標 `(N, 3) float64`。N >= 1 を要求する(空メッシュは不正 —
            `vertices` を持たないメッシュを表現する用途は無い)。
        faces: 三角形の頂点 index `(M, 3) int64`。M >= 0(面無しメッシュは許容 —
            `vertices` の N>=1 制約とは非対称。頂点だけあって面が未確定の中間状態を
            表現できるようにするため意図的)。
        uv: 頂点 UV `(N, 2) float32`(テクスチャなしメッシュは `None`)。
        maps: テクスチャ画像の辞書。値は `float32 [0, 1]` の ndarray で、
            `(H, W, C)` の3次元(channels last)、または `(H, W)` の2次元
            グレースケール(単チャンネルの省略形)を許容する。正規名は
            `"basecolor"` / `"normal"` / `"metallic_roughness"`。V方向規約:
            row 0 = 画像上端 = V=0(glTF 規約。OBJ 由来の場合も io 層で
            この規約に揃える)。
        source_vertex: render 頂点 → 元頂点(weld 前)の index `(N,) int64`。
            io では恒等写像(`np.arange(N)`)または trimesh の weld 由来。
            `None` は「元頂点との対応を追跡していない」ことを表す。
    """

    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None = None
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    source_vertex: np.ndarray | None = None

    def __post_init__(self) -> None:
        """フィールドの shape/dtype 契約を構築時に検証する。

        docstring が謳う shape/dtype と食い違う `MeshData` の暗黙構築を早期に弾く。
        検証は型・shape・dtype のみで、値域(座標範囲・UV の [0,1] 等)は見ない
        (コストと責務の観点で呼び出し側の契約とする)。不正時は `ValueError` を
        投げ、メッセージに実際の shape/dtype を含める。
        """
        if not isinstance(self.vertices, np.ndarray):
            raise ValueError(
                f"vertices must be a numpy ndarray, got {type(self.vertices).__name__}"
            )
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (N, 3), got {self.vertices.shape}"
            )
        if self.vertices.shape[0] < 1:
            raise ValueError(
                f"vertices must have at least one row (N >= 1), got shape "
                f"{self.vertices.shape}"
            )
        if self.vertices.dtype != np.float64:
            raise ValueError(
                f"vertices must have dtype float64, got {self.vertices.dtype}"
            )
        n_vertices = self.vertices.shape[0]

        if not isinstance(self.faces, np.ndarray):
            raise ValueError(
                f"faces must be a numpy ndarray, got {type(self.faces).__name__}"
            )
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (M, 3), got {self.faces.shape}")
        if self.faces.dtype != np.int64:
            raise ValueError(f"faces must have dtype int64, got {self.faces.dtype}")

        if self.uv is not None:
            if not isinstance(self.uv, np.ndarray):
                raise ValueError(
                    f"uv must be None or a numpy ndarray, got {type(self.uv).__name__}"
                )
            if self.uv.ndim != 2 or self.uv.shape[1] != 2:
                raise ValueError(f"uv must have shape (N, 2), got {self.uv.shape}")
            if self.uv.shape[0] != n_vertices:
                raise ValueError(
                    f"uv must have the same N as vertices ({n_vertices}), got "
                    f"{self.uv.shape[0]}"
                )
            if self.uv.dtype != np.float32:
                raise ValueError(f"uv must have dtype float32, got {self.uv.dtype}")

        if not isinstance(self.maps, dict):
            raise ValueError(f"maps must be a dict, got {type(self.maps).__name__}")
        for name, image in self.maps.items():
            if not isinstance(name, str):
                raise ValueError(f"maps keys must be str, got {type(name).__name__}")
            if not isinstance(image, np.ndarray):
                raise ValueError(
                    f"maps[{name!r}] must be a numpy ndarray, got "
                    f"{type(image).__name__}"
                )
            if image.ndim not in (2, 3):
                raise ValueError(
                    f"maps[{name!r}] must be 2D (H, W) or 3D (H, W, C), got shape "
                    f"{image.shape}"
                )
            if image.dtype != np.float32:
                raise ValueError(
                    f"maps[{name!r}] must have dtype float32, got {image.dtype}"
                )

        if self.source_vertex is not None:
            if not isinstance(self.source_vertex, np.ndarray):
                raise ValueError(
                    "source_vertex must be None or a numpy ndarray, got "
                    f"{type(self.source_vertex).__name__}"
                )
            if self.source_vertex.ndim != 1:
                raise ValueError(
                    f"source_vertex must have shape (N,), got "
                    f"{self.source_vertex.shape}"
                )
            if self.source_vertex.shape[0] != n_vertices:
                raise ValueError(
                    f"source_vertex must have the same N as vertices ({n_vertices}), "
                    f"got {self.source_vertex.shape[0]}"
                )
            if self.source_vertex.dtype != np.int64:
                raise ValueError(
                    "source_vertex must have dtype int64, got "
                    f"{self.source_vertex.dtype}"
                )
