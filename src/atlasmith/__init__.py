"""Atlasmith: mesh texture rebaking toolkit (UV re-unwrap, atlas packing, bake).

公開 API は 5 関数 + CLI(横断規約): `load_mesh` / `save_mesh`(io)、`bake_maps`
(bake)、`masked_psnr`(metrics)、そして高水準ラッパ `rebake`(この module)。
`rebake` は io → pack(internal 再展開)→ bake → io の結線であり、CLI はこの薄い
ラッパになる(CLI 結線は Step 1-3)。
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from atlasmith.bake import bake_maps
from atlasmith.io import load_mesh, save_mesh
from atlasmith.metrics import masked_psnr
from atlasmith.pack import _naive_unwrap_and_pack
from atlasmith.types import MeshData

# インストール済みメタデータを唯一の情報源にする(pyproject.toml と二重管理しない)。
# 未インストール・メタデータ破損時は import 時点で例外を送出させ、隠さず顕在化させる。
__version__ = version("atlasmith")

__all__ = [
    "MeshData",
    "bake_maps",
    "load_mesh",
    "masked_psnr",
    "rebake",
    "save_mesh",
]


def rebake(
    input_path: str | Path,
    output_path: str | Path,
    *,
    texture_size: int = 1024,
    padding_px: int = 8,
) -> None:
    """メッシュを読み込み、UV を再展開してテクスチャを焼き直し、書き出す。

    処理は io.load_mesh → pack(internal 再展開)→ bake.bake_maps → io.save_mesh の
    結線。新旧の面対応 `face_map` で旧面を行整列してから `bake_maps` を呼ぶ(bake は
    対応表を持たない・裁定5)。

    引数:
        input_path: 入力メッシュ(.glb/.gltf/.obj)。
        output_path: 出力メッシュ(拡張子で形式が決まる)。
        texture_size: 焼き先テクスチャの一辺(テクセル)。xatlas のパッキング解像度と
            bake の出力サイズの双方に使う。
        padding_px: チャート間パディング兼ガター膨張回数(テクセル)。xatlas と bake で
            同じ値を使い単一ソースで同期する(C9)。

    備考:
        テクスチャを持たないメッシュ(maps が空)はジオメトリと新 UV のみを書き出す。
        maps があるのに UV が無い入力は焼き元 UV を欠くため ValueError にする。
    """
    mesh = load_mesh(input_path)
    new_mesh, face_map = _naive_unwrap_and_pack(
        mesh, resolution=texture_size, padding_px=padding_px
    )

    baked_maps: dict = {}
    if mesh.maps:
        if mesh.uv is None:
            raise ValueError(
                "rebake: mesh has texture maps but no UV coordinates to sample from"
            )
        # 旧面を face_map で行整列 → 新面と行・corner 整列(bake の入力契約・裁定5)。
        faces_old_aligned = mesh.faces[face_map]
        result = bake_maps(
            new_mesh.faces,
            new_mesh.uv,
            faces_old_aligned,
            mesh.uv,
            mesh.maps,
            size=(texture_size, texture_size),
            padding_px=padding_px,
        )
        baked_maps = result.maps

    out_mesh = MeshData(
        vertices=new_mesh.vertices,
        faces=new_mesh.faces,
        uv=new_mesh.uv,
        maps=baked_maps,
        source_vertex=new_mesh.source_vertex,
    )
    save_mesh(out_mesh, output_path)
