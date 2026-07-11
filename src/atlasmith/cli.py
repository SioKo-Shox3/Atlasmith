"""Atlasmith CLI エントリポイント。公開 `atlasmith.rebake()` を呼ぶ薄いラッパ
(読込→UV再展開→テクスチャ焼き直し→書出を一気通貫で実行する)。
"""

from __future__ import annotations

import argparse
import warnings

from atlasmith import rebake
from atlasmith.io import load_mesh


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。`<input>` を再展開+焼き直しし `-o <output>` へ書き出す。"""
    parser = argparse.ArgumentParser(prog="atlasmith")
    parser.add_argument("input", help="Input mesh file (.glb/.gltf/.obj)")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output mesh file (.glb/.gltf/.obj)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Chart padding / gutter dilation in texels (default: 8)",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=1024,
        help="Output texture edge length in texels (default: 1024)",
    )
    args = parser.parse_args(argv)

    # WHY: normal map 警告の判定用に load_mesh を一度呼ぶ。`rebake()` は公開 API の
    # 薄いラッパに留める契約(src/atlasmith/__init__.py は変更禁止)で、preloaded
    # MeshData を受け取る口が無いため、事前チェック用の読み込みを CLI 側に別途持つ
    # 以外の選択肢がない。結果として同一ファイルを CLI 実行1回につき2回 load_mesh
    # することになるが、単発 CLI 実行(ホットパスではない)なのでコストは許容する。
    # 却下した代替案: ファイル拡張子ごとに normal map の有無だけを覗く専用ロジックを
    # cli.py に持つ — load_mesh の解析ロジック(GLB/glTF/OBJ の material 属性名の
    # 違い等)を cli.py に部分的に複製することになり、io 層の実装詳細への依存が
    # 二重管理化するため不採用。
    mesh = load_mesh(args.input)
    if "normal" in mesh.maps:
        warnings.warn(
            "Input mesh has a normal map. Atlasmith transfers it to the new UV "
            "layout, but re-unwrapping the UVs changes the tangent-space basis, "
            "so lighting correctness after rebaking is not guaranteed.",
            UserWarning,
            stacklevel=2,
        )
    # WHY: rebake() が同じファイルを再度 load_mesh するため、判定用コピーを早期解放
    # する(メモリ二重常駐の抑制)。
    del mesh

    rebake(
        args.input,
        args.output,
        texture_size=args.texture_size,
        padding_px=args.padding,
    )
    return 0
