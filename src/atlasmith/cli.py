"""Atlasmith CLI エントリポイント。load→save のラウンドトリップを結線する
(`--padding`/`--texture-size` 等の再展開オプションは Phase 1 で追加)。
"""

import argparse

from atlasmith.io import load_mesh, save_mesh


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。`<input>` を読み `-o <output>` へ書き出す。"""
    parser = argparse.ArgumentParser(prog="atlasmith")
    parser.add_argument("input", help="Input mesh file (.glb/.gltf/.obj)")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output mesh file (.glb/.gltf/.obj)",
    )
    args = parser.parse_args(argv)

    mesh = load_mesh(args.input)
    save_mesh(mesh, args.output)
    return 0
