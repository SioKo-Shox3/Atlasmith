"""Atlasmith CLI エントリポイント。公開 `atlasmith.rebake()` を呼ぶ薄いラッパ
(読込→部位分割→UV再展開→テクスチャ焼き直し→書出を一気通貫で実行する)。

**`--segmenter` は `--seg-*` フラグではない**(計画v4 §4.3 の明文化。3 箇所に書く
うちの1つ): 衝突判定でいう「`--seg-*` フラグ」は `--seg-angle` / `--seg-min-faces` /
`--seg-views` / `--seg-model` の 4 つだけを指す。接頭辞が紛らわしいので、判定表
(`_SEG_PARAM_FLAGS`)とヘルプ文にも同じ断り書きを置く。

**CLI 既定 `--segmenter sam2` と API 既定 `DihedralSegmenter()` は異なる**
(2026-07-27 ユーザー裁定 E)。`rebake` が既定で SAM2 を構築すると
`atlasmith(rebake) → segmentation`(numpy のみ)の依存方向が壊れるため、ML の既定は
この CLI レイヤだけに置く(`src/atlasmith/__init__.py` の module docstring 参照)。
"""

from __future__ import annotations

import argparse
import contextlib
import warnings
from collections.abc import Iterator
from typing import Any

from atlasmith import rebake
from atlasmith.io import load_mesh
from atlasmith.segmentation import DihedralSegmenter, SegmentationBackend

# `--segmenter` 未指定を見分けるための番兵。既定値そのものは
# `_DEFAULT_SEGMENTER` が持つ(argparse の `default=` に本来の既定を書くと
# 「明示指定されたか」が判定できず、衝突判定規約2/3 が書けない — 計画v2 BL-6 方式)。
_DEFAULT_SEGMENTER = "sam2"

# 衝突判定でいう「`--seg-*` フラグ」の全体(`--segmenter` は**含まない**)。
# 値は (dest, フラグ名) で、メッセージにフラグ名をそのまま出すために持つ。
_SEG_PARAM_FLAGS = (
    ("seg_angle", "--seg-angle"),
    ("seg_min_faces", "--seg-min-faces"),
    ("seg_views", "--seg-views"),
    ("seg_model", "--seg-model"),
)

# `--segmenter geometric` を**明示**したときに併用できない sam2 専用フラグ。
_SAM2_ONLY_FLAGS = (("seg_views", "--seg-views"), ("seg_model", "--seg-model"))


def _build_parser() -> argparse.ArgumentParser:
    """CLI の引数パーサを組み立てる。"""
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
    parser.add_argument(
        "--granularity",
        choices=("part", "naive"),
        default="part",
        help=(
            "UV island granularity: 'part' unwraps each detected part separately "
            "so that every UV island lies inside a single part; 'naive' unwraps "
            "the whole mesh in one go (Phase 1 behaviour). (default: part)"
        ),
    )
    parser.add_argument(
        "--segmenter",
        choices=("geometric", "sam2"),
        default=None,  # 番兵。実効既定は `_DEFAULT_SEGMENTER`。
        help=(
            f"Part segmentation backend (default: {_DEFAULT_SEGMENTER}). "
            "'geometric' is dihedral-angle clustering (no extra dependencies); "
            "'sam2' is multi-view SAM2 and needs the [ml] extra. NOTE: "
            "--segmenter is NOT one of the --seg-* flags below."
        ),
    )
    parser.add_argument(
        "--seg-angle",
        type=float,
        default=None,
        help=(
            "Dihedral angle threshold in degrees; adjacent faces at or below it "
            "stay in the same part. Valid for both backends (a geometric prior "
            "for sam2). (default: backend default)"
        ),
    )
    parser.add_argument(
        "--seg-min-faces",
        type=int,
        default=None,
        help=(
            "Parts smaller than this many faces are merged into a neighbour. "
            "Valid for both backends. (default: backend default)"
        ),
    )
    parser.add_argument(
        "--seg-views",
        type=int,
        default=None,
        help="Number of rendered viewpoints (--segmenter sam2 only).",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default=None,
        help="SAM2 model id, e.g. facebook/sam2.1-hiera-large (--segmenter sam2 only).",
    )
    return parser


def _check_flag_conflicts(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """計画v4 §4.3 の衝突判定 3 規則を適用する(違反は `parser.error` = exit 2)。

    番兵方式(`default=None`)なので、**明示指定されたときだけ**判定する。黙って
    無視される指定を残さないためのゲートであり、既定値経由での組み合わせは許容する。

    規則:
      1. `--granularity naive` + `--seg-*` のいずれかを明示。
      2. `--granularity naive` + `--segmenter` を明示(既定値経由は許容)。
      3. `--segmenter geometric` を**明示** + `--seg-views` / `--seg-model` を明示。
         **既定経由は除外する**(2026-07-27 裁定 E で既定が `sam2` になったため、
         「既定で geometric」という状況自体が存在しない。`[ml]` 未導入時の
         geometric フォールバックは argparse 解析より後段で起きるので、ここでは
         判定できないし判定しない)。

    Args:
        parser: エラー送出に使うパーサ。
        args: 解析済み引数。
    """
    if args.granularity == "naive":
        for dest, flag in _SEG_PARAM_FLAGS:
            if getattr(args, dest) is not None:
                parser.error(
                    f"{flag} has no effect with --granularity naive (the naive path "
                    "does not segment the mesh). Drop the flag or use "
                    "--granularity part."
                )
        if args.segmenter is not None:
            parser.error(
                "--segmenter has no effect with --granularity naive (the naive path "
                "does not segment the mesh). Drop the flag or use --granularity part."
            )
    if args.segmenter == "geometric":
        for dest, flag in _SAM2_ONLY_FLAGS:
            if getattr(args, dest) is not None:
                parser.error(
                    f"{flag} only applies to --segmenter sam2, but --segmenter "
                    "geometric was given explicitly. Drop the flag or switch to "
                    "--segmenter sam2."
                )


@contextlib.contextmanager
def _segmentation_backend(
    args: argparse.Namespace,
) -> Iterator[SegmentationBackend | None]:
    """`rebake` へ渡す分割バックエンドを組み立て、**その寿命を所有する**。

    計画v4 §2.1 の所有権規約: `rebake` は注入されたバックエンドを閉じないので、
    CLI が自分で作ったものは CLI が `with` で閉じる。`DihedralSegmenter` は資源を
    持たないので context manager ではなく、そのまま yield する。

    `[ml]` 未導入時の扱い(§0-B 裁定 E の帰結2):
      - `--segmenter sam2` を**明示**した → `ImportError` をそのまま伝播させる
        (3 経路を提示する `sam2_masks` のメッセージが CLI の非ゼロ終了とともに
        出る)。「黙ってフォールバックしない」原則は明示指定経路で保たれる。
      - **既定経由**(`--segmenter` 未指定)→ `warnings.warn` して `geometric` へ
        フォールバックする。CI(`uv sync --locked` = extras 無し)と既定 CLI パスを
        踏む既存テストを green に保つための裁定であり、警告を出すので「黙って」
        ではない。

    Args:
        args: 解析済み引数(衝突判定通過後)。

    Yields:
        `granularity="naive"` のときは `None`(素朴経路は分割しない。`rebake` は
        naive + backend 指定を `ValueError` にするので None を渡す必要がある)、
        それ以外は `SegmentationBackend`。
    """
    if args.granularity == "naive":
        yield None
        return

    explicit = args.segmenter is not None
    name = args.segmenter if explicit else _DEFAULT_SEGMENTER

    # 両バックエンド共通のパラメータ。番兵 None は「バックエンドの既定に任せる」。
    shared: dict[str, Any] = {}
    if args.seg_angle is not None:
        shared["angle_deg"] = args.seg_angle
    if args.seg_min_faces is not None:
        shared["min_faces"] = args.seg_min_faces

    if name == "sam2":
        sam2_kwargs: dict[str, Any] = dict(shared)
        if args.seg_views is not None:
            sam2_kwargs["n_views"] = args.seg_views
        if args.seg_model is not None:
            sam2_kwargs["model_id"] = args.seg_model
        # WHY 関数内 import: `segmentation.multiview` は torch/sam2/moderngl を引く
        # 隔離モジュールへの入口であり、module 直下で import すると
        # `import atlasmith.cli`(= console script の起動)だけで ML 依存の解決が
        # 走る。`--segmenter geometric` や `--granularity naive` の利用者に ML の
        # コストと失敗経路を負わせないため、sam2 を選んだときだけ触る
        # (計画v4 §2.1 隔離モジュール規約 / §4.3)。
        from atlasmith.segmentation.multiview import make_sam2_segmenter

        segmenter = None
        try:
            segmenter = make_sam2_segmenter(**sam2_kwargs)
        except ImportError as error:
            if explicit:
                raise
            warnings.warn(
                "--segmenter sam2 is the default but the [ml] extra is not "
                f"installed ({error}); falling back to --segmenter geometric "
                "(dihedral-angle clustering). Install the extra with "
                '`uv sync --extra ml` (or `pip install "atlasmith[ml]"`) to use '
                "SAM2, or pass --segmenter geometric to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
        if segmenter is not None:
            # `MultiViewSegmenter` は proposer(= SAM2 の重み)を所有する
            # context manager。ここが唯一の閉じ手。
            with segmenter as entered:
                yield entered
            return

    yield DihedralSegmenter(**shared)


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。`<input>` を再展開+焼き直しし `-o <output>` へ書き出す。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _check_flag_conflicts(parser, args)

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

    # WHY ここにアトラス寸法まわりの通知を書かないか: ガター消失(`g == 0`)の警告も
    # `D > texture_size` の `logging.info` も `rebake` が出す(計画v4 §2.5 /
    # 2026-08-06 裁定2)。API 直呼びの利用者にも必要な情報であり、CLI にも置くと
    # 同じ状況で 2 回出る。
    with _segmentation_backend(args) as backend:
        rebake(
            args.input,
            args.output,
            texture_size=args.texture_size,
            padding_px=args.padding,
            granularity=args.granularity,
            segmentation=backend,
        )
    return 0
