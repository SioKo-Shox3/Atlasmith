"""Atlasmith: mesh texture rebaking toolkit (UV re-unwrap, atlas packing, bake).

公開 API は 5 関数 + 4 公開型 + CLI(横断規約): `load_mesh` / `save_mesh`(io)、
`bake_maps`(bake)、`masked_psnr`(metrics)、そして高水準ラッパ `rebake`
(この module)。公開型は `MeshData` / `BakeResult` に加え、Phase 2 で
`SegmentationBackend` / `DihedralSegmenter`(計画v4 §4.2 の裁定 C')。
`rebake` は io → segmentation → pack(internal 再展開)→ bake → io の結線であり、
CLI はこの薄いラッパになる。

**`rebake` は sam2 を知らない**(計画v4 §4.1 / §0-B 裁定 E の帰結3): この module が
import するのは `atlasmith.segmentation`(numpy のみ)だけで、ML バックエンドは
呼び出し側が `segmentation=` に注入する。したがって API 既定は
`DihedralSegmenter()` であり、**CLI 既定(`--segmenter sam2`)とは異なる**。
`rebake` が既定で SAM2 を構築すると `segmentation.multiview` への依存が生まれ、
`atlasmith(rebake) → segmentation`(numpy のみ)という依存方向が壊れるため。
"""

from __future__ import annotations

import logging
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np

from atlasmith.bake import bake_maps
from atlasmith.io import load_mesh, save_mesh
from atlasmith.metrics import masked_psnr
from atlasmith.pack import _naive_unwrap_and_pack, _part_unwrap_and_pack
from atlasmith.segmentation import DihedralSegmenter, SegmentationBackend
from atlasmith.types import MeshData

# インストール済みメタデータを唯一の情報源にする(pyproject.toml と二重管理しない)。
# 未インストール・メタデータ破損時は import 時点で例外を送出させ、隠さず顕在化させる。
__version__ = version("atlasmith")

__all__ = [
    "DihedralSegmenter",
    "MeshData",
    "SegmentationBackend",
    "bake_maps",
    "load_mesh",
    "masked_psnr",
    "rebake",
    "save_mesh",
]

_LOGGER = logging.getLogger(__name__)

# `granularity` が取りうる値。未知の値は黙って既定へ倒さず `ValueError`(裁定B)。
GRANULARITIES = ("part", "naive")


def _validate_size_argument(
    value: object, name: str, flag: str, *, minimum: int
) -> int:
    """`rebake` の寸法系引数(テクセル数)を入口で検疫する。

    **WHY 入口か(2026-08-07 外部レビュー指摘)**: これらの値は検証されないまま
    xatlas の `PackOptions` へ流れていた。実測の被害は 2 種類:

      - `texture_size=0` は**エラーにならない**。1450x726 のアトラスを組む高コスト
        処理を最後まで走らせてから `(0, 0, 3)` の無効なテクスチャを書き出す。
      - 負値は xatlas 内部の生 `TypeError` になる。引数名も、直し方も出ない。

    どちらも「呼び出した瞬間に、名前つきで落ちる」べきもの。`bool` を弾くのは
    `isinstance(True, int)` が真だからで、この規約は本リポジトリの他の検証関数
    (`multiview` の `n_views` 等)と同じ形に揃えてある。

    Args:
        value: 検疫対象の値。
        name: `rebake` の引数名(メッセージに出す)。
        flag: 対応する CLI フラグ名(利用者の入口はほぼこちら)。
        minimum: 許す最小値。

    Returns:
        `int` へ正規化した値(`np.integer` も受ける)。

    Raises:
        ValueError: 整数でない、または `minimum` 未満のとき。
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"rebake: {name} must be an int (texels), got "
            f"{type(value).__name__} {value!r}"
        )
    number = int(value)
    if number < minimum:
        raise ValueError(
            f"rebake: {name} must be >= {minimum}, got {number}. It is measured in "
            f"texels and is passed straight to the xatlas packer, which rejects "
            f"or silently mis-packs out-of-range values. Pass a valid size "
            f"(CLI: {flag})."
        )
    return number


def _gutter_iterations(padding_px: int, texture_size: int, atlas_edge: int) -> int:
    """部位経路のガター反復数 `g` を計算する(padding 同期規約 v2・計画v2 §2.3a 手順3)。

    `g = min(padding_px, floor(padding_px * texture_size / D))`(`D = max(w, h)`)。

    **WHY 部位経路だけか**: xatlas の `padding` は**アトラス画素**基準だが、bake の
    ガター膨張は**出力テクスチャのテクセル**基準である。部位経路は規約 (b) で UV を
    `1/D` スケールして正方形 `texture_size²` へ焼くので、`D > texture_size` のとき
    アトラス画素 1 個は 1 テクセル未満に縮む。素の `padding_px` 回だけ膨張させると
    チャート間ガターが隣のチャートを踏み越え、相互汚染(滲み)が起きる。比
    `texture_size/D` でスケールした回数まで縮めると、テクセル空間での膨張幅が
    xatlas の保証内に収まる。
    `texture_size >= D` なら `g == padding_px` で Phase 1 と同じ関係に戻る。

    **WHY `g == 0` を 1 に床上げしないか**(2026-08-06 オーケストレーター裁定1):
    `padding_px * texture_size < D` ではテクセル空間でチャートが既に隣接しており、
    1 回でも膨張させれば確実に汚染する。ガターが張れない状況であることを利用者へ
    伝えるのは `rebake` の `g == 0` 警告の役割で、床上げで隠す対象ではない。

    Args:
        padding_px: 利用者が指定したパディング(テクセル)。
        texture_size: 焼き先テクスチャの一辺(テクセル)。
        atlas_edge: `D = max(atlas.width, atlas.height)`(アトラス画素)。**正**。

    Returns:
        `bake_maps` へ渡すガター反復数 `g`(`0 <= g <= padding_px`)。
    """
    # `atlas_edge >= 1` は `_check_atlas_dims`(pack 側)が保証済み。
    return min(padding_px, padding_px * texture_size // atlas_edge)


def rebake(
    input_path: str | Path,
    output_path: str | Path,
    *,
    texture_size: int = 1024,
    padding_px: int = 8,
    granularity: Literal["part", "naive"] = "part",
    segmentation: SegmentationBackend | None = None,
) -> None:
    """メッシュを読み込み、UV を再展開してテクスチャを焼き直し、書き出す。

    処理は io.load_mesh → (部位分割) → pack(internal 再展開)→ bake.bake_maps →
    io.save_mesh の結線。新旧の面対応 `face_map` で旧面を行整列してから `bake_maps`
    を呼ぶ(bake は対応表を持たない・裁定5)。

    引数:
        input_path: 入力メッシュ(.glb/.gltf/.obj)。
        output_path: 出力メッシュ(拡張子で形式が決まる)。
        texture_size: 焼き先テクスチャの一辺(テクセル)。**1 以上の int**。xatlas の
            パッキング解像度と bake の出力サイズの双方に使う。
        padding_px: チャート間パディング兼ガター膨張回数(テクセル)。**0 以上の
            int**(0 = ガター無しを意図的に選ぶ状態)。xatlas へは
            この値をそのまま渡す。**部位経路では** bake のガター反復数だけが
            アトラス実寸法に応じて `_gutter_iterations` で縮む(計画v2 §2.3a 手順3)。
            naive 経路は Phase 1 と bit 一致を保つため素の値のまま使う。
        granularity: `"part"`(既定)なら部位単位アイランド、`"naive"` ならメッシュ
            全体を 1 回で展開する Phase 1 経路。未知の値は `ValueError`。
        segmentation: 部位分割バックエンド(`SegmentationBackend`)。`None`(既定)
            なら `DihedralSegmenter()` を使う。`granularity="naive"` との併用は
            `ValueError`(素朴経路は分割を行わないので、指定が黙って無視される
            のを防ぐ)。

    Raises:
        ValueError: `texture_size` / `padding_px` が整数でないか範囲外
            (`texture_size >= 1` / `padding_px >= 0` — `_validate_size_argument`)、
            `granularity` が未知、`granularity="naive"` に `segmentation`
            を指定した、maps があるのに UV が無い、または pack 層の検疫・不変条件に
            反したとき。

    Warns:
        UserWarning: 部位経路で**ガター反復数 `g` が 0 に落ちた**とき(= アトラスが
            `texture_size` に対して大きすぎて、チャート外周にガターを 1 テクセルも
            張れない)。焼き上がりテクスチャをバイリニアでサンプルするとチャート
            境界で背景が混ざる — 実害が確定するのはこの場合だけなので、警告は
            ここに置く。**この警告は `rebake` が出す**(計画v4 §2.5 / 2026-08-06
            オーケストレーター裁定)。API 直呼びの利用者にも必要な情報なので CLI 側
            には置かない — CLI は二重に出さない。

        **WHY `D > texture_size` そのものは警告にしないか**(2026-08-06 裁定・
        実測に基づく本文の訂正): 計画 §2.5 は `D > texture_size` を「アトラス超過」
        という**例外的状況**として設計していたが、実測でそれは誤りだった。
        **xatlas の `PackOptions.resolution` はアトラス寸法の上限ではなく、
        テクセル密度のヒントである** — 実測 2026-08-06、`cube` /
        `capped_cylinder` / `two_cubes` / `torus` の 4 fixture × `angle_deg`
        60/180 × `resolution` 64/128/256/512/1024/2048 の **48 構成すべてで
        `D > resolution`** になり、比 `D/resolution` は 1.05〜1.9、部位数 P とは
        無関係だった(P=1 でも超過する)。つまりこの条件は部位経路の**常態**で
        あり、警告にすると全実行で出る定常ノイズになる。密度低下の程度は
        `g = min(padding_px, floor(padding_px·texture_size/D))` に現れ、
        `D/ts=1.9` でも `padding_px=8` なら `g=4`(半減)、`D/ts=1.05` なら
        `g=7`(ほぼ不変)で実害は無い。**実害が出るのは `g` が 0 に落ちるときだけ**
        なので、`D > texture_size` は `logging.info` で数値つきに記録するに留める。

    備考:
        **注入された `segmentation` を `rebake` は閉じない**(計画v4 §2.1 / §4.1 の
        所有権規約)。`MultiViewSegmenter` のような資源を持つバックエンドの
        `close()` / `__exit__` を呼ぶのは**呼び出し側の責任**である:

            with make_sam2_segmenter() as backend:
                rebake(src, dst, segmentation=backend)

        **API 既定と CLI 既定は異なる**(§0-B 裁定 E): 本関数の既定は
        `DihedralSegmenter()`(依存ゼロ・決定的)だが、CLI の `--segmenter` 既定は
        `sam2` である。理由はこの module の docstring を参照。

        テクスチャを持たないメッシュ(maps が空)はジオメトリと新 UV のみを書き出す。
        maps があるのに UV が無い入力は焼き元 UV を欠くため ValueError にする。
    """
    # 検疫は**読み込みより前**に済ませる(無効な設定で数十秒の処理を始めない)。
    texture_size = _validate_size_argument(
        texture_size, "texture_size", "--texture-size", minimum=1
    )
    padding_px = _validate_size_argument(
        padding_px, "padding_px", "--padding", minimum=0
    )
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"rebake: unknown granularity {granularity!r}, expected one of "
            f"{list(GRANULARITIES)}"
        )
    if granularity == "naive" and segmentation is not None:
        raise ValueError(
            "rebake: granularity='naive' does not segment the mesh, so the "
            "`segmentation` backend would be silently ignored. Pass "
            "granularity='part' to use it, or drop the `segmentation` argument."
        )

    mesh = load_mesh(input_path)

    if granularity == "part":
        backend = DihedralSegmenter() if segmentation is None else segmentation
        labels = backend.segment(mesh)
        new_mesh, face_map, dims = _part_unwrap_and_pack(
            mesh, labels, resolution=texture_size, padding_px=padding_px
        )
        atlas_edge = max(dims.width, dims.height)
        gutter_px = _gutter_iterations(padding_px, texture_size, atlas_edge)
        if atlas_edge > texture_size:
            # 常態(48/48 で成立 — docstring の実測参照)。警告ではなく記録に留める。
            _LOGGER.info(
                "part atlas is %dx%d (D=%d) for texture_size=%d: the UV layout is "
                "scaled down to fit, so the effective texel density is %.2fx what "
                "was requested and the bake gutter is %d of %d texel(s). This is "
                "normal - xatlas treats `resolution` as a density hint, not a cap.",
                dims.width,
                dims.height,
                atlas_edge,
                texture_size,
                texture_size / atlas_edge,
                gutter_px,
                padding_px,
            )
        # 実害が確定する唯一のケース: チャート外周に 1 テクセルもガターが無い。
        # `padding_px == 0` は利用者が**自分でガター無しを選んだ**状態なので除外する
        # (要求どおりの結果に警告を出さない)。警告するのは「ガターを頼んだのに
        # アトラス超過で消えた」ときだけ。
        if gutter_px == 0 and padding_px > 0:
            warnings.warn(
                f"the bake gutter collapsed to 0 texels: xatlas packed the parts "
                f"into a {dims.width}x{dims.height} atlas (D={atlas_edge}) while "
                f"texture_size={texture_size} and padding_px={padding_px}, so "
                f"padding_px * texture_size ({padding_px * texture_size}) is smaller "
                f"than D. Chart borders get no gutter at all, so bilinear sampling "
                "of the baked texture will bleed background into the chart edges. "
                "Raise texture_size or padding_px (--texture-size / --padding).",
                UserWarning,
                stacklevel=2,
            )
    else:
        # naive 経路は Phase 1 と**同一の値**を使う。ガター規約を持ち込むと
        # 「手組みパイプラインと bit 一致」という後方互換の合否基準が壊れる。
        new_mesh, face_map = _naive_unwrap_and_pack(
            mesh, resolution=texture_size, padding_px=padding_px
        )
        gutter_px = padding_px

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
            padding_px=gutter_px,
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
