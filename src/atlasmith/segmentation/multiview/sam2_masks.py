"""SAM2 自動マスク生成の `MaskProposer` アダプタ(隔離モジュール2)。

**隔離規約(計画v4 §2.1)**: torch / sam2 に触れてよいのは本ファイルと
`render.py` だけで、しかも `_import_torch()` / `_import_sam2()` の**関数内
import だけ**。module 直下に書くと `tests/test_import_isolation.py` の AST
ゲートが落ちる。未導入時の `ImportError` は行動可能なメッセージへ変換して
`from e` で再送出する(導入済み環境でも monkeypatch でこの経路をテストできる
— 規約2 の WHY)。

既定パラメータの出典(すべて 2026-08-01 の probe 実測 — 設計根拠):

  - 単色 peanut(6016 面)は `sam2-hiera-tiny` + AMG 既定閾値で **P=1 全滅**。
  - `sam2.1-hiera-large` + 閾値 0.5/0.7 + シルエット内点グリッド +
    面積帯 [0.05, 0.85] で **P=2 / accuracy 0.9574 / Δ +0.4574**。
  - 切り分け: **SDF(厚み)チャンネル単独 = 複合と同値(0.9574)、法線
    シェーディング単独 = 全滅(0.5000)** — SDF チャンネルは必須。

構成(2026-08-01 オーケストレーター裁定):

  - 裁定A: SDF は per-mesh、`MaskProposer.propose(view)` は per-view なので、
    `MultiViewSegmenter` の**サブクラス**が `segment(mesh)` の冒頭で面厚みを
    計算して proposer へ渡す(`MultiViewSegmenter` 本体は変更しない — Step 2-4
    のレビュー済みコードを凍結する)。厚みは毎 `segment()` で上書きし、
    メッシュ跨ぎの stale を作らない。
  - 裁定B: SDF 画像は proposer 内で `face_id` ルックアップにより合成し、
    チャンネル(既定 `("sdf", "shading")`)ごとに AMG を回して候補を連結する。
"""

from __future__ import annotations

import difflib
import inspect
import logging
import time
import warnings
from typing import Any, Callable, Sequence

import numpy as np

from atlasmith.segmentation.multiview import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SHADING,
    MeshRenderer,
    MultiViewSegmenter,
    RenderedView,
)
from atlasmith.segmentation.multiview import thickness as thickness_module
from atlasmith.segmentation.multiview.cameras import DEFAULT_N_VIEWS
from atlasmith.types import MeshData

__all__ = [
    "DEFAULT_AREA_BAND",
    "DEFAULT_CHANNELS",
    "DEFAULT_CROP_N_LAYERS",
    "DEFAULT_GRID_SIDE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PRED_IOU_THRESH",
    "DEFAULT_STABILITY_SCORE_THRESH",
    "Sam2MaskProposer",
    "build_sam2_segmenter",
]

LOG = logging.getLogger(__name__)

# 既定モデル。**WHY large**: tiny は単色 peanut で全滅、large は 0.9574(冒頭の
# 実測)。tiny を既定へ下げる判断はユーザー裁定事項(2026-08-01 指示)。
DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-large"
# AMG の filter 閾値。SAMesh(arXiv 2408.13679)parity の低閾値 — AMG 既定
# (0.88/0.95)では滑らかな単色形状の候補が全て落ちる(probe 実測)。
DEFAULT_PRED_IOU_THRESH = 0.5
DEFAULT_STABILITY_SCORE_THRESH = 0.7
# AMG のクロップ階層数。**0 以外は受理しない** — 理由は `_validate_crop_n_layers`。
DEFAULT_CROP_N_LAYERS = 0
# シルエット内点グリッドの一辺(点数はその 2 乗)。一様グリッドだと背景に
# シードの大半を捨てる — 前景画素から等間隔に採る(probe 方式)。
DEFAULT_GRID_SIDE = 32
# マスク面積 ÷ シルエット面積の許容帯。全体マスク(~1.0)と微小パッチを落とす。
# 比の分母はシルエット、分子は背景画素も数えるので 1.0 を超えうる。
DEFAULT_AREA_BAND = (0.05, 0.85)
# AMG へ渡す画像の種類。**WHY sdf が先頭で必須級か**: 冒頭の切り分け実測。
# `("sdf",)` に絞る構成も許す(shading を落としても品質は同値だった)。
DEFAULT_CHANNELS = ("sdf", "shading")

# 許されるチャンネル名。
_CHANNEL_NAMES = ("sdf", "shading")

# シード整合検査: 返却候補の point_coords がシルエット内に載っている率の下限と、
# 検査に使う候補数の上限(全候補を見る必要はない — 差し替えの有効性確認だけ)。
_SEED_INSIDE_MIN = 0.9
_SEED_CHECK_MAX_RECORDS = 64

# 「検証済み」= 冒頭 probe と Step 2-5 の ML ゲートが実際に測った構成。既定を
# **下回る** 値は未検証であり、実測(2026-08-03 反証レビュー B-3)では ML の寄与が
# ゼロへ崩壊した — 下の `_warn_if_below_validated_defaults` がそれを告知する。
_COLLAPSE_EVIDENCE = (
    "Only the defaults (n_views=%d, image_size=%d) have measured quality: with "
    "n_views=8 alone, or image_size=512 alone, the measured accuracy on the "
    "monochrome peanut fell from 0.9574 to 0.5000 with P=1 - identical to the "
    "geometric prior, i.e. the ML contribution collapsed to zero."
) % (DEFAULT_N_VIEWS, DEFAULT_IMAGE_SIZE)


def _is_absent(error: ImportError, module: str) -> bool:
    """`error` が「`module` **本体**がそもそも存在しない」ことを示すか。

    **不在と破損を分ける唯一の述語**(2026-08-07 外部レビュー指摘)。CLI の既定
    経路は「`[ml]` extra 未導入」のときだけ幾何バックエンドへ落ちてよく、
    **壊れたインストールを黙って別アルゴリズムで代替してはならない**(exit 0 の
    まま別物の成果物が出て、原因も誤って報告される)。

    判定は 2 条件の連言:

      - `ModuleNotFoundError` であること — ABI 不整合や DLL ロード失敗、
        `cannot import name ...` は素の `ImportError` なのでここで落ちる。
      - 欠落モジュール名が `module` に**完全一致**すること — 推移的依存の欠落
        (`No module named 'hydra'`)や部分的に壊れた配布物
        (`No module named 'sam2.automatic_mask_generator'` = パッケージは在るが
        中身が無い)は不在ではなく破損なので、`.split(".")[0]` のような緩い
        判定はしない。
    """
    return isinstance(error, ModuleNotFoundError) and error.name == module


def _broken_install_error(module: str, error: ImportError) -> ImportError:
    """「導入済みだが import に失敗した」ときのエラーを組む(不在とは別物)。

    素の `ImportError`(= `ModuleNotFoundError` **ではない**)を返すので、CLI の
    既定フォールバックは `_is_absent` の判定でこれを掴まず、そのまま伝播する。
    """
    return ImportError(
        f"{module} is installed but importing it failed: {error}. This is a "
        f"broken {module} installation (ABI/CUDA mismatch, or a missing "
        "transitive dependency), not a missing optional extra, so Atlasmith "
        "will not silently continue with a different segmentation algorithm. "
        f"Repair the environment (`uv sync --reinstall --extra ml` reinstalls "
        "the ML stack), or pass `--segmenter geometric` to run with no ML "
        "dependency at all."
    )


def _import_torch() -> Any:
    """torch を**関数内で** import する(計画v4 §2.1 規約2)。

    Returns:
        torch モジュール。戻り値が `Any` なのは、型注釈に torch の型を書くと
        module 直下 import が要り隔離規約に反するため(`render.py` と同じ判断)。

    Raises:
        ModuleNotFoundError: torch が**未導入**のとき(`name="torch"`)。導入手順と
            代替経路を提示する。CLI の既定経路はこれだけをフォールバック条件に
            する(`_is_absent`)。
        ImportError: torch は導入済みだが import に失敗したとき(破損)。**別の
            メッセージ**で不在と区別でき、フォールバックの対象にはならない。
    """
    try:
        import torch
    except ImportError as e:
        if not _is_absent(e, "torch"):
            raise _broken_install_error("torch", e) from e
        raise ModuleNotFoundError(
            "torch is required by the SAM2 multi-view segmenter but is not "
            "installed. Install the optional extra with `uv sync --extra ml` "
            '(or `pip install "atlasmith[ml]"`), then install a CUDA build of '
            "torch, or use `--segmenter geometric`, which needs no GPU at all.",
            name="torch",
        ) from e
    return torch


def _import_sam2() -> Any:
    """`SAM2AutomaticMaskGenerator` を**関数内で** import する(§2.1 規約2)。

    Returns:
        `SAM2AutomaticMaskGenerator` クラス(モジュールではなくクラスを返す —
        呼び出し側が使うのはこれだけで、`from_pretrained` も classmethod)。

    Raises:
        ModuleNotFoundError: sam2 が**未導入**のとき(`name="sam2"`)。導入手順と
            代替経路を提示する。
        ImportError: sam2 は導入済みだが import に失敗したとき(破損)。詳細は
            `_import_torch` と同じ — `_is_absent` の docstring 参照。
    """
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as e:
        if not _is_absent(e, "sam2"):
            raise _broken_install_error("sam2", e) from e
        raise ModuleNotFoundError(
            "sam2 is required by the SAM2 multi-view segmenter but is not "
            "installed. Install the optional extra with `uv sync --extra ml` "
            '(or `pip install "atlasmith[ml]"`), or use `--segmenter geometric`, '
            "which needs no ML dependencies at all.",
            name="sam2",
        ) from e
    return SAM2AutomaticMaskGenerator


# ---------------------------------------------------------------------------
# パラメータ検証(numpy のみ — 非 ML テストが CI で常時実行する)
# ---------------------------------------------------------------------------


def validate_area_band(area_band: Sequence[float]) -> tuple[float, float]:
    """面積帯 `(lo, hi)` の契約(`0 <= lo < hi`、有限)を検証して返す。

    `hi > 1` は合法(マスクは背景画素も数えるので比が 1 を超えうる)。
    """
    try:
        low, high = (float(v) for v in area_band)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"area_band must be a (lo, hi) pair of numbers, got {area_band!r}"
        ) from e
    if not (np.isfinite(low) and np.isfinite(high)) or not 0.0 <= low < high:
        raise ValueError(f"area_band requires finite 0 <= lo < hi, got ({low}, {high})")
    return low, high


def validate_grid_side(grid_side: int) -> int:
    """シルエット内点グリッドの一辺(1 以上の整数)を検証して返す。"""
    if isinstance(grid_side, bool) or not isinstance(grid_side, (int, np.integer)):
        raise ValueError(f"grid_side must be an int, got {type(grid_side).__name__}")
    if int(grid_side) < 1:
        raise ValueError(f"grid_side must be >= 1, got {grid_side}")
    return int(grid_side)


def validate_channels(channels: Sequence[str]) -> tuple[str, ...]:
    """チャンネル列(非空・既知名のみ・重複なし)を検証してタプルで返す。"""
    if isinstance(channels, str):
        raise ValueError(
            f"channels must be a sequence of channel names, got the string "
            f"{channels!r} (did you mean ({channels!r},)?)"
        )
    names = tuple(channels)
    if not names:
        raise ValueError(f"channels must not be empty; choose from {_CHANNEL_NAMES}")
    for name in names:
        if name not in _CHANNEL_NAMES:
            raise ValueError(
                f"unknown channel {name!r}, expected one of {_CHANNEL_NAMES}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"channels must not contain duplicates, got {names}")
    return names


def _validate_unit_interval(value: float, name: str) -> float:
    """`[0, 1]` の有限 float を検証して返す(AMG の閾値 2 種に使う)。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")
    return result


def _is_below(value: Any, default: int) -> bool:
    """`value` が数値として `default` を下回るか(非数値は False)。

    非数値・bool を False で素通しするのは、型の番人が別に居る(`n_views` は親
    `MultiViewSegmenter.__init__`)ためで、ここで `TypeError` を出すと本来の
    `ValueError` を横取りしてしまうから。
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return False
    return float(value) < float(default)


def _warn_if_below_validated_defaults(n_views: Any, image_size: Any) -> None:
    """検証済み既定を**下回る** `n_views` / `image_size` を警告する(裁定 B-3)。

    **エラーにしない WHY**: 探索や高速化のために下げる自由は残す(反証レビュー
    B-3 の裁定)。ただし「未検証の領域で走っている」ことは必ず告知する — 実測では
    この領域で出力が幾何プライアと完全に同値へ崩れ、**警告 0 件**だった。
    """
    weakened = [
        f"{name}={value} (validated default {default})"
        for name, value, default in (
            ("n_views", n_views, DEFAULT_N_VIEWS),
            ("image_size", image_size, DEFAULT_IMAGE_SIZE),
        )
        if _is_below(value, default)
    ]
    if not weakened:
        return
    warnings.warn(
        "SAM2 multi-view segmentation is configured below its validated "
        "defaults: " + ", ".join(weakened) + ". " + _COLLAPSE_EVIDENCE + " Lower "
        "values are allowed (exploration, speed), but do not read the result as "
        "an ML-quality segmentation.",
        UserWarning,
        stacklevel=3,
    )


def _keyword_names(func: Any) -> frozenset[str]:
    """`func` のキーワード専用引数名を signature から読む(名前表を手書きしない)。

    `**kwargs`(VAR_KEYWORD)は「何でも受ける」印なので名前として数えない —
    それを数えると検疫が常に空振りする。
    """
    return frozenset(
        name
        for name, parameter in inspect.signature(func).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    )


def _validate_segmenter_kwargs(segmenter_kwargs: dict[str, Any]) -> None:
    """親へ透過する `**segmenter_kwargs` の**名前**を検証する(裁定: ロード前)。

    **WHY(2026-08-07 外部レビュー指摘)**: `make_sam2_segmenter` →
    `build_sam2_segmenter` → `MultiViewSegmenter.__init__` の経路は未知名を
    `**kwargs` に抱えたまま運び、`TypeError` が出るのは親へ展開する瞬間 =
    **重みロードの後**である。実測では `merge_threhold`(`merge_threshold` の
    誤記)1 文字で、数百 MB の取得と GPU ロードを終えてから失敗した。名前の
    照合は依存ゼロ・ミリ秒なので、**安価な検証を先に全部済ませる**(この
    ファイルの fail-fast 方針)に揃える。

    候補名の提示に `build_sam2_segmenter` 自身の引数も混ぜるのは、`model_di` の
    ような誤記が(`**segmenter_kwargs` に落ちる以上)ここでしか捕まらないため。

    Raises:
        ValueError: 未知のキーワード名が含まれるとき。近い名前があれば提示する。
    """
    accepted = _keyword_names(MultiViewSegmenter.__init__)
    unknown = sorted(name for name in segmenter_kwargs if name not in accepted)
    if not unknown:
        return
    # 誤記の候補プールは「この呼び出しで書けたはずの名前」全体。
    pool = sorted(accepted | _keyword_names(build_sam2_segmenter))
    reported = []
    for name in unknown:
        close = difflib.get_close_matches(name, pool, n=1)
        suggestion = f" (did you mean {close[0]!r}?)" if close else ""
        reported.append(f"{name!r}{suggestion}")
    raise ValueError(
        "unknown keyword argument(s) for the SAM2 segmenter: "
        + ", ".join(reported)
        + ". Accepted names are: "
        + ", ".join(pool)
        + "."
    )


def _validate_crop_n_layers(crop_n_layers: int) -> int:
    """`crop_n_layers` を検証して返す — **受理するのは 0 だけ**。

    **WHY 0 のみか(2026-08-03 反証レビュー B-1)**: この proposer は視点ごとに
    シルエット内点グリッドを 1 つだけ作って `generator.point_grids = [grid]` と
    差し替える(`Sam2MaskProposer.propose`)。sam2 の AMG は crop レイヤ
    `i` の処理で `point_grids[i]` を引くので、`crop_n_layers >= 1` では
    `point_grids[1]` が存在せず、**重みロードとレンダを終えた数分後に**
    `IndexError: list index out of range` で落ちる(実重みで実測)。レイヤ数ぶんの
    グリッドを複製して通すことは可能だが、そうしない: SAMesh(arXiv 2408.13679)
    自身が `crop_n_layers=0` であり、レイヤが品質に効く証拠が無く、probe でも 0 で
    裁定E の床を越えている。「動くが未検証」を黙って許すより入口で拒否する。
    """
    if isinstance(crop_n_layers, bool) or not isinstance(
        crop_n_layers, (int, np.integer)
    ):
        raise ValueError(
            f"crop_n_layers must be an int, got {type(crop_n_layers).__name__}"
        )
    if int(crop_n_layers) != 0:
        raise ValueError(
            f"crop_n_layers must be 0, got {crop_n_layers}: this proposer swaps in "
            "one silhouette point grid per view, but SAM2 indexes point_grids by "
            "crop layer, so any non-zero value would fail with IndexError deep "
            "inside the mask generator. SAMesh runs with 0 as well and there is no "
            "measured quality benefit from crop layers here, so 0 is the only "
            "supported value."
        )
    return int(crop_n_layers)


# ---------------------------------------------------------------------------
# シルエット内点グリッド(probe 実証済みの方式)
# ---------------------------------------------------------------------------


def silhouette_point_grid(coverage: np.ndarray, grid_side: int) -> np.ndarray:
    """前景画素から `grid_side**2` 点を等間隔サンプリングする(正規化座標)。

    raster 順の前景画素列から等間隔 index で採り、`(x/W, y/H)` の
    `(grid_side**2, 2) float64` を返す(AMG は生成時に `* (W, H)` して画素座標へ
    戻す)。前景画素が点数より少なければ重複するが AMG は許容する。

    Raises:
        ValueError: `coverage` が 2 次元 bool でない、または前景が空のとき
            (呼び出し側 `Sam2MaskProposer.propose` は空シルエットを先に
            K=0 として返すので、ここに空が届くのは契約違反)。
    """
    cover = np.asarray(coverage)
    if cover.ndim != 2 or cover.dtype != np.bool_:
        raise ValueError(
            f"coverage must be a (H, W) bool array, got shape {cover.shape} "
            f"dtype {cover.dtype}"
        )
    side = validate_grid_side(grid_side)
    ys, xs = np.nonzero(cover)
    if xs.size == 0:
        raise ValueError("coverage has no foreground pixel; cannot place seeds")
    n_points = side * side
    selection = np.round(np.linspace(0, xs.size - 1, num=n_points)).astype(np.int64)
    height, width = cover.shape
    return np.stack(
        [xs[selection] / float(width), ys[selection] / float(height)], axis=1
    ).astype(np.float64)


def _uniform_point_grid(grid_side: int) -> np.ndarray:
    """AMG 構築時に渡すダミーグリッド(propose が毎回視点別グリッドへ差し替える)。"""
    side = validate_grid_side(grid_side)
    linear = (np.arange(side, dtype=np.float64) + 0.5) / side
    grid_x, grid_y = np.meshgrid(linear, linear)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def _check_seed_alignment(records: list[dict], coverage: np.ndarray) -> None:
    """`point_grids` 差し替えが効いているかを返却シードの所在で確かめる。

    シードは全点シルエット内から採っているので、返ってきた候補の
    `point_coords` もシルエット内に載るはず。9 割未満なら差し替えが効いて
    いない(sam2 の版が generate 時に `point_grids` 属性を読まなくなった)と
    みなして止める — 黙って一様グリッド相当の品質へ落ちるより、壊れたことを
    知らせて止まる方が安全(probe 実証済みの検査)。

    **空の `records` は「検査できなかった」であって「合格」ではない。** 呼び出し
    側(`Sam2MaskProposer.propose`)は候補が得られるまで検査を保留するので、
    通常ここへ空は届かない。下の早期 return は手書き呼び出し向けの防御であり、
    「検査済み」を意味しない(その判定は呼び出し側にある)。
    """
    if not records:
        return
    height, width = coverage.shape
    points = np.asarray(
        [record["point_coords"][0] for record in records[:_SEED_CHECK_MAX_RECORDS]],
        dtype=np.float64,
    )
    xs = np.clip(np.round(points[:, 0]).astype(np.int64), 0, width - 1)
    ys = np.clip(np.round(points[:, 1]).astype(np.int64), 0, height - 1)
    inside = float(coverage[ys, xs].mean())
    if inside < _SEED_INSIDE_MIN:
        raise RuntimeError(
            f"point_grids swap seems ineffective: only {inside:.0%} of returned "
            "seed points lie inside the silhouette. This sam2 version does not "
            "read the point_grids attribute at generate() time; the proposer "
            "would need to rebuild the generator per view instead."
        )


# ---------------------------------------------------------------------------
# MaskProposer 実装
# ---------------------------------------------------------------------------


class Sam2MaskProposer:
    """`SAM2AutomaticMaskGenerator` を `MaskProposer` 契約へ適合させるアダプタ。

    **所有権**: 構築済みの generator(= モデル重み)を所有し、`close()` で参照を
    手放す(冪等)。本 proposer 自体の所有者は `MultiViewSegmenter` であり、
    そちらの `close()` / `__exit__` がここへ届く(計画v4 §2.1)。

    `propose(view)` の手順(裁定B):

      1. 空シルエットなら K=0 を返す(その視点でメッシュが見えないだけ —
         fusion は未観測として幾何プライアへ劣化できるので、落とすより優雅)。
      2. `coverage` からシルエット内点グリッドを作り、`generator.point_grids`
         を差し替える(効いているかは初回に返却シードの所在で実測検査)。
      3. チャンネルごとに画像を用意して AMG を回す — `"shading"` は
         `view.color` そのまま、`"sdf"` は `set_face_thickness()` で受けた
         面厚みを `face_id` で引いたグレースケール合成。
      4. 全候補に面積帯フィルタを適用し、`(K, H, W) bool` で返す。重複や順序の
         正規化は下流の `fusion.normalize_masks` の責務(ここではしない)。

    非決定性はこの 1 段に閉じる(計画v4 §2.4)— CUDA 推論は「安定」であって
    「決定的」ではないので、下流ゲートは一致率で測る(Step 2-5 ゲート9)。
    """

    __slots__ = (
        "_area_band",
        "_channels",
        "_closed",
        "_generator",
        "_grid_side",
        "_seed_checked",
        "_thickness01",
    )

    def __init__(
        self,
        generator: Any,
        *,
        channels: Sequence[str] = DEFAULT_CHANNELS,
        grid_side: int = DEFAULT_GRID_SIDE,
        area_band: Sequence[float] = DEFAULT_AREA_BAND,
    ) -> None:
        """パラメータを検証し、generator の所有権を引き受ける(fail-fast)。

        Args:
            generator: 構築済みの `SAM2AutomaticMaskGenerator`(またはテスト用の
                互換スタブ)。`point_grids` 属性と `generate(image)` を持つこと。
            channels: AMG へ渡す画像の種類(既定 `("sdf", "shading")`)。
            grid_side: シルエット内点グリッドの一辺。
            area_band: マスク面積 ÷ シルエット面積の許容帯 `(lo, hi)`。

        Raises:
            ValueError: パラメータが契約外、または generator が `point_grids`
                属性を持たないとき(視点別グリッドの差し替えが物理的に不可能)。
        """
        self._channels = validate_channels(channels)
        self._grid_side = validate_grid_side(grid_side)
        self._area_band = validate_area_band(area_band)
        if not hasattr(generator, "point_grids"):
            raise ValueError(
                "generator has no `point_grids` attribute; the per-view "
                "silhouette grid swap cannot work with this object "
                "(expected a SAM2AutomaticMaskGenerator)"
            )
        self._generator = generator
        self._thickness01: np.ndarray | None = None
        self._closed = False
        self._seed_checked = False

    @property
    def channels(self) -> tuple[str, ...]:
        """構成済みのチャンネル列(読み取り専用)。"""
        return self._channels

    def set_face_thickness(self, thickness01: np.ndarray) -> None:
        """`"sdf"` チャンネルが使う面厚み `(M,) [0,1]` を差し替える(裁定A)。

        呼ぶのは `_Sam2MultiViewSegmenter.segment()` — メッシュごとに毎回
        上書きされるので、メッシュ跨ぎの stale な厚みは残らない。防御的に
        コピーして保持する(呼び出し側の配列を後から書き換えられても壊れない)。

        Raises:
            ValueError: 1 次元でない・非有限・[0, 1] 外の値を含むとき。
        """
        array = np.asarray(thickness01, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"thickness01 must have shape (M,), got {array.shape}")
        if array.size and (
            not np.isfinite(array).all()
            or float(array.min()) < 0.0
            or float(array.max()) > 1.0
        ):
            raise ValueError("thickness01 values must be finite and within [0, 1]")
        self._thickness01 = array.copy()

    def propose(self, view: RenderedView) -> np.ndarray:
        """視点 1 つぶんのマスク候補 `(K, H, W) bool` を返す。

        Raises:
            RuntimeError: `close()` 後に呼ばれたとき、`"sdf"` チャンネル構成で
                厚み未設定のとき、または `point_grids` 差し替えが効いていないと
                実測されたとき。
            ValueError: `view` の contract 違反(`thickness_to_image` の検査由来)。
        """
        if self._closed:
            raise RuntimeError(
                "Sam2MaskProposer.propose called after close(); the generator "
                "has been released. Build a new segmenter with "
                "make_sam2_segmenter()."
            )
        coverage = np.asarray(view.coverage)
        height, width = coverage.shape
        silhouette_area = int(np.count_nonzero(coverage))
        if silhouette_area == 0:
            return np.zeros((0, height, width), dtype=bool)
        grid = silhouette_point_grid(coverage, self._grid_side)
        low, high = self._area_band
        collected: list[np.ndarray] = []
        for channel in self._channels:
            image = self._channel_image(view, channel)
            # AMG は generate 時に point_grids 属性を読むだけなので差し替えで
            # 足りる(効いているかは初回の _check_seed_alignment が実測検証)。
            self._generator.point_grids = [grid]
            started = time.perf_counter()
            records = self._generator.generate(np.ascontiguousarray(image))
            LOG.info(
                "sam2 amg (%s): %d candidate(s) in %.1fs",
                channel,
                len(records),
                time.perf_counter() - started,
            )
            # **候補ゼロの結果で「検査済み」にしない**(2026-08-07 外部レビュー
            # 指摘): 検査は返却シードの所在を見るので、`records` が空だと**何も
            # 見ていない**。それで `_seed_checked` を立てると、以降の全視点・全
            # チャンネルで検査が二度と走らなくなる。既定構成 `("sdf", "shading")`
            # では「1 回目(sdf)は候補ゼロ、2 回目(shading)は候補あり」が現実に
            # 起こりうるので、この取りこぼしは仮定ではなく実在の穴だった。
            # 検査は「`point_grids` 差し替えが効いているか」の唯一の警報装置
            # なので、**実際に候補が得られるまで保留する**。
            if records and not self._seed_checked:
                _check_seed_alignment(records, coverage)
                self._seed_checked = True
            for record in records:
                segmentation = np.asarray(record["segmentation"], dtype=bool)
                ratio = int(np.count_nonzero(segmentation)) / silhouette_area
                if low <= ratio <= high:
                    collected.append(segmentation)
        if not collected:
            return np.zeros((0, height, width), dtype=bool)
        return np.stack(collected)

    def _channel_image(self, view: RenderedView, channel: str) -> np.ndarray:
        """チャンネル名を AMG 入力画像 `(H, W, 3) uint8` へ写す。"""
        if channel == "shading":
            return np.asarray(view.color)
        # channel == "sdf"(validate_channels 済みなので他は来ない)。
        if self._thickness01 is None:
            raise RuntimeError(
                "the 'sdf' channel needs per-mesh face thickness, but "
                "set_face_thickness() has not been called. "
                "make_sam2_segmenter() wires this automatically; a hand-built "
                "Sam2MaskProposer must call it before propose()."
            )
        return thickness_module.thickness_to_image(view.face_id, self._thickness01)

    def close(self) -> None:
        """generator への参照を手放す。**冪等**(2 回目以降は無操作)。

        重み(GPU メモリ数 GB)の実解放は参照を失った torch / GC に委ねる —
        明示的な `torch.cuda.empty_cache()` はプロセス全体の CUDA 状態に触る
        副作用であり、ライブラリ側からは行わない。
        """
        if self._closed:
            return
        self._closed = True
        self._generator = None


# ---------------------------------------------------------------------------
# MultiViewSegmenter サブクラス(裁定A — SDF の per-mesh 配管)
# ---------------------------------------------------------------------------


class _Sam2MultiViewSegmenter(MultiViewSegmenter):
    """`segment()` の冒頭で面厚みを proposer へ渡す `MultiViewSegmenter`。

    **WHY サブクラスか(裁定A)**: `MaskProposer.propose(view)` は per-view の
    契約で mesh を見られないが、SDF(厚み)は per-mesh の量。`MultiViewSegmenter`
    本体へ prepare フックを足す案は却下済み(Step 2-4 のレビュー済みコードを
    凍結する)。よって mesh を最初に見る `segment()` をオーバーライドし、
    厚み計算 → `set_face_thickness` → `super().segment()` の順で流す。

    戻り値・寿命・エラー契約はすべて親と同一(呼び出し側から見た型は
    `MultiViewSegmenter` — 計画v4 §4.2 の契約)。
    """

    __slots__ = ("_needs_thickness", "_sam2_proposer")

    def __init__(
        self,
        proposer: Any,
        renderer_factory: Callable[[MeshData], MeshRenderer],
        *,
        needs_thickness: bool,
        **segmenter_kwargs: Any,
    ) -> None:
        """親の検証をそのまま通し、厚み配管の有無だけを追加で覚える。

        Args:
            proposer: `MaskProposer` 契約 + (`needs_thickness` のとき)
                `set_face_thickness` を持つ提案器。親と同じく**所有する**。
            renderer_factory: 親と同じ(`segment()` ごとに `with` で生成・破棄)。
            needs_thickness: `"sdf"` チャンネルが構成されているか。False なら
                厚み計算を丸ごとスキップする(`channels=("shading",)` 構成で
                20 万面のレイキャストを無駄に払わない)。
            **segmenter_kwargs: 親 `MultiViewSegmenter.__init__` へそのまま
                渡す(検証も親が行う — 述語を 2 箇所に書かない)。
        """
        super().__init__(proposer, renderer_factory, **segmenter_kwargs)
        self._sam2_proposer = proposer
        self._needs_thickness = bool(needs_thickness)

    def segment(self, mesh: MeshData) -> np.ndarray:
        """面厚みを計算して proposer へ渡してから、親の segment を実行する。

        親が P==1(単一部位)を返したら `UserWarning` を出す(裁定 B-3)。
        **再計算はしない** — 崩壊を告知するだけで、コストを二重に払わない。
        文面には**この実行の `n_views`** と検証済み既定の実測値を両方載せる。

        Warns:
            UserWarning: 結果が単一部位で、幾何フォールバックと区別できないとき。
        """
        # 閉鎖済みなら親に即 RuntimeError を出させる(数十秒かかりうる厚み計算を
        # 「どうせ失敗する呼び出し」のために払わない)。`_closed` は親の内部状態
        # だが、同一クラス階層内の読み取りであり契約(閉鎖後 segment は
        # RuntimeError)は親 docstring が公開している。
        if not self._closed and self._needs_thickness:
            faces = np.asarray(mesh.faces)
            if faces.shape[0] > 0:
                thickness01 = thickness_module.compute_face_thickness01(
                    mesh.vertices, mesh.faces
                )
                self._sam2_proposer.set_face_thickness(thickness01)
        labels = super().segment(mesh)
        # 親の `total_masks == 0` ガードは「1 枚もマスクが出なかった」経路しか
        # 捉えない。マスクは出たが融合後に 1 部位へ潰れた場合(実測: n_views=8 /
        # image_size=512 で accuracy 0.5000 / P=1 / 警告 0 件)も、出力は幾何
        # プライアと区別が付かない。**本体は凍結**なのでサブクラス側で告知する。
        if labels.size and int(labels.max()) == 0:
            # 現在値を文面に入れる(B-3): 「既定は 24」だけでは、実際にいくつで
            # 走ったのかが分からず原因追跡が 1 往復増える。`image_size` は
            # renderer_factory が閉じ込める値でここからは見えない(`__init__.py`
            # の DEFAULT_IMAGE_SIZE の WHY コメント)ので、見えないことを明示する。
            warnings.warn(
                "the SAM2 backend produced a single part (P=1) for this mesh "
                f"(this run: n_views={self.n_views}; image_size is baked into "
                "the injected renderer_factory and is not visible here): every "
                "face carries the same label, which is indistinguishable from "
                "the geometric fallback. " + _COLLAPSE_EVIDENCE + " If one part "
                "is the intended answer, select the geometric backend "
                "explicitly with `--segmenter geometric`.",
                UserWarning,
                stacklevel=2,
            )
        return labels


# ---------------------------------------------------------------------------
# 組み立て(`make_sam2_segmenter` の実体)
# ---------------------------------------------------------------------------


def _resolve_device(device: str | None) -> str:
    """デバイスを確定する(計画v4 §2.4.4)。

    `None`(既定)なら `torch.cuda.is_available()` で cuda / cpu を選び、cpu へ
    落ちるときは **UserWarning** で所要時間の桁を知らせる。明示指定されたときは
    警告しない(意図的な選択を尊重する)。
    """
    torch = _import_torch()
    if device is not None:
        if not isinstance(device, str) or not device:
            raise ValueError(f"device must be a non-empty string, got {device!r}")
        return device
    if bool(torch.cuda.is_available()):
        return "cuda"
    warnings.warn(
        "torch.cuda.is_available() is False: SAM2 automatic mask generation "
        "will run on the CPU, which is orders of magnitude slower (minutes per "
        "view instead of seconds). Install a CUDA build of torch, or use "
        "`--segmenter geometric`.",
        UserWarning,
        stacklevel=3,
    )
    return "cpu"


def build_sam2_segmenter(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    pred_iou_thresh: float = DEFAULT_PRED_IOU_THRESH,
    stability_score_thresh: float = DEFAULT_STABILITY_SCORE_THRESH,
    crop_n_layers: int = DEFAULT_CROP_N_LAYERS,
    grid_side: int = DEFAULT_GRID_SIDE,
    area_band: Sequence[float] = DEFAULT_AREA_BAND,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    shading: str = DEFAULT_SHADING,
    **segmenter_kwargs: Any,
) -> MultiViewSegmenter:
    """SAM2 を結線した `MultiViewSegmenter`(context manager)を組み立てる。

    公開入口は `atlasmith.segmentation.multiview.make_sam2_segmenter`(そちらが
    本関数へ委譲する)。パラメータの意味と既定値の実測根拠は module docstring と
    各 `DEFAULT_*` のコメントを参照。

    Args:
        model_id: HuggingFace のモデル ID。初回は重みの DL が走る(数百 MB)。
        device: `None` で自動(cuda → cpu + 警告)。明示指定は警告なし。
        pred_iou_thresh: AMG の予測 IoU 下限([0, 1])。
        stability_score_thresh: AMG の stability 下限([0, 1])。
        crop_n_layers: AMG のクロップ階層数。**0 のみ**(WHY は
            `_validate_crop_n_layers` — 非 0 は AMG 内部で IndexError になる)。
        grid_side: シルエット内点グリッドの一辺(点数はその 2 乗)。
        area_band: マスク面積 ÷ シルエット面積の許容帯 `(lo, hi)`。
        channels: AMG へ渡す画像種(`("sdf", "shading")` / `("sdf",)` 等)。
        image_size: レンダ解像度(renderer_factory へ焼き込む)。既定を**下回る**と
            警告(下記 Warns)。
        shading: 色付けモード(同上)。テクスチャ無しメッシュでは renderer が
            `"normal"` へ自動で落ちる(黙って変えない — logging に出る)。
        **segmenter_kwargs: `MultiViewSegmenter.__init__` へ透過(`n_views` /
            `merge_threshold` 等。検証も親が行う)。`n_views` は既定を**下回る**と
            警告(下記 Warns)。

    Returns:
        `MultiViewSegmenter` 互換のインスタンス(計画v4 §4.2 の契約)。
        proposer を所有するので、使い終わったら `with` か `close()` で閉じること。

    Warns:
        UserWarning: `device` 自動解決が cpu へ落ちたとき、または `n_views` /
            `image_size` が検証済み既定を下回るとき(2026-08-03 反証レビュー B-3)。

    Raises:
        ModuleNotFoundError: torch / sam2 が**未導入**のとき(導入手順つき)。
        ImportError: torch / sam2 は導入済みだが import に失敗したとき(破損 —
            不在とは別メッセージで、CLI の既定フォールバック対象にならない)。
        ValueError: いずれかのパラメータが契約外のとき、または
            `**segmenter_kwargs` に未知のキーワード名が含まれるとき。
    """
    # --- 0. **名前**の検疫(重みロードより前 — `_validate_segmenter_kwargs` の WHY)。
    _validate_segmenter_kwargs(segmenter_kwargs)

    # --- 1. 安価な検証を最初に全部済ませる(fail-fast)。数百 MB の重みロードの
    # 後に typo の ValueError を出さない。image_size / shading の値の番人は
    # `_ModernglRenderer.__init__` だが、それが走るのは最初の `segment()`
    # (= ロード後、数分後)なので、ここでは **render 側の定数を再利用して**
    # 同じ述語を前倒しする(値のリストを 2 箇所に書かない)。
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"model_id must be a non-empty string, got {model_id!r}")
    pred_iou = _validate_unit_interval(pred_iou_thresh, "pred_iou_thresh")
    stability = _validate_unit_interval(
        stability_score_thresh, "stability_score_thresh"
    )
    crop_layers = _validate_crop_n_layers(crop_n_layers)
    side = validate_grid_side(grid_side)
    band = validate_area_band(area_band)
    channel_names = validate_channels(channels)

    # WHY 関数内 import: `render` は隔離モジュール1。module 自体は numpy のみ
    # だが、隔離ファイルを触る側は常に関数内 import とする規約
    # (`multiview/__init__.py` docstring)に従う。
    from atlasmith.segmentation.multiview.render import MIN_IMAGE_SIZE, SHADING_MODES

    if isinstance(image_size, bool) or not isinstance(image_size, (int, np.integer)):
        raise ValueError(f"image_size must be an int, got {type(image_size).__name__}")
    if int(image_size) < MIN_IMAGE_SIZE:
        raise ValueError(f"image_size must be >= {MIN_IMAGE_SIZE}, got {image_size}")
    if shading not in SHADING_MODES:
        raise ValueError(
            f"unknown shading {shading!r}, expected one of {sorted(SHADING_MODES)}"
        )
    size = int(image_size)
    # 崩壊領域の告知は**重みロードより前**に出す(数分待ってから「その構成は
    # 未検証」と言われても遅い)。`n_views` の型/範囲の番人は親なのでここでは見ない。
    _warn_if_below_validated_defaults(
        segmenter_kwargs.get("n_views", DEFAULT_N_VIEWS), size
    )

    # --- 2. デバイス確定(torch import)→ モデル構築(sam2 import + 重み)。
    resolved_device = _resolve_device(device)
    generator_cls = _import_sam2()
    started = time.perf_counter()
    generator = generator_cls.from_pretrained(
        model_id,
        points_per_side=None,
        point_grids=[_uniform_point_grid(side)],
        pred_iou_thresh=pred_iou,
        stability_score_thresh=stability,
        crop_n_layers=crop_layers,
        device=resolved_device,
    )
    LOG.info(
        "sam2 model %s loaded on %s in %.1fs (includes the HF download when uncached)",
        model_id,
        resolved_device,
        time.perf_counter() - started,
    )

    proposer = Sam2MaskProposer(
        generator, channels=channel_names, grid_side=side, area_band=band
    )

    def renderer_factory(mesh: MeshData) -> MeshRenderer:
        # WHY 関数内 import: 上の MIN_IMAGE_SIZE と同じ規約。moderngl 自体は
        # `_ModernglRenderer.__enter__` まで import されない。
        from atlasmith.segmentation.multiview.render import _ModernglRenderer

        return _ModernglRenderer(mesh, image_size=size, shading=shading)

    # --- 3. サブクラス組み立て。親の検証(segmenter_kwargs)が落ちたら、
    # ここまでに確保した重みを持つ proposer を閉じてから伝播する(リーク防止)。
    try:
        return _Sam2MultiViewSegmenter(
            proposer,
            renderer_factory,
            needs_thickness="sdf" in channel_names,
            **segmenter_kwargs,
        )
    except BaseException:
        proposer.close()
        raise
