"""マスク→面割当と2パス視点間融合(計画v4 §2.4.4 / §2.4.5、numpy のみ・決定的)。

責務は3つ:

  1. **マスク正規化**(`normalize_masks`)— 空マスク除去・完全一致の重複除去・
     **全順序キー**による整列。ここから下流はマスク**内容**だけに依存する。
  2. **段階A: 視点内の面割当**(`fold_masks_to_label_map` / `assign_view_faces`)—
     マスク集合を画素単位の単一ラベル図へ畳み、面の可視画素上の最頻値を採る。
  3. **段階B〜E: 視点間融合**(`edge_vote_statistics` / `fuse_view_segments`)—
     面隣接の辺ごとに視点票を集め、**2パス union** でラベルへ落とす。

**このモジュールは決定的である**: RNG を使わず、整列と union-find の tie-break
だけで結果が決まる。パイプライン中で非決定的なのは `MaskProposer`(SAM2)の
1段だけで、その出力を渡された後の処理はすべてここに閉じている。

**共有部品を再利用する**(計画v4 §2.3 末尾「重複実装しない」): weld・面隣接・
幾何プライアは `segmentation.adjacency`、union-find・小部位マージ・正規化 relabel・
ラベル契約検証は `segmentation.labels` のものをそのまま呼ぶ。特に幾何プライアは
`smooth_edge_mask` 1 本に閉じている(二面角しきい値と零面積カットは組でないと
正しくないため)。

**既知の限界 — 部分遮蔽領域の過分割**(計画v4 §0-A 条件12・2026-07-29 記録):
段階C を 2 パスにした帰結として、**「観測面どうしを結ぶ未観測辺」は恒久カット**に
なる。パス2 が救うのは未観測*面*だけで、両端がすでに観測されている辺には幾何
プライアの橋を掛けない。したがって

  - 意味境界の一部で票が割れてカットされた領域が、別の未観測辺で再結合される
    事故は起きない(v3 の1パス設計の欠陥。§2.4.5 の BL-2)が、
  - **部分遮蔽で votes が割れる滑らかな面は過分割されうる**(v3 では `w=1.0` の
    幾何プライアが偶然埋めていた)。

緩和策は段階D の小部位マージだけで、これを検出するゲートも無い。実アセットでの
観測は Step 2-7 以降の課題として残っている。**視点数を増やす**(未観測辺を減らす)
か `min_votes` を上げる(票の少ない辺を幾何プライア側へ回す)のが運用上の逃げ道。

**既知の性質 — 出力は決定的だが「面の並び」に依存する**(2026-07-30 反証レビュー
N6 の実測記録。挙動は仕様どおりなので変えていない):

  - 同一入力に対する出力はビット決定的(RNG 不使用)。
  - しかし**面 index を置換した同一形状**では分割が変わりうる。実測: `_two_pass_union`
    のみを面順の無作為置換 120 通りで回すと **42 通りで分割が変わった**。原因は全件
    パス2-2 の「`prior=True` 境界辺数」が同点で、tie-break が「最小の観測成分代表
    index」= 面 index に落ちるため(§2.4.5 が指定する決定的規約そのもの)。
  - **`DihedralSegmenter` にはこの性質が無い**(同じ置換 120 通りで差 0)。幾何だけの
    連結成分は面順に依存しないため。つまり**面順依存は多視点融合が新しく持ち込んだ
    性質**であり、実用上は「エクスポータが面順を変えると部位割りが変わりうる」。
    形状が同じでも面順が違えば結果が違いうる、という点を利用者へ伝える必要がある。

依存方向(計画v4 §2.1): `segmentation` 配下は **numpy のみ**。trimesh / xatlas /
PIL / torch / moderngl / sam2 と `atlasmith.io` / `atlasmith.pack` /
`atlasmith.bake` は、module 直下・関数内のいずれでも import しない。
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from atlasmith.segmentation.adjacency import (
    build_face_adjacency,
    smooth_edge_mask,
    validate_angle_deg,
    weld_vertices,
)

# **private 定数を import している WHY**: `min_faces=None` の自動値は
# `DihedralSegmenter` と**ビット単位で同一**でなければならない — 計画v4 §5
# Step 2-4 ゲート11 が「全マスク未割当の入力で出力が `DihedralSegmenter` と
# `np.array_equal`」を要求しており、既定 `min_faces` の解決式がずれるとその等価性が
# 静かに壊れる。値を写して 2 箇所に持つと片方だけが変更されうるので、定義元を
# 参照する(`geometric.py` は本ステップの禁止パスなので公開名に昇格させられない)。
from atlasmith.segmentation.geometric import (
    _MIN_FACES_AUTO_DIVISOR,
    _MIN_FACES_AUTO_FLOOR,
)
from atlasmith.segmentation.labels import (
    UnionFind,
    merge_small_parts,
    normalize_labels,
    validate_labels,
)
from atlasmith.segmentation.multiview.faceid import validate_coverage_consistency

__all__ = [
    "DEFAULT_ANGLE_DEG",
    "DEFAULT_ASSIGNED_WARN",
    "DEFAULT_ASSIGN_RATIO",
    "DEFAULT_MAX_MASKS_PER_VIEW",
    "DEFAULT_MERGE_THRESHOLD",
    "DEFAULT_MIN_VOTES",
    "DEFAULT_VISIBLE_WARN",
    "UNASSIGNED",
    "EdgeVotes",
    "ViewAssignment",
    "assign_view_faces",
    "edge_vote_statistics",
    "fold_masks_to_label_map",
    "fuse_view_segments",
    "normalize_masks",
    "resolve_min_faces",
    "validate_assign_ratio",
    "validate_max_masks_per_view",
    "validate_merge_threshold",
    "validate_min_faces",
    "validate_min_votes",
    "validate_warn_ratio",
]

# 「どのマスクにも属さない / 割当なし」を表す番兵。画素図 `seg` と面ごとの
# `view_segment` で同じ値を使う(面IDの背景 -1 とも一致していて読みやすい)。
UNASSIGNED = -1

# 既定パラメータ(計画v4 §2.4.5 末尾 + 2026-07-29 オーケストレーター裁定3)。
# `angle_deg` は `DihedralSegmenter` の既定と同値 — 幾何プライアの供給元が
# 同じ判定を使うので、ここだけ別の値にすると劣化経路の意味が変わる。
DEFAULT_ANGLE_DEG = 60.0
DEFAULT_ASSIGN_RATIO = 0.5
# **既定 2 の WHY**(計画v4 §2.4.5): 1 視点でしか観測されない辺は `w in {0, 1}` の
# 二値になり、その 1 視点が誤ると確定してしまう。2 なら 1 票の辺は「未観測」として
# パス2(幾何プライア)へ回る。
DEFAULT_MIN_VOTES = 2
DEFAULT_MERGE_THRESHOLD = 0.5
DEFAULT_VISIBLE_WARN = 0.95
DEFAULT_ASSIGNED_WARN = 0.90
DEFAULT_MAX_MASKS_PER_VIEW = 512


class ViewAssignment(NamedTuple):
    """段階A の出力(1 視点ぶん)。

    フィールド:
        segment: `(M,) int32`。視点内のマスク index、割当なしは `UNASSIGNED`。
            **値はその視点内でのみ意味を持つ**(視点間でマスク index の対応は無い)。
        visible: `(M,) bool`。その視点の面IDバッファに 1 画素以上現れた面。

    `visible` を `segment >= 0` と**別に返す WHY**: 被覆率の警告 2 種
    (`visible_ratio` / `assigned_ratio`)を独立に判定するため。片方だけを持つと
    「見えていないから割当が無い」と「見えているがマスクが覆わない」を区別できず、
    どちらの警告も同時に出て原因(視点数 vs マスク粒度)の切り分けができない。
    """

    segment: np.ndarray
    visible: np.ndarray


class EdgeVotes(NamedTuple):
    """段階B の出力(面隣接の辺ごと)。

    フィールド:
        votes: `(E,) int64`。両端の面がともに割当済みだった視点数。
        agree: `(E,) int64`。さらに同一マスクだった視点数。
        weight: `(E,) float64`。`agree / votes`。**未観測辺(`votes < min_votes`)は
            `nan`** — 「定義されていない」ことを値で表す(計画v4 §2.4.5 は
            `votes >= min_votes` のときのみ `w` を定義する)。
        observed: `(E,) bool`。`votes >= min_votes`。

    `weight` を `nan` にしてある WHY: 0.0 で埋めると未観測辺が「全視点が不一致」と
    区別できなくなり、`w < merge_threshold` の判定に紛れ込んで**恒久カット**として
    扱われてしまう(パス2 の幾何プライア経路に回らなくなる)。`nan` は比較が常に
    False なので、マスクを忘れた実装は「繋がらない」側へ倒れる。
    """

    votes: np.ndarray
    agree: np.ndarray
    weight: np.ndarray
    observed: np.ndarray


def validate_assign_ratio(assign_ratio: float) -> float:
    """割当しきい値の契約(有限かつ `0 < assign_ratio <= 1`)を検証して返す。

    **下限を開区間にする WHY**: `0` を許すと「可視画素が 1 画素でもマスクに触れて
    いれば割当」になり、最頻値の要求(段階A-3)と噛み合わない。上限は閉 —
    `1.0` は「面が完全に 1 つのマスクに含まれるときだけ割当」という有効な設定。

    Args:
        assign_ratio: 検証する比率。

    Returns:
        `float` へ正規化した値。

    Raises:
        ValueError: 非有限、または `(0, 1]` の範囲外のとき。
    """
    ratio = float(assign_ratio)
    if not np.isfinite(ratio):
        raise ValueError(f"assign_ratio must be finite, got {assign_ratio!r}")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"assign_ratio must be in (0, 1], got {ratio}")
    return ratio


def validate_merge_threshold(merge_threshold: float) -> float:
    """結合しきい値の契約(有限かつ `0 <= merge_threshold <= 1`)を検証して返す。

    両端とも閉。`0` は「観測辺は一致票が 1 票も無くても繋ぐ」= 幾何を無視して
    観測辺を全部繋ぐ設定、`1` は「全視点が一致した辺だけ繋ぐ」設定で、どちらも
    有効な極端値。判定は `w >= merge_threshold`(**閉**、計画v4 §5 ゲート8)。

    Args:
        merge_threshold: 検証するしきい値。

    Returns:
        `float` へ正規化した値。

    Raises:
        ValueError: 非有限、または `[0, 1]` の範囲外のとき。
    """
    threshold = float(merge_threshold)
    if not np.isfinite(threshold):
        raise ValueError(f"merge_threshold must be finite, got {merge_threshold!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"merge_threshold must be in [0, 1], got {threshold}")
    return threshold


def validate_min_votes(min_votes: int) -> int:
    """`min_votes` の契約(整数かつ 1 以上)を検証して返す。

    `0` を許さない WHY: `votes >= 0` は常に真なので、票が 1 票も無い辺まで
    「観測辺」になり `w = 0/0` が発生する。未観測をパス2 へ回す設計が壊れる。

    Args:
        min_votes: 検証する最小票数。

    Returns:
        `int` へ正規化した値。

    Raises:
        ValueError: 整数でない(bool を含む)、または 1 未満のとき。
    """
    if isinstance(min_votes, bool) or not isinstance(min_votes, (int, np.integer)):
        raise ValueError(f"min_votes must be an int, got {type(min_votes).__name__}")
    if int(min_votes) < 1:
        raise ValueError(f"min_votes must be >= 1, got {min_votes}")
    return int(min_votes)


def validate_min_faces(min_faces: int | None) -> int | None:
    """`min_faces` の契約(`None` か 1 以上の整数)を検証して返す。

    `DihedralSegmenter.__init__`(`geometric.py`)と**同じ述語**。値の解決は
    `resolve_min_faces` が行う。

    Args:
        min_faces: 検証する値。`None` は「面数から自動決定」。

    Returns:
        `None` または `int`。

    Raises:
        ValueError: 整数でない(bool を含む)、または 1 未満のとき。
    """
    if min_faces is None:
        return None
    if isinstance(min_faces, bool) or not isinstance(min_faces, (int, np.integer)):
        raise ValueError(
            f"min_faces must be None or an int, got {type(min_faces).__name__}"
        )
    if int(min_faces) < 1:
        raise ValueError(f"min_faces must be >= 1, got {min_faces}")
    return int(min_faces)


def validate_warn_ratio(value: float, name: str) -> float:
    """被覆率警告のしきい値(有限かつ `0 <= value <= 1`)を検証して返す。

    `visible_warn` / `assigned_warn` の共用。`0` は「警告しない」、`1` は
    「1 面でも欠けたら警告する」で、どちらも有効な設定。

    Args:
        value: 検証するしきい値。
        name: エラーメッセージに出すパラメータ名。

    Returns:
        `float` へ正規化した値。

    Raises:
        ValueError: 非有限、または `[0, 1]` の範囲外のとき。
    """
    ratio = float(value)
    if not np.isfinite(ratio):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {ratio}")
    return ratio


def validate_max_masks_per_view(max_masks_per_view: int) -> int:
    """`max_masks_per_view` の契約(整数かつ 1 以上)を検証して返す。

    Args:
        max_masks_per_view: 検証する上限。

    Returns:
        `int` へ正規化した値。

    Raises:
        ValueError: 整数でない(bool を含む)、または 1 未満のとき。
    """
    if isinstance(max_masks_per_view, bool) or not isinstance(
        max_masks_per_view, (int, np.integer)
    ):
        raise ValueError(
            "max_masks_per_view must be an int, got "
            f"{type(max_masks_per_view).__name__}"
        )
    if int(max_masks_per_view) < 1:
        raise ValueError(f"max_masks_per_view must be >= 1, got {max_masks_per_view}")
    return int(max_masks_per_view)


def resolve_min_faces(min_faces: int | None, n_faces: int) -> int:
    """`min_faces=None` を面数から自動解決する(`DihedralSegmenter` と同一の式)。

    Args:
        min_faces: `validate_min_faces` を通した値。
        n_faces: メッシュの面数 M。

    Returns:
        小部位マージに渡す実効 `min_faces`。
    """
    if min_faces is not None:
        return min_faces
    return max(_MIN_FACES_AUTO_FLOOR, n_faces // _MIN_FACES_AUTO_DIVISOR)


def normalize_masks(
    masks: np.ndarray,
    *,
    max_masks_per_view: int = DEFAULT_MAX_MASKS_PER_VIEW,
    context: str = "masks",
) -> np.ndarray:
    """マスク集合を決定的な正規形へ整える(計画v4 §2.4.4・**決定性の要**)。

    手順:
      1. 空マスク(True が 0 画素)を除去する。
      2. **完全一致**のマスクを重複除去する(最初に現れた 1 枚を残す)。
      3. **全順序キー** `(-pixel_count, first_true_flat_index,
         np.packbits(mask.ravel()).tobytes())` で昇順に並べる。

    **第3キーが要る WHY【BL-8】**: `(-count, first_index)` の 2 キーは全順序では
    ない。反例は 8x8 上の `{(0,0),(0,1)}` と `{(0,0),(0,2)}` — どちらも
    `(-2, 0)` になる。安定ソートは同キーの入力順を保存するので、同じマスク集合を
    逆順に渡すと結果が変わる。内容そのもののバイト列を第3キーに置けば、集合が
    同じなら並びも同じになる(= 提案器がどの順で返しても下流が変わらない)。

    並び順の帰結: **index 0 が最大面積**で、以降小さくなる。`fold_masks_to_label_map`
    はこの順に描き込むことで「小さいマスクが上書きする」規約(§2.4.5 段階A-1)を
    実現するので、**この関数を通さないマスクを下流へ渡してはならない**。

    Args:
        masks: `(K, H, W) bool`。**書き換えない**(新しい配列を返す)。
        max_masks_per_view: 提案器が返してよいマスク数の上限。超過は `ValueError`
            (**空マスク除去・重複除去の前**、つまり生の K で判定する — 提案器が
            出しすぎたことを知らせるガードだから)。
        context: エラーメッセージに出す文脈(例 `"view 3"`)。

    Returns:
        `(K', H, W) bool` の新しい配列(`K' <= K`)。

    Raises:
        ValueError: shape が `(K, H, W)` でない、dtype が bool でない、または
            `K > max_masks_per_view` のとき。
    """
    limit = validate_max_masks_per_view(max_masks_per_view)
    array = np.asarray(masks)
    if array.ndim != 3:
        raise ValueError(
            f"{context}: masks must have shape (K, H, W), got {array.shape}"
        )
    if array.dtype != np.bool_:
        raise ValueError(f"{context}: masks must have dtype bool, got {array.dtype}")
    n_masks = int(array.shape[0])
    if n_masks > limit:
        raise ValueError(
            f"{context}: the mask proposer returned {n_masks} masks, which exceeds "
            f"max_masks_per_view={limit}. Raise `min_mask_region_area` (or lower "
            "`points_per_side`) so the proposer emits fewer, larger masks; the "
            "limit is never silently truncated."
        )

    empty_result = np.zeros((0, *array.shape[1:]), dtype=bool)
    if n_masks == 0:
        # **`reshape(0, -1)` を踏ませない**: numpy は要素数 0 の配列で `-1` を
        # 解決できず `ValueError: cannot reshape array of size 0 into shape
        # (0,newaxis)` になる。K==0(提案器が 1 枚も返さなかった)は §2.6 が
        # 想定する正常系なので、ここで先に返す。
        return empty_result

    flat = array.reshape(n_masks, -1)
    pixel_counts = flat.sum(axis=1, dtype=np.int64)
    kept = np.flatnonzero(pixel_counts > 0)
    if kept.size == 0:
        return empty_result

    kept_flat = flat[kept]
    # `argmax` は bool 配列で「最初の True」を返す(空マスクは既に除いてある)。
    first_true = np.argmax(kept_flat, axis=1)
    # 内容そのものを比較可能な 1 本のバイト列へ。長さ固定なので packbits の
    # 末尾パディングも決定的で、`bytes` の比較は「同一マスク <=> 同一キー」。
    packed = np.packbits(kept_flat, axis=1)
    content = [packed[index].tobytes() for index in range(kept.size)]
    order = sorted(
        range(kept.size),
        key=lambda index: (
            -int(pixel_counts[kept[index]]),
            int(first_true[index]),
            content[index],
        ),
    )
    # 完全一致マスクは 3 キーがすべて同値なので整列後に必ず隣接する。
    unique: list[int] = []
    previous: bytes | None = None
    for index in order:
        if content[index] == previous:
            continue
        unique.append(int(kept[index]))
        previous = content[index]
    return array[unique]


def fold_masks_to_label_map(masks: np.ndarray) -> np.ndarray:
    """マスク集合を画素単位の単一ラベル図 `seg (H, W) int32` へ畳む(段階A-1)。

    **重なりの解決規約**: 面積の大きいマスクから順に描き込み、**小さい(= 細かい)
    マスクが上書きする**。SAM2 の自動マスクは入れ子(全身/胴/腕)になり得るので、
    細かい方を採るのが「部位」の意図に近い(計画v4 §2.4.5 段階A-1)。

    **前提**: `masks` は `normalize_masks` の戻り値であること。面積降順という
    並びはそこで確立され、本関数は**並べ替えを行わない**(index をラベル値として
    使うので、ここで並べ替えると `normalize_masks` の全順序と食い違う)。

    Args:
        masks: `(K, H, W) bool`(面積降順)。書き換えない。

    Returns:
        `(H, W) int32`。値はマスク index、どのマスクにも属さない画素は `UNASSIGNED`。
        **`int32` にしてある WHY**: int16 だと K の上限が dtype として暗黙に
        決まってしまう(計画v4 §2.4.5)。

    Raises:
        ValueError: shape が `(K, H, W)` でない、または dtype が bool でないとき。
    """
    array = np.asarray(masks)
    if array.ndim != 3:
        raise ValueError(f"masks must have shape (K, H, W), got {array.shape}")
    if array.dtype != np.bool_:
        raise ValueError(f"masks must have dtype bool, got {array.dtype}")
    label_map = np.full(array.shape[1:], UNASSIGNED, dtype=np.int32)
    for index in range(array.shape[0]):
        label_map[array[index]] = index
    return label_map


def assign_view_faces(
    masks: np.ndarray,
    face_id: np.ndarray,
    coverage: np.ndarray,
    *,
    n_faces: int,
    assign_ratio: float = DEFAULT_ASSIGN_RATIO,
    context: str = "view",
) -> ViewAssignment:
    """1 視点のマスクを面ごとの割当へ落とす(段階A、計画v4 §2.4.5)。

    面 f の**可視画素**(`coverage & (face_id == f)`)上で `seg` の最頻値を求め、
    それが `UNASSIGNED` でなく、かつ占有率が `assign_ratio` **以上**なら割当。

    **`seg == UNASSIGNED` も最頻値の候補に含める WHY**(計画v4 §2.4.5 段階A-2):
    除外すると、面の 90% がどのマスクにも属さないのに残り 10% の細片マスクが
    勝ってしまう。含めれば `assign_ratio` の分母が「可視画素数」になり、
    「面のどれだけがそのマスクに覆われているか」という定義が一貫する。

    **数値契約(NEP 50 対応)**: 背景画素を除いたうえで
    `key = face_id * (K + 1) + (seg + 1)` を **int64 で**組み、`np.unique` で
    数え上げる。uint8/int32 のまま掛けると numpy 2.x では溢れが `OverflowError` か
    無言の切り詰めになる。tie(同数)は**小さい `seg` を優先**する。

    Args:
        masks: `normalize_masks` を通した `(K, H, W) bool`。書き換えない。
        face_id: `(H, W)` の整数配列。背景は負(`RenderedView` 契約では -1)。
        coverage: `(H, W) bool`。前景 = True。
        n_faces: メッシュの面数 M(戻り値の長さ)。
        assign_ratio: 占有率のしきい値。境界は**閉**(`占有率 >= assign_ratio`)。
        context: エラーメッセージに出す文脈(例 `"view 3"`)。

    Returns:
        `ViewAssignment(segment (M,) int32, visible (M,) bool)`。

    Raises:
        ValueError: shape/dtype が契約と違う、`n_faces` が負、`face_id` が
            `n_faces` 以上の面を指す、または `coverage <=> (face_id >= 0)` が
            破れているとき(§2.6 の「ディザ/ブレンド事故の検出」)。
    """
    ratio_floor = validate_assign_ratio(assign_ratio)
    if int(n_faces) < 0:
        raise ValueError(f"n_faces must be non-negative, got {n_faces}")
    n_faces = int(n_faces)

    mask_array = np.asarray(masks)
    ids = np.asarray(face_id)
    cover = np.asarray(coverage)
    if ids.ndim != 2:
        raise ValueError(f"{context}: face_id must have shape (H, W), got {ids.shape}")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(
            f"{context}: face_id must have an integer dtype, got {ids.dtype}"
        )
    if cover.shape != ids.shape:
        raise ValueError(
            f"{context}: coverage shape {cover.shape} does not match face_id "
            f"shape {ids.shape}"
        )
    if cover.dtype != np.bool_:
        raise ValueError(f"{context}: coverage must have dtype bool, got {cover.dtype}")
    if mask_array.ndim != 3 or mask_array.shape[1:] != ids.shape:
        raise ValueError(
            f"{context}: masks must have shape (K, {ids.shape[0]}, {ids.shape[1]}) "
            f"to match face_id, got {mask_array.shape}"
        )
    # 面IDバッファの production 不変条件をここでも確認する: 段階A は注入された
    # レンダラ(スタブを含む)の出力を受け取る境界であり、壊れた面IDを融合へ
    # 流すと原因が特定できない形でラベルが狂う。
    validate_coverage_consistency(ids, cover, context=context)
    if ids.size and int(ids.max()) >= n_faces:
        raise ValueError(
            f"{context}: face_id references face {int(ids.max())} but the mesh has "
            f"only {n_faces} face(s)"
        )

    label_map = fold_masks_to_label_map(mask_array)
    n_masks = int(mask_array.shape[0])
    segment = np.full(n_faces, UNASSIGNED, dtype=np.int32)
    visible = np.zeros(n_faces, dtype=bool)

    valid = cover & (ids >= 0)
    valid_faces = ids[valid].astype(np.int64)
    if valid_faces.size == 0:
        return ViewAssignment(segment, visible)
    pixels = np.bincount(valid_faces, minlength=n_faces)
    visible = pixels > 0

    stride = np.int64(n_masks + 1)
    key = valid_faces * stride + (label_map[valid].astype(np.int64) + np.int64(1))
    unique_key, counts = np.unique(key, return_counts=True)
    key_face = unique_key // stride
    key_seg = unique_key % stride - np.int64(1)
    # 第一キー = 面、第二キー = 出現数の降順、第三キー = seg 昇順(tie は小さい seg)。
    order = np.lexsort((key_seg, -counts, key_face))
    ordered_face = key_face[order]
    leading = np.ones(ordered_face.shape[0], dtype=bool)
    leading[1:] = ordered_face[1:] != ordered_face[:-1]
    winner = order[leading]

    winner_face = key_face[winner]
    winner_seg = key_seg[winner]
    share = counts[winner] / pixels[winner_face]
    assigned = (winner_seg >= 0) & (share >= ratio_floor)
    segment[winner_face[assigned]] = winner_seg[assigned].astype(np.int32)
    return ViewAssignment(segment, visible)


def edge_vote_statistics(
    view_segment: np.ndarray,
    adjacency: np.ndarray,
    *,
    min_votes: int = DEFAULT_MIN_VOTES,
) -> EdgeVotes:
    """面隣接の辺ごとに視点票を集計する(段階B、計画v4 §2.4.5)。

    ```
    both_v  = (view_segment[v, a] >= 0) & (view_segment[v, b] >= 0)
    agree_v = both_v & (view_segment[v, a] == view_segment[v, b])
    votes = sum_v both_v      agree = sum_v agree_v      w = agree / votes
    ```

    視点ごとのマスク index には視点間の対応が無いので、**「同じ視点の中で同じ
    マスクに入ったか」だけ**を数える(異なる視点のラベル値は決して比較しない)。
    これが「視点ごとのラベルを直接多数決」という却下案(§2.4.5)との違い。

    Args:
        view_segment: `(V, M) int32`。段階A の出力を視点方向に積んだもの。
        adjacency: `build_face_adjacency` の戻り値 `(E, 2)`。
        min_votes: この票数**以上**を「観測辺」とみなす。

    Returns:
        `EdgeVotes`。

    Raises:
        ValueError: `view_segment` の shape/dtype が契約と違う、`adjacency` の
            shape が `(E, 2)` でない、または `adjacency` が `view_segment` の
            面数の範囲外を指すとき。
    """
    votes_floor = validate_min_votes(min_votes)
    segments = _as_view_segment(view_segment)
    pairs = np.asarray(adjacency)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"adjacency must have shape (E, 2), got {pairs.shape}")
    if pairs.shape[0] == 0:
        empty_int = np.zeros(0, dtype=np.int64)
        return EdgeVotes(
            votes=empty_int,
            agree=empty_int.copy(),
            weight=np.zeros(0, dtype=np.float64),
            observed=np.zeros(0, dtype=bool),
        )
    n_faces = segments.shape[1]
    if int(pairs.min()) < 0 or int(pairs.max()) >= n_faces:
        raise ValueError(
            f"adjacency references faces outside view_segment of width {n_faces} "
            f"(min={int(pairs.min())}, max={int(pairs.max())})"
        )

    left = segments[:, pairs[:, 0]]
    right = segments[:, pairs[:, 1]]
    both = (left >= 0) & (right >= 0)
    agreeing = both & (left == right)
    votes = both.sum(axis=0, dtype=np.int64)
    agree = agreeing.sum(axis=0, dtype=np.int64)
    observed = votes >= votes_floor
    weight = np.full(votes.shape, np.nan, dtype=np.float64)
    np.divide(agree, votes, out=weight, where=observed)
    return EdgeVotes(votes=votes, agree=agree, weight=weight, observed=observed)


def fuse_view_segments(
    vertices: np.ndarray,
    faces: np.ndarray,
    view_segment: np.ndarray,
    view_visible: np.ndarray,
    *,
    angle_deg: float = DEFAULT_ANGLE_DEG,
    min_votes: int = DEFAULT_MIN_VOTES,
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    min_faces: int | None = None,
    visible_warn: float = DEFAULT_VISIBLE_WARN,
    assigned_warn: float = DEFAULT_ASSIGNED_WARN,
) -> np.ndarray:
    """視点ごとの面割当を 1 本の部位ラベルへ融合する(段階B〜E、**2パス union**)。

    段階(計画v4 §2.4.5):
      B. 辺ごとの `votes` / `agree` / `w` と幾何プライア `prior`
         (`smooth_edge_mask` — 二面角しきい値と零面積カットの組)。
      C. **2 パス union**(下の詳細)。
      D. 小部位マージ(`labels.merge_small_parts`)。
      E. 正規化 relabel(`labels.normalize_labels`)。

    **段階C の 2 パス**:

      - **観測辺** = `votes >= min_votes`、**観測面** = 観測辺を 1 本以上持つ面。
      - **パス1**: 観測辺のうち `w >= merge_threshold` を union。`w <` の観測辺は
        **カット確定**で、以後どのパスでも復活しない。
      - **パス2**: 未観測辺は **2 つの確定済み観測成分を橋渡ししない**。
        1. 両端がともに未観測面の辺で `prior == True` のものを union。
        2. 各未観測成分を、`prior == True` の境界辺が最多の観測成分へ吸収
           (tie は最小の観測成分代表 index)。**パス1後の状態から一括計算**する
           ので逐次 union の順序に依存しない。
        3. どの観測成分とも `prior == True` で繋がらない未観測成分はそのまま残す。

    **全辺が未観測のとき(マスクが 1 枚も出ない / 全面不可視)は
    `DihedralSegmenter(angle_deg=..., min_faces=...)` と厳密に一致する** —
    パス1 が無操作、パス2-1 が幾何プライアの連結成分そのものになるため
    (計画v4 §5 ゲート11)。

    Args:
        vertices: 頂点 `(N, 3)`。読むだけで書き換えない。
        faces: 三角形 `(M, 3)`。読むだけで書き換えない。
        view_segment: `(V, M) int32`。段階A の出力(`V >= 1`)。
        view_visible: `(V, M) bool`。段階A の可視フラグ。
        angle_deg: 幾何プライアのしきい値(度、`0 < angle_deg <= 180`)。
        min_votes: 観測辺とみなす最小票数(1 以上)。
        merge_threshold: パス1 の結合しきい値(`[0, 1]`、判定は `>=`)。
        min_faces: 小部位マージのしきい値。`None` なら `max(2, M // 100)`
            (`DihedralSegmenter` と同一)。
        visible_warn: 可視面率がこれ未満なら `warnings.warn`。
        assigned_warn: **可視面のうち**割当を得た率がこれ未満なら `warnings.warn`。

    Returns:
        `(M,) int64`、値は `0..P-1` の連番。面数 0 のメッシュには `(0,)` を返す。

    Raises:
        ValueError: パラメータが範囲外、`view_segment` / `view_visible` の
            shape/dtype が契約と違う、割当済みなのに不可視な面がある、または
            算出したラベルが `validate_labels` の契約を満たさないとき。
    """
    angle = validate_angle_deg(angle_deg)
    votes_floor = validate_min_votes(min_votes)
    threshold = validate_merge_threshold(merge_threshold)
    faces_floor = validate_min_faces(min_faces)
    visible_floor = validate_warn_ratio(visible_warn, "visible_warn")
    assigned_floor = validate_warn_ratio(assigned_warn, "assigned_warn")

    verts = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {face_array.shape}")
    n_faces = int(face_array.shape[0])
    segments = _as_view_segment(view_segment, n_faces=n_faces)
    visible = _as_view_visible(view_visible, segments.shape)
    if n_faces == 0:
        return np.zeros(0, dtype=np.int64)
    _warn_low_coverage(segments, visible, visible_floor, assigned_floor)

    weld_map = weld_vertices(verts)
    adjacency = build_face_adjacency(face_array, weld_map)
    # 幾何プライアは `smooth_edge_mask` 1 本に閉じている(二面角しきい値と零面積
    # 面のカットは組でしか正しくない — `adjacency.py` の docstring 参照)。
    prior = smooth_edge_mask(verts, face_array, adjacency, angle_deg=angle)
    votes = edge_vote_statistics(segments, adjacency, min_votes=votes_floor)
    _warn_if_nothing_observed(votes, segments.shape[0], votes_floor)

    roots = _two_pass_union(n_faces, adjacency, prior, votes, threshold)
    # マージにはカット前の全隣接を渡す(カット後では異なる部位どうしが定義上
    # 隣接せず、マージが一度も起きない — `merge_small_parts` の docstring)。
    merged = merge_small_parts(
        roots, adjacency, resolve_min_faces(faces_floor, n_faces)
    )
    labels = normalize_labels(merged)
    # production 不変条件: 契約違反のラベルを黙って下流(pack)へ流さない。
    validate_labels(labels, n_faces)
    return labels


def _two_pass_union(
    n_faces: int,
    adjacency: np.ndarray,
    prior: np.ndarray,
    votes: EdgeVotes,
    merge_threshold: float,
) -> np.ndarray:
    """段階C(2パス union)の本体。各面の成分代表 index `(M,) int64` を返す。

    `UnionFind` は常に小さい根を親にするので、成分代表は**成分内の最小面 index**
    であり union の呼び出し順に依存しない(`labels.UnionFind` の docstring)。
    パス2-2 の tie-break「最小の観測成分代表 index」もこの性質に乗っている。
    """
    union_find = UnionFind(n_faces)
    left = adjacency[:, 0]
    right = adjacency[:, 1]

    # --- パス1(観測パス): 観測辺のうち一致率がしきい値以上のものだけを繋ぐ。
    # `weight` は未観測辺で nan なので比較は False になるが、意図を明示するため
    # `observed` も明示的に AND する。
    merging = votes.observed & (votes.weight >= merge_threshold)
    for face_a, face_b in adjacency[merging]:
        union_find.union(int(face_a), int(face_b))

    observed_face = np.zeros(n_faces, dtype=bool)
    observed_face[left[votes.observed]] = True
    observed_face[right[votes.observed]] = True

    # --- パス2-1(未観測成分どうしの結合)。両端が未観測面の辺は定義上必ず
    # 未観測辺なので(観測辺は両端を観測面にする)、`votes.observed` の再確認は不要。
    unobserved_pair = ~observed_face[left] & ~observed_face[right]
    for face_a, face_b in adjacency[unobserved_pair & prior]:
        union_find.union(int(face_a), int(face_b))
    roots = union_find.roots()

    # --- パス2-2(吸収): 未観測成分 → 隣接する観測成分ごとの `prior` 境界辺数。
    boundary = prior & (observed_face[left] != observed_face[right])
    if not boundary.any():
        return roots
    boundary_left = left[boundary]
    boundary_right = right[boundary]
    left_is_observed = observed_face[boundary_left]
    observed_root = np.where(
        left_is_observed, roots[boundary_left], roots[boundary_right]
    )
    unobserved_root = np.where(
        left_is_observed, roots[boundary_right], roots[boundary_left]
    )
    # 無向ペアを 1 本の整数キーへ畳んでから C 側で集計する(`labels.py` の
    # `_build_part_neighbours` と同じ手)。根は面 index なので `M^2` は int64 に
    # 収まる(M <= 2^24 なら 2.8e14 << 9.2e18)。
    stride = np.int64(n_faces)
    pair_key, counts = np.unique(
        unobserved_root * stride + observed_root, return_counts=True
    )
    key_unobserved = pair_key // stride
    key_observed = pair_key % stride
    # 第一キー = 未観測成分、第二キー = 境界辺数の降順、第三キー = 観測成分代表の昇順。
    order = np.lexsort((key_observed, -counts, key_unobserved))
    ordered_unobserved = key_unobserved[order]
    leading = np.ones(ordered_unobserved.shape[0], dtype=bool)
    leading[1:] = ordered_unobserved[1:] != ordered_unobserved[:-1]
    winner = order[leading]
    # 吸収先は**パス1後の状態から一括で**決めてある。ここで union を回しても
    # 上の `counts` は変わらないので、適用順は結果に影響しない。
    for unobserved, observed in zip(
        key_unobserved[winner].tolist(), key_observed[winner].tolist()
    ):
        union_find.union(int(unobserved), int(observed))
    return union_find.roots()


def _warn_if_nothing_observed(votes: EdgeVotes, n_views: int, min_votes: int) -> None:
    """観測辺が 1 本も無い = 融合が幾何プライアそのものに落ちたことを知らせる。

    **WHY(2026-07-30 反証レビュー B2)**: §2.6 は「ML が寄与しなかった」結末に
    告知を義務づけており、`MultiViewSegmenter.segment` は **K=0(マスクが 1 枚も
    出ない)経路**にその警告を持っている。しかし**同じ結末には他の入口もある**:

      - `min_votes > n_views`(例: `n_views=1` に既定の `min_votes=2`)。
        構造的にどの辺も観測辺になれない。
      - マスクは出ているが、どの辺も両端そろって `min_votes` 票に届かない
        (遮蔽が強い / 面が細かすぎてマスクからこぼれる)。

    どれも出力は `DihedralSegmenter(同一 angle_deg/min_faces)` とビット同一になる。
    K=0 だけ告知して他を黙るのは一貫しないので、**結末の側**で 1 本にまとめる。
    合成ビューでは全面が可視・割当済みでも起きうる(= `visible_warn` /
    `assigned_warn` では捕まらない)ことが、この警告を別に置く決め手。
    """
    if votes.observed.any():
        return
    warnings.warn(
        f"no adjacency edge reached min_votes={min_votes} across the {n_views} "
        f"view(s), so no ML vote survived: this segmentation is the geometric "
        "prior alone (identical to DihedralSegmenter with the same "
        "angle_deg/min_faces). Lower min_votes, raise n_views, or use "
        "`--segmenter geometric` deliberately.",
        stacklevel=3,
    )


def _warn_low_coverage(
    view_segment: np.ndarray,
    view_visible: np.ndarray,
    visible_warn: float,
    assigned_warn: float,
) -> None:
    """被覆率 2 種を**独立に**判定して警告する(計画v4 §2.6 / §5 ゲート13)。

    - `visible_ratio` = 1 視点以上で面IDバッファに現れた面 / 全面。低いのは
      **内部空洞・視点数不足**の兆候。
    - `assigned_ratio` = 1 視点以上で割当を得た面 / **可視面**。低いのは
      **マスク粒度・解像度**の兆候。

    **`assigned_ratio` の分母を可視面にする WHY**: 全面で割ると「見えていない
    から割当が無い」面が両方の率を同時に下げ、2 つの警告が常に連動して出る。
    そうなると利用者は原因(視点配置 vs マスク粒度)を切り分けられない。
    計画v4 §5 ゲート13 が「独立に計算され、片方だけが出る」ことを要求している。
    """
    n_faces = view_segment.shape[1]
    if n_faces == 0:
        return
    assigned_any = (view_segment >= 0).any(axis=0)
    visible_any = view_visible.any(axis=0)
    # production 不変条件: 割当は可視画素の最頻値なので、不可視な面が割当を
    # 持つことはあり得ない。破れているのは段階A の外で配列が組み替えられた証拠。
    inconsistent = int((assigned_any & ~visible_any).sum())
    if inconsistent:
        raise ValueError(
            f"{inconsistent} face(s) carry a mask assignment while being invisible "
            "in every view; view_segment and view_visible disagree (an assignment "
            "can only come from visible pixels)"
        )

    n_visible = int(visible_any.sum())
    visible_ratio = n_visible / n_faces
    if visible_ratio < visible_warn:
        warnings.warn(
            f"visible_ratio={visible_ratio:.4f} is below visible_warn="
            f"{visible_warn}: {n_faces - n_visible} of {n_faces} faces never "
            "appear in any face-id buffer, so only the geometric prior decides "
            "their part. Interior cavities and too few views are the usual "
            "causes; raise n_views or fall back to `--segmenter geometric`.",
            stacklevel=3,
        )
    assigned_ratio = int(assigned_any.sum()) / n_visible if n_visible else 0.0
    if assigned_ratio < assigned_warn:
        warnings.warn(
            f"assigned_ratio={assigned_ratio:.4f} is below assigned_warn="
            f"{assigned_warn}: {n_visible - int(assigned_any.sum())} of "
            f"{n_visible} visible faces got no mask in any view, so the geometric "
            "prior decides their part. Masks that are too coarse (or an image_size "
            "too small for thin faces) are the usual causes.",
            stacklevel=3,
        )


def _as_view_segment(
    view_segment: np.ndarray, *, n_faces: int | None = None
) -> np.ndarray:
    """`(V, M) int32` として検証した `view_segment` を返す(コピーはしない)。

    **dtype を int32 に固定する WHY**: 段階A の出力契約(計画v4 §2.4.5 段階A-4)。
    `float` や `int8` を黙って受けると、`>= 0` の判定や `==` の比較が
    「マスク index の同一性」ではなく丸め/切り詰めの結果になる。
    """
    array = np.asarray(view_segment)
    if array.ndim != 2:
        raise ValueError(f"view_segment must have shape (V, M), got {array.shape}")
    if array.dtype != np.int32:
        raise ValueError(
            f"view_segment must have dtype int32 (the stage-A contract), got "
            f"{array.dtype}"
        )
    if array.shape[0] < 1:
        raise ValueError(
            "view_segment must hold at least one view (V >= 1); an empty stack is "
            "rejected rather than silently degraded to the geometric prior"
        )
    if n_faces is not None and array.shape[1] != n_faces:
        raise ValueError(
            f"view_segment must have one column per face ({n_faces}), got "
            f"{array.shape[1]}"
        )
    return array


def _as_view_visible(view_visible: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """`view_segment` と同形の `(V, M) bool` として検証した配列を返す。"""
    array = np.asarray(view_visible)
    if array.shape != shape:
        raise ValueError(
            f"view_visible must have the same shape as view_segment {shape}, got "
            f"{array.shape}"
        )
    if array.dtype != np.bool_:
        raise ValueError(f"view_visible must have dtype bool, got {array.dtype}")
    return array
