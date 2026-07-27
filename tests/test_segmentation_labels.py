"""`segmentation/labels.py` — 契約検証・union-find・小部位マージ・relabel のテスト。

ここは合成ラベルだけで自立する(メッシュ fixture に依存しない)。`fusion.py`
(Step 2-4)が同じ部品を再利用するため、幾何バックエンド抜きで契約を固定する。
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Callable

import numpy as np
import pytest

from atlasmith.segmentation.labels import (
    UnionFind,
    _resolve_absorption_chains,
    merge_small_parts,
    normalize_labels,
    union_find_labels,
    validate_labels,
)

# ---------------------------------------------------------------------------
# union-find
# ---------------------------------------------------------------------------


def test_union_find_root_is_the_minimum_index_of_the_component() -> None:
    union_find = UnionFind(5)
    # 大きい index から結合しても根は最小 index に落ち着く(決定性の根拠)。
    assert union_find.union(4, 3) is True
    assert union_find.union(3, 1) is True
    assert union_find.union(1, 4) is False  # 既に同一成分
    assert union_find.find(4) == 1
    assert union_find.roots().tolist() == [0, 1, 2, 1, 1]


def test_union_find_result_is_independent_of_union_order() -> None:
    forward = UnionFind(6)
    for a, b in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        forward.union(a, b)
    backward = UnionFind(6)
    for a, b in [(5, 4), (4, 3), (2, 1), (1, 0)]:
        backward.union(a, b)
    assert np.array_equal(forward.roots(), backward.roots())


def test_union_find_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        UnionFind(-1)


def test_union_find_labels_on_pairs_and_empty_input() -> None:
    pairs = np.array([[0, 2], [2, 4]], dtype=np.int64)
    roots = union_find_labels(5, pairs)
    assert roots.dtype == np.int64
    assert roots.tolist() == [0, 1, 0, 3, 0]
    empty = union_find_labels(3, np.zeros((0, 2), dtype=np.int64))
    assert empty.tolist() == [0, 1, 2]
    assert union_find_labels(0, np.zeros((0, 2), dtype=np.int64)).shape == (0,)


def test_union_find_labels_rejects_bad_pair_shape() -> None:
    with pytest.raises(ValueError, match=r"pairs must have shape \(E, 2\)"):
        union_find_labels(3, np.zeros((3,), dtype=np.int64))


# ---------------------------------------------------------------------------
# ラベル契約検証
# ---------------------------------------------------------------------------


def test_validate_labels_accepts_consecutive_labels_and_returns_part_count() -> None:
    assert validate_labels(np.array([0, 1, 1, 2], dtype=np.int64), 4) == 3
    assert validate_labels(np.zeros(0, dtype=np.int64), 0) == 0


@pytest.mark.parametrize(
    ("labels", "n_faces", "match"),
    [
        ([0, 1], 2, "numpy ndarray"),
        (np.zeros((2, 1), dtype=np.int64), 2, r"shape \(M,\)"),
        (np.zeros(2, dtype=np.int64), 3, "one entry per face"),
        (np.zeros(2, dtype=np.int32), 2, "dtype int64"),
        (np.array([0, 2], dtype=np.int64), 2, "consecutive"),
        (np.array([-1, 0], dtype=np.int64), 2, "consecutive"),
        (np.array([1, 1], dtype=np.int64), 2, "consecutive"),
    ],
)
def test_validate_labels_rejects_contract_violations(
    labels: object, n_faces: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_labels(labels, n_faces)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 小部位マージ
# ---------------------------------------------------------------------------


def test_merge_keeps_the_destination_label_not_the_smaller_one() -> None:
    """吸収された側(ラベル 0)が消え、**マージ先**のラベル 1 が存続すること。"""
    labels = np.array([0, 1, 1, 1], dtype=np.int64)
    adjacency = np.array([[0, 1]], dtype=np.int64)
    merged = merge_small_parts(labels, adjacency, min_faces=2)
    assert merged.tolist() == [1, 1, 1, 1]


def test_merge_prefers_the_neighbour_with_most_shared_edges() -> None:
    """共有境界辺数がラベル順より優先すること(ラベル 1 が ラベル 0 に勝つ)。"""
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    adjacency = np.array([[0, 5], [3, 5], [4, 5]], dtype=np.int64)
    # 部位: 0={面0,1,2}(3面) / 1={面3,4}(2面) / 2={面5}(1面・唯一の小部位)
    # 面5 は部位0 と 1 本、部位1 と 2 本を共有 → ラベルが大きい部位1 が勝つ。
    merged = merge_small_parts(labels, adjacency, min_faces=2)
    assert merged.tolist() == [0, 0, 0, 1, 1, 1]


def test_merge_order_is_smallest_part_then_smallest_label() -> None:
    """同数の小部位が並ぶとき、ラベル値が小さい方から処理し tie-break も最小ラベル。"""
    labels = np.array([0, 1, 2, 2], dtype=np.int64)
    adjacency = np.array([[0, 1], [0, 2]], dtype=np.int64)
    # 部位0 と 部位1 が同数(1面)→ ラベル最小の部位0 を先に処理。
    # 部位0 の隣接は 部位1 と 部位2 が各 1 本 → tie-break で最小ラベルの 部位1 へ。
    merged = merge_small_parts(labels, adjacency, min_faces=2)
    assert merged.tolist() == [1, 1, 2, 2]


def test_merge_leaves_isolated_small_parts_alone() -> None:
    labels = np.array([0, 0, 1], dtype=np.int64)
    adjacency = np.array([[0, 1]], dtype=np.int64)  # 面2 はどこにも隣接しない
    assert merge_small_parts(labels, adjacency, min_faces=2).tolist() == [0, 0, 1]


def test_merge_is_a_no_op_below_min_faces_two() -> None:
    labels = np.array([0, 1, 2], dtype=np.int64)
    adjacency = np.array([[0, 1], [1, 2]], dtype=np.int64)
    for min_faces in (0, 1):
        assert np.array_equal(
            merge_small_parts(labels, adjacency, min_faces=min_faces), labels
        )


def test_merge_collapses_everything_when_min_faces_exceeds_mesh() -> None:
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    adjacency = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    merged = merge_small_parts(labels, adjacency, min_faces=99)
    assert len(np.unique(merged)) == 1


def test_merge_is_non_destructive_and_returns_int64() -> None:
    labels = np.array([0, 1, 1, 1], dtype=np.int64)
    original = labels.copy()
    merged = merge_small_parts(labels, np.array([[0, 1]], dtype=np.int64), min_faces=2)
    assert merged.dtype == np.int64
    assert np.array_equal(labels, original)


def test_merge_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match=r"labels must have shape \(M,\)"):
        merge_small_parts(np.zeros((2, 2), dtype=np.int64), np.zeros((0, 2)), 2)
    with pytest.raises(ValueError, match=r"adjacency must have shape \(E, 2\)"):
        merge_small_parts(np.zeros(2, dtype=np.int64), np.zeros((2, 3)), 2)


def test_absorption_chain_resolution_fails_loudly_on_a_cycle() -> None:
    """内部不変条件が壊れたとき、無限ループではなく `ValueError` で止まること。

    `absorbed_into` は正しく構築されれば森なので、閉路は契約内の入力では起きない
    (無作為突き合わせ 72 ケースでも 0 件)。ただし `labels.py` は Step 2-4 の
    `multiview/fusion.py` が再利用する共有部品であり、閉路を作る変異(例: 吸収時の
    逆参照張り替え漏れ)を入れると経路探索が**永遠に回る** — 実測ではテストが
    7 分間無出力のままタイムアウトし、原因の特定が極めて難しかった。将来の実装者が
    「静かなハング」ではなく「大声の失敗」を受け取れることを固定する。

    private ヘルパを直接叩くのは、契約内の入力からはこの状態に到達できないため
    (到達できるなら本体の欠陥であって、ガードの問題ではない)。
    """
    with pytest.raises(ValueError, match="absorption chain"):
        _resolve_absorption_chains([1, 0])  # 0 -> 1 -> 0 の閉路


# ---------------------------------------------------------------------------
# 小部位マージ — 素朴リファレンスとの等価性
#
# production は「1回に1部位ずつ吸収する」規約を *増分更新* で実装しており、
# 素朴な逐次実装と結果が一致することが正しさの条件になっている。そこで規約だけを
# 共有し、実装構造を共有しないリファレンスを置いて無作為入力で突き合わせる。
# ---------------------------------------------------------------------------


def _reference_merge_small_parts(
    labels: np.ndarray, adjacency: np.ndarray, min_faces: int
) -> np.ndarray:
    """素朴な逐次リファレンス(独立実装 — production の構造を共有しない)。

    共有するのは規約だけ: 候補順 = 面数昇順 → ラベル昇順 / マージ先 = 共有境界辺数
    最大、同数ならラベル最小 / マージ先のラベルが存続 / 孤立部位は残す。

    実装上の独立性(逐語コピーにしないための意図的な差):
      - numpy を使わず `collections.Counter` と list で書く。
      - 部位 index 空間へ写さず **ラベル値そのもの** を扱う。
      - ヒープも増分更新も使わず、毎反復で面数と共有辺数を全走査から作り直す。
    """
    current = [int(value) for value in labels.tolist()]
    pairs = [(int(a), int(b)) for a, b in adjacency.tolist()]
    while True:
        sizes = Counter(current)
        if len(sizes) <= 1:
            break
        candidates = sorted(
            (count, label) for label, count in sizes.items() if count < min_faces
        )
        for _, label in candidates:
            shared: Counter[int] = Counter()
            for face_a, face_b in pairs:
                label_a, label_b = current[face_a], current[face_b]
                if label_a == label and label_b != label:
                    shared[label_b] += 1
                elif label_b == label and label_a != label:
                    shared[label_a] += 1
            if not shared:
                continue  # 孤立部位はそのまま残す。
            destination = max(shared.items(), key=lambda item: (item[1], -item[0]))[0]
            current = [destination if v == label else v for v in current]
            break
        else:
            break
    return np.array(current, dtype=np.int64)


def _random_merge_case(seed: int, max_faces: int) -> tuple[np.ndarray, np.ndarray, int]:
    """無作為な (labels, adjacency, min_faces) を作る。

    ラベル値は連番にしない(`7k + 3`)— 部位 index とラベル値を取り違える実装を
    弾くため。隣接には重複ペアも入れる(同じ2部位が複数の辺を共有する状況)。
    """
    rng = np.random.default_rng(seed)
    n_faces = int(rng.integers(1, max_faces + 1))
    n_parts = int(rng.integers(1, n_faces + 1))
    labels = (rng.integers(0, n_parts, size=n_faces) * 7 + 3).astype(np.int64)
    n_edges = int(rng.integers(0, 2 * n_faces + 1))
    raw = rng.integers(0, n_faces, size=(n_edges, 2))
    keep = raw[:, 0] != raw[:, 1]
    pairs = np.sort(raw[keep], axis=1).astype(np.int64).reshape(-1, 2)
    return labels, pairs, int(rng.integers(1, n_faces + 3))


@pytest.mark.parametrize("seed", range(60))
def test_merge_matches_the_naive_reference_on_random_inputs(seed: int) -> None:
    labels, adjacency, min_faces = _random_merge_case(seed, max_faces=30)
    assert np.array_equal(
        merge_small_parts(labels, adjacency, min_faces),
        _reference_merge_small_parts(labels, adjacency, min_faces),
    )


@pytest.mark.parametrize("seed", range(1000, 1012))
def test_merge_matches_the_naive_reference_on_larger_random_inputs(seed: int) -> None:
    """連鎖マージ(吸収先がさらに吸収される)が深くなる規模でも一致すること。"""
    labels, adjacency, min_faces = _random_merge_case(seed, max_faces=140)
    assert np.array_equal(
        merge_small_parts(labels, adjacency, min_faces),
        _reference_merge_small_parts(labels, adjacency, min_faces),
    )


# ---------------------------------------------------------------------------
# 小部位マージ — 性能退行ゲート
# ---------------------------------------------------------------------------

# 1本鎖の面数。大きい方は小さい方の 4 倍。
_SCALING_SMALL_FACES = 10_000
_SCALING_LARGE_FACES = 40_000
# 4 倍規模での所要時間比の上限。準線形(ヒープぶんの log を含む)なら 4〜5、
# 部位ごとにラベル配列を舐め直す実装なら 12〜16 になる(実測: 旧実装 12.17、
# 新実装 4.26 — いずれも当開発機、5000 → 20000 面)。両者の中間に置く。
_SCALING_MAX_RATIO = 8.0


def _merge_chain_case(n_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """全面が別ラベルの1本鎖 — 小部位マージが `n_faces - 1` 回連鎖する最悪形。"""
    labels = np.arange(n_faces, dtype=np.int64)
    adjacency = np.stack(
        [np.arange(n_faces - 1), np.arange(1, n_faces)], axis=1
    ).astype(np.int64)
    return labels, adjacency


def _merge_grid_case(n_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """格子状の隣接(部位あたりの次数 ~4)。全面が別ラベル。

    1本鎖は次数 2 なので、「隣接数の多い部位だけを劣化させる退行」(例: 吸収のたびに
    マージ先の隣接表を作り直す実装)を素通ししてしまう。三角メッシュの部位隣接は
    高々 3 * 部位面数 なので、次数 ~4 の格子は実メッシュ側の上限に近い形。
    """
    side = int(round(n_faces**0.5))
    index = np.arange(side * side, dtype=np.int64).reshape(side, side)
    right = np.stack([index[:, :-1].ravel(), index[:, 1:].ravel()], axis=1)
    down = np.stack([index[:-1, :].ravel(), index[1:, :].ravel()], axis=1)
    adjacency = np.concatenate([right, down], axis=0).astype(np.int64)
    return np.arange(side * side, dtype=np.int64), adjacency


_SCALING_SHAPES = {"chain": _merge_chain_case, "grid": _merge_grid_case}


def _fastest_seconds(call: Callable[[], object], repeats: int = 3) -> float:
    """最速値を採る(GC や OS のスケジューリング揺らぎを落とすため)。"""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("shape", sorted(_SCALING_SHAPES))
def test_merge_does_not_scale_quadratically_with_part_count(shape: str) -> None:
    """マージ段が部位数に対して二次にならないこと(性能退行ゲート)。

    **壁時計時間の絶対値ではなく比を見る**: 絶対値は CI マシンの速度で変わるが、
    規模を 4 倍にしたときの比はアルゴリズムの次数で決まる。どちらの形状も
    「マージ前部位数 P == 面数 M」かつ「マージが M-1 回連鎖する」最悪形で、
    部位ごとに全ラベル配列を舐め直す実装ではここが O(M^2) になる。

    **形状を 2 つ回す WHY**: 1本鎖は部位あたりの隣接が 2 本しかないので、
    高次数の部位グラフだけを劣化させる退行を捕まえられない。格子(次数 ~4)を
    併せて回す。

    **空虚でないことの担保**: 両規模とも実際に P == 1 まで畳まれる(マージ段が
    素通りしていない)ことを併せて assert する。
    """
    build = _SCALING_SHAPES[shape]
    small_labels, small_adjacency = build(_SCALING_SMALL_FACES)
    large_labels, large_adjacency = build(_SCALING_LARGE_FACES)

    def run_small() -> object:
        return merge_small_parts(small_labels, small_adjacency, len(small_labels))

    def run_large() -> object:
        return merge_small_parts(large_labels, large_adjacency, len(large_labels))

    assert len(np.unique(run_small())) == 1
    assert len(np.unique(run_large())) == 1
    # 規模がきっちり 4 倍であること(比の分母/分子が同じ倍率で伸びている前提)。
    assert len(large_labels) == 4 * len(small_labels)

    ratio = _fastest_seconds(run_large) / _fastest_seconds(run_small)
    assert ratio <= _SCALING_MAX_RATIO, (
        f"merge_small_parts ({shape}) scaled by {ratio:.1f}x when the part count "
        f"grew 4x (limit {_SCALING_MAX_RATIO}) - the merge stage looks super-linear "
        "again"
    )


# ---------------------------------------------------------------------------
# 正規化 relabel
# ---------------------------------------------------------------------------


def test_normalize_orders_parts_by_their_smallest_face_index() -> None:
    labels = np.array([7, 3, 7, 9, 3], dtype=np.int64)
    # 最小面 index: 7 -> 0, 3 -> 1, 9 -> 3 なので順位は 7, 3, 9。
    normalized = normalize_labels(labels)
    assert normalized.dtype == np.int64
    assert normalized.tolist() == [0, 1, 0, 2, 1]


def test_normalize_is_identity_on_already_normalized_labels() -> None:
    labels = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    assert np.array_equal(normalize_labels(labels), labels)


def test_normalize_handles_negative_and_empty_inputs() -> None:
    assert normalize_labels(np.array([-5, -9, -5], dtype=np.int64)).tolist() == [
        0,
        1,
        0,
    ]
    assert normalize_labels(np.zeros(0, dtype=np.int64)).shape == (0,)


def test_normalize_is_non_destructive() -> None:
    labels = np.array([7, 3, 7], dtype=np.int64)
    original = labels.copy()
    normalize_labels(labels)
    assert np.array_equal(labels, original)


def test_normalize_output_satisfies_the_label_contract() -> None:
    labels = np.array([12, 5, 12, 99, 5, 5], dtype=np.int64)
    normalized = normalize_labels(labels)
    assert validate_labels(normalized, len(labels)) == 3


def test_normalize_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"labels must have shape \(M,\)"):
        normalize_labels(np.zeros((2, 2), dtype=np.int64))
