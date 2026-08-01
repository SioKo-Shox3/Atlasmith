"""2パス視点間融合と `MultiViewSegmenter` のゲート(計画v4 §5 Step 2-4 の 17 項目)。

**GPU も GL コンテキストも重みも要らない**(だから `gl` / `ml` マーカーは付かず、
CI の `-m "not ml and not gl"` でも全件走る)。合成 `RenderedView` と固定マスクを
注入することで、非決定的な `MaskProposer` を切り離した下流全体を厳密ゲートに
かけられる — これが計画v4 §2.4 の「非決定的なのは1段だけ」という分離の実利。

テスト関数名は `test_gateNN_...` の形で、計画v4 §5 Step 2-4 の合否基準 1〜17 に
1 対 1 で対応させてある(追加の契約テストは `test_...` のみ)。

主ゲートは 3 本:
  - ゲート7 = 融合 E2E(`MultiViewSegmenter.segment()` を通した 6:6 分割)。
  - ゲート9 = **2パス設計の反証**(v3 の1パス設計なら必ず落ちる)。
  - ゲート11 = 幾何プライアへの完全劣化(`DihedralSegmenter` とビット一致)。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from atlasmith.segmentation import DihedralSegmenter, SegmentationBackend
from atlasmith.segmentation.adjacency import (
    build_face_adjacency,
    smooth_edge_mask,
    weld_vertices,
)
from atlasmith.segmentation.labels import (
    normalize_labels,
    union_find_labels,
    validate_labels,
)
from atlasmith.segmentation.multiview import MultiViewSegmenter, fusion
from atlasmith.types import MeshData

# `cube_mesh` の 6:6 分割(計画v4 §5 Step 2-4 の主 fixture)。`_build_cube_geometry`
# の面順は +X, -X, +Y, -Y, +Z, -Z の各 2 面。
_CUBE_A = (0, 1, 4, 5, 8, 9)  # +X, +Y, +Z
_CUBE_B = (2, 3, 6, 7, 10, 11)  # -X, -Y, -Z
_CUBE_FACES = 12
# 既定の視点数(`cameras.DEFAULT_N_VIEWS`)。E2E ゲートはこの本数の合成ビューを渡す。
_N_VIEWS = 24


# ---------------------------------------------------------------------------
# テスト側ヘルパ(production の union-find / relabel に依存しない独立実装)
# ---------------------------------------------------------------------------


def _adjacency(mesh: MeshData) -> np.ndarray:
    """`mesh` の weld 面隣接 `(E, 2)`。融合が内部で使うものと同じ構築。"""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    return build_face_adjacency(faces, weld_vertices(vertices))


def _edge_row(adjacency: np.ndarray, face_a: int, face_b: int) -> int:
    """辺 `(face_a, face_b)` の行 index。無ければ `AssertionError`。"""
    low, high = min(face_a, face_b), max(face_a, face_b)
    rows = np.flatnonzero((adjacency[:, 0] == low) & (adjacency[:, 1] == high))
    assert rows.size == 1, f"edge ({low}, {high}) is not a unique adjacency row"
    return int(rows[0])


def _is_connected(members: tuple[int, ...], adjacency: np.ndarray) -> bool:
    """`members` が隣接グラフ上で連結かを BFS で判定する(独立実装)。"""
    wanted = set(members)
    neighbours: dict[int, set[int]] = {face: set() for face in wanted}
    for face_a, face_b in adjacency.tolist():
        if face_a in wanted and face_b in wanted:
            neighbours[face_a].add(face_b)
            neighbours[face_b].add(face_a)
    start = next(iter(wanted))
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop()
        for neighbour in neighbours[current]:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen == wanted


def _partition(labels: np.ndarray) -> list[frozenset[int]]:
    """ラベル値ではなく **面の分割の仕方** を「最小要素の昇順」で正規化する。"""
    groups: dict[int, set[int]] = {}
    for index, value in enumerate(labels.tolist()):
        groups.setdefault(value, set()).add(index)
    return [frozenset(group) for group in sorted(groups.values(), key=min)]


def _n_parts(labels: np.ndarray) -> int:
    return int(np.unique(labels).shape[0])


def _row_from_groups(n_faces: int, groups: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """1 視点ぶんの `view_segment` 行。`groups[k]` の面に `k`、残りに `UNASSIGNED`。"""
    row = np.full(n_faces, fusion.UNASSIGNED, dtype=np.int32)
    for index, group in enumerate(groups):
        for face in group:
            row[face] = index
    return row


def _stack(rows: list[np.ndarray]) -> np.ndarray:
    """視点方向に積んで `(V, M) int32` にする。"""
    return np.stack(rows, axis=0).astype(np.int32)


def _all_visible(view_segment: np.ndarray) -> np.ndarray:
    """全面が全視点で可視だった、という `view_visible`。"""
    return np.ones(view_segment.shape, dtype=bool)


def _run_static_segmenter(
    mesh: MeshData,
    groups_per_view: list[tuple[tuple[int, ...], ...]],
    renderer_class: type,
    proposer_class: type,
    build_view,
    masks_from_face_groups,
    **kwargs: object,
) -> np.ndarray:
    """合成ビュー + 固定マスクで `MultiViewSegmenter.segment()` を通す。"""
    n_faces = len(mesh.faces)
    view = build_view(n_faces)
    masks = [masks_from_face_groups(view.face_id, groups) for groups in groups_per_view]
    renderer = renderer_class([view] * len(groups_per_view))
    proposer = proposer_class(masks)
    with MultiViewSegmenter(
        proposer,
        lambda _mesh: renderer,
        n_views=len(groups_per_view),
        **kwargs,  # type: ignore[arg-type]
    ) as segmenter:
        return segmenter.segment(mesh)


# ---------------------------------------------------------------------------
# ゲート1: fixture 前提の検証
# ---------------------------------------------------------------------------


def test_gate01_cube_fixture_supports_the_six_six_split(
    cube_mesh: MeshData, two_cubes_mesh: MeshData
) -> None:
    """`cube_mesh` は E==18・A/B が各々連結・cross 辺 6 本(計画v4 §5 ゲート1)。

    どう壊れたら落ちるか: `_build_cube_geometry` の面順や weld が変わって A/B の
    面 index が別の facet を指すようになった瞬間。以降のゲートはすべてこの 3 つの
    数値に乗っているので、**先にここで前提を証明する**。
    """
    adjacency = _adjacency(cube_mesh)
    assert adjacency.shape[0] == 18
    assert set(_CUBE_A) | set(_CUBE_B) == set(range(_CUBE_FACES))
    assert _is_connected(_CUBE_A, adjacency)
    assert _is_connected(_CUBE_B, adjacency)
    in_a = np.isin(adjacency, np.array(_CUBE_A, dtype=np.int64))
    cross = in_a[:, 0] != in_a[:, 1]
    assert int(cross.sum()) == 6
    # cross 辺の面ペア。計画v4 §5 の facet 列挙(+X--Y, +X--Z, +Y--X, +Y--Z,
    # +Z--X, +Z--Y)を面 index に翻訳したもの。以降のゲートがこの 6 本を名指しで
    # 使うので、ここで一意に固定する。
    assert sorted(tuple(row) for row in adjacency[cross].tolist()) == [
        (0, 10),  # +X--Z
        (1, 6),  # +X--Y
        (2, 5),  # -X-+Y
        (2, 9),  # -X-+Z
        (4, 10),  # +Y--Z
        (6, 8),  # -Y-+Z
    ]

    # BL-1 の記録: `two_cubes_mesh` は cube 間の辺が 0 本なので融合ゲートに使えない。
    two_adjacency = _adjacency(two_cubes_mesh)
    first_cube = two_adjacency < _CUBE_FACES
    assert int((first_cube[:, 0] != first_cube[:, 1]).sum()) == 0


# ---------------------------------------------------------------------------
# ゲート2: マスク正規化の全順序性【BL-8】
# ---------------------------------------------------------------------------


def _ambiguous_mask_groups() -> tuple[tuple[int, ...], ...]:
    """同画素数・同 first true index の非同一マスク対を含むグループ集合。

    面 0/1 の帯と面 0/2 の帯はどちらも 2 面ぶん(同画素数)で、最初の True 画素も
    面 0 のブロック先頭(同一)。内容だけが違うので、第3キー(packbits)が無いと
    順序が入力順に依存する。
    """
    return ((0, 1), (0, 2), tuple(range(3, _CUBE_FACES)))


def test_gate02_mask_normalization_is_a_total_order(
    cube_mesh: MeshData,
    build_block_view,
    masks_from_face_groups,
    static_renderer: type,
    static_mask_proposer: type,
) -> None:
    """正順/逆順/シャッフルで `normalize_masks` / `view_segment` / `labels` が不変。

    どう壊れたら落ちるか: 整列キーから第3キー(内容のバイト列)を落とすと、
    同画素数・同 first index の 2 枚の順序が入力順で決まり、逆順入力で
    `view_segment` が変わる(= ここが落ちる)。
    """
    view = build_block_view(_CUBE_FACES)
    groups = _ambiguous_mask_groups()
    masks = masks_from_face_groups(view.face_id, groups)
    # 前提: 曖昧なマスク対が本当に「同画素数・同 first index」であること。
    flat = masks.reshape(masks.shape[0], -1)
    assert int(flat[0].sum()) == int(flat[1].sum())
    assert int(np.argmax(flat[0])) == int(np.argmax(flat[1]))
    assert not np.array_equal(masks[0], masks[1])

    orders = {
        "forward": [0, 1, 2],
        "reverse": [2, 1, 0],
        "shuffled": [1, 2, 0],
    }
    normalized: dict[str, np.ndarray] = {}
    segments: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for name, order in orders.items():
        permuted = masks[order]
        normalized[name] = fusion.normalize_masks(permuted)
        segments[name] = fusion.assign_view_faces(
            normalized[name], view.face_id, view.coverage, n_faces=_CUBE_FACES
        ).segment
        labels[name] = _run_static_segmenter(
            cube_mesh,
            [tuple(groups[index] for index in order)] * _N_VIEWS,
            static_renderer,
            static_mask_proposer,
            build_block_view,
            masks_from_face_groups,
        )
    for name in ("reverse", "shuffled"):
        assert np.array_equal(normalized[name], normalized["forward"])
        assert np.array_equal(segments[name], segments["forward"])
        assert np.array_equal(labels[name], labels["forward"])
    # 空虚でないこと: マスクの区別が実際にラベルへ効いている。
    assert _n_parts(labels["forward"]) >= 2


def test_normalize_masks_drops_empty_and_duplicate_masks() -> None:
    """空マスク除去と完全一致の重複除去(§2.4.4 手順1〜2)。"""
    masks = np.zeros((5, 4, 4), dtype=bool)
    masks[0, 0, 0] = True  # 1 画素
    masks[1] = False  # 空 -> 除去
    masks[2, 0, 0] = True  # masks[0] と完全一致 -> 重複除去
    masks[3, :2, :2] = True  # 4 画素(最大)
    masks[4, 3, 3] = True  # 1 画素、first index は最大
    normalized = fusion.normalize_masks(masks)
    assert normalized.shape == (3, 4, 4)
    assert np.array_equal(normalized[0], masks[3])  # 面積降順
    assert np.array_equal(normalized[1], masks[0])
    assert np.array_equal(normalized[2], masks[4])
    # 引数非破壊。
    assert masks[1].sum() == 0 and normalized.base is None


# ---------------------------------------------------------------------------
# ゲート3〜6: 段階A の規約
# ---------------------------------------------------------------------------


def _quadrant_face_id() -> np.ndarray:
    """4 面が 2x2 ブロック(各 4x4)に写る 8x8 の面IDバッファ。"""
    face_id = np.empty((8, 8), dtype=np.int32)
    face_id[:4, :4] = 0
    face_id[:4, 4:] = 1
    face_id[4:, :4] = 2
    face_id[4:, 4:] = 3
    return face_id


def test_gate03_stage_a_positive_control() -> None:
    """手組みの面IDバッファ+既知マスクで期待 `view_segment` と完全一致(ゲート3)。"""
    face_id = _quadrant_face_id()
    coverage = np.ones_like(face_id, dtype=bool)
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, :4, :] = True  # 面 0/1 を丸ごと(32 画素)
    masks[1, 4:, :4] = True  # 面 2 を丸ごと(16 画素)
    normalized = fusion.normalize_masks(masks)

    expected_seg = np.full((8, 8), -1, dtype=np.int32)
    expected_seg[:4, :] = 0
    expected_seg[4:, :4] = 1
    assert np.array_equal(fusion.fold_masks_to_label_map(normalized), expected_seg)

    assignment = fusion.assign_view_faces(
        normalized, face_id, coverage, n_faces=4, assign_ratio=0.5
    )
    assert np.array_equal(assignment.segment, np.array([0, 0, 1, -1], dtype=np.int32))
    assert assignment.segment.dtype == np.int32
    assert np.array_equal(assignment.visible, np.array([True, True, True, True]))


def _single_face_view(interior: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """面 0 が `interior x interior` の内側に写り、外周 1 画素が背景のバッファ。"""
    size = interior + 2
    face_id = np.full((size, size), -1, dtype=np.int32)
    face_id[1 : 1 + interior, 1 : 1 + interior] = 0
    return face_id, face_id >= 0


def _masks_over_face_pixels(face_id: np.ndarray, counts: tuple[int, ...]) -> np.ndarray:
    """面 0 の可視画素を走査順に切り分け、`counts[k]` 画素の第 k マスクを作る。"""
    rows, cols = np.nonzero(face_id == 0)
    masks = np.zeros((len(counts), *face_id.shape), dtype=bool)
    start = 0
    for index, count in enumerate(counts):
        masks[index][rows[start : start + count], cols[start : start + count]] = True
        start += count
    return masks


@pytest.mark.parametrize(
    ("winner_pixels", "runner_up_pixels"), [(49, 26), (50, 26), (51, 25)]
)
@pytest.mark.parametrize("assign_ratio", [0.49, 0.50, 0.51])
def test_gate04_assign_ratio_boundary_is_closed(
    winner_pixels: int, runner_up_pixels: int, assign_ratio: float
) -> None:
    """占有率 0.49/0.50/0.51 を作り分け、`>=` で割当される(ゲート4)。

    **最頻値が `-1` にならないよう 2 枚目のマスクを置いてある WHY**: 単に
    「49 画素のマスク + 51 画素の未割当」にすると最頻値が `-1` になり、
    落ちる理由が `assign_ratio` ではなくゲート6 の規約になってしまう。
    """
    face_id, coverage = _single_face_view()
    masks = _masks_over_face_pixels(face_id, (winner_pixels, runner_up_pixels))
    normalized = fusion.normalize_masks(masks)
    assignment = fusion.assign_view_faces(
        normalized, face_id, coverage, n_faces=1, assign_ratio=assign_ratio
    )
    share = winner_pixels / 100.0
    expected = 0 if share >= assign_ratio else fusion.UNASSIGNED
    assert int(assignment.segment[0]) == expected


def test_gate05_nested_masks_let_the_smaller_one_win() -> None:
    """大マスクと小マスクの入れ子で小マスクが勝つ(ゲート5、期待 `seg` と完全一致)。"""
    face_id = np.empty((8, 8), dtype=np.int32)
    face_id[:, :4] = 0
    face_id[:, 4:] = 1
    coverage = np.ones_like(face_id, dtype=bool)
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, :, :] = True  # 大マスク(64 画素)
    masks[1, :, 4:] = True  # 小マスク(32 画素・大マスクに完全に含まれる)
    assert bool((masks[1] & ~masks[0]).sum() == 0)  # 入れ子であることの前提
    normalized = fusion.normalize_masks(masks)

    expected_seg = np.zeros((8, 8), dtype=np.int32)
    expected_seg[:, 4:] = 1
    assert np.array_equal(fusion.fold_masks_to_label_map(normalized), expected_seg)

    assignment = fusion.assign_view_faces(
        normalized, face_id, coverage, n_faces=2, assign_ratio=0.5
    )
    assert np.array_equal(assignment.segment, np.array([0, 1], dtype=np.int32))


def test_gate06_unassigned_is_a_mode_candidate() -> None:
    """面の 90% が未割当・10% が細片マスクなら `-1` になる(ゲート6)。

    `assign_ratio` を 0.05 まで下げても `-1` のままであることを併せて確認する —
    落ちる理由が占有率ではなく「`-1` を最頻値候補に含める」規約であることの証明。
    """
    face_id, coverage = _single_face_view()
    masks = _masks_over_face_pixels(face_id, (10,))
    normalized = fusion.normalize_masks(masks)
    assert int(normalized[0].sum()) == 10
    for assign_ratio in (0.05, 0.5):
        assignment = fusion.assign_view_faces(
            normalized, face_id, coverage, n_faces=1, assign_ratio=assign_ratio
        )
        assert int(assignment.segment[0]) == fusion.UNASSIGNED
        assert bool(assignment.visible[0]) is True


# ---------------------------------------------------------------------------
# ゲート7: 融合 E2E(主ゲート)
# ---------------------------------------------------------------------------


def test_gate07_fusion_end_to_end_on_the_cube_six_six_split(
    cube_mesh: MeshData,
    build_block_view,
    masks_from_face_groups,
    static_renderer: type,
    static_mask_proposer: type,
) -> None:
    """(i) 全 24 視点が A|B を割る → P==2 かつ分割が `{A, B}`、(ii) 1 マスク → P==1。

    これが Step 2-4 の主ゲート: 合成マスク+合成 `RenderedView` を注入すれば
    GPU 無しで `MultiViewSegmenter.segment()` が決定的にラベルを返す、という
    本ステップの挙動変化そのもの。
    """
    split = _run_static_segmenter(
        cube_mesh,
        [(_CUBE_A, _CUBE_B)] * _N_VIEWS,
        static_renderer,
        static_mask_proposer,
        build_block_view,
        masks_from_face_groups,
    )
    assert _n_parts(split) == 2
    assert _partition(split) == [frozenset(_CUBE_A), frozenset(_CUBE_B)]
    validate_labels(split, _CUBE_FACES)

    single = _run_static_segmenter(
        cube_mesh,
        [(tuple(range(_CUBE_FACES)),)] * _N_VIEWS,
        static_renderer,
        static_mask_proposer,
        build_block_view,
        masks_from_face_groups,
    )
    assert _n_parts(single) == 1
    assert np.array_equal(single, np.zeros(_CUBE_FACES, dtype=np.int64))


# ---------------------------------------------------------------------------
# ゲート8: しきい値対照(3 分岐)
# ---------------------------------------------------------------------------

# (disagree する視点数, agree する視点数, 期待 w, 期待 P)。24 視点で
# w = agree/24 を作り、`w >= merge_threshold(=0.5)` の閉境界を 3 点で挟む。
_THRESHOLD_CASES = (
    (16, 8, 8 / 24, 2),
    (8, 16, 16 / 24, 1),
    (12, 12, 12 / 24, 1),
)


@pytest.mark.parametrize(
    ("n_disagree", "n_agree", "expected_weight", "expected_parts"), _THRESHOLD_CASES
)
def test_gate08_merge_threshold_is_closed_on_the_cross_edges(
    cube_mesh: MeshData,
    build_block_view,
    masks_from_face_groups,
    static_renderer: type,
    static_mask_proposer: type,
    n_disagree: int,
    n_agree: int,
    expected_weight: float,
    expected_parts: int,
) -> None:
    """16:8 → カット / 8:16 → 結合 / **12:12 → `>=` により結合**(ゲート8)。

    `n_disagree` 視点は A|B を別マスクに割り、`n_agree` 視点は全面を 1 マスクに
    入れる。cross 辺 6 本はすべて同じ投票パターンになり、A/B 内部の辺は
    どの視点でも一致するので `w = 1`。融合レベル(`view_segment` を直に組む)と
    E2E(`segment()` 経由)の両方で同じラベルが出ることを確認する。
    """
    groups_per_view = [(_CUBE_A, _CUBE_B)] * n_disagree + [
        (tuple(range(_CUBE_FACES)),)
    ] * n_agree
    view_segment = _stack(
        [_row_from_groups(_CUBE_FACES, groups) for groups in groups_per_view]
    )
    adjacency = _adjacency(cube_mesh)
    votes = fusion.edge_vote_statistics(view_segment, adjacency, min_votes=2)

    in_a = np.isin(adjacency, np.array(_CUBE_A, dtype=np.int64))
    cross = in_a[:, 0] != in_a[:, 1]
    assert np.array_equal(votes.votes, np.full(18, _N_VIEWS, dtype=np.int64))
    assert votes.observed.all()
    assert np.allclose(votes.weight[cross], expected_weight)
    assert np.allclose(votes.weight[~cross], 1.0)

    labels = fusion.fuse_view_segments(
        cube_mesh.vertices,
        cube_mesh.faces,
        view_segment,
        _all_visible(view_segment),
        merge_threshold=0.5,
    )
    assert _n_parts(labels) == expected_parts
    if expected_parts == 2:
        assert _partition(labels) == [frozenset(_CUBE_A), frozenset(_CUBE_B)]

    end_to_end = _run_static_segmenter(
        cube_mesh,
        groups_per_view,
        static_renderer,
        static_mask_proposer,
        build_block_view,
        masks_from_face_groups,
    )
    assert np.array_equal(end_to_end, labels)


# ---------------------------------------------------------------------------
# ゲート9: 2パス設計の反証ゲート【BL-2】
# ---------------------------------------------------------------------------

# 未観測にする cross 辺(A 側の面, B 側の面)。3 本とも A 側・B 側の面が互いに
# 重複しないので、「A 側を隠す視点」と「B 側を隠す視点」に分けるだけで
# votes=0 を作れる(残る cross 3 本は片側だけが隠れるので観測され続ける)。
_HIDDEN_CROSS_EDGES = ((0, 10), (5, 2), (8, 6))
_HIDDEN_A = (0, 5, 8)
_HIDDEN_B = (10, 2, 6)


def _gate09_view_segment() -> np.ndarray:
    """前半 12 視点で A 側 3 面、後半 12 視点で B 側 3 面を未割当にした `view_segment`。

    こうすると未観測にしたい cross 辺 3 本だけが votes==0 になる。
    """
    remaining_a = tuple(face for face in _CUBE_A if face not in _HIDDEN_A)
    remaining_b = tuple(face for face in _CUBE_B if face not in _HIDDEN_B)
    first_half = _row_from_groups(_CUBE_FACES, (remaining_a, _CUBE_B))
    second_half = _row_from_groups(_CUBE_FACES, (_CUBE_A, remaining_b))
    return _stack([first_half] * 12 + [second_half] * 12)


def test_gate09_two_pass_union_keeps_observed_cuts(cube_mesh: MeshData) -> None:
    """`angle_deg=120` でも観測カットが未観測辺に潰されない(ゲート9 / BL-2)。

    設定: cross 辺 6 本のうち 3 本は**全視点 disagree**(`w=0` → カット確定)、
    残り 3 本は **votes=0**(未観測)。A/B 内部の辺はすべて観測 agree。
    `angle_deg=120` なので幾何プライアは cube の全 18 辺を「滑らか」と判定する。

    **期待: P == 2 が維持される。** v3 の1パス設計(未観測辺に `prior` を混ぜて
    union する)なら未観測 cross 3 本が橋になり **P == 1 に潰れる** — その反実仮想も
    同じ入力で計算して、この設定が本当に「落ちられる」ことを示す。
    """
    adjacency = _adjacency(cube_mesh)
    prior = smooth_edge_mask(
        cube_mesh.vertices, cube_mesh.faces, adjacency, angle_deg=120.0
    )
    assert prior.all()  # 幾何プライアが全辺を繋ぎたがっている状況であることの前提

    view_segment = _gate09_view_segment()
    votes = fusion.edge_vote_statistics(view_segment, adjacency, min_votes=2)

    hidden_rows = [_edge_row(adjacency, a, b) for a, b in _HIDDEN_CROSS_EDGES]
    in_a = np.isin(adjacency, np.array(_CUBE_A, dtype=np.int64))
    cross = in_a[:, 0] != in_a[:, 1]
    observed_cross = np.flatnonzero(cross)
    observed_cross = [row for row in observed_cross if row not in hidden_rows]
    # 前提の実証: 未観測 3 本は votes==0、観測 cross 3 本は w==0、内部辺は w==1。
    assert [int(votes.votes[row]) for row in hidden_rows] == [0, 0, 0]
    assert not votes.observed[hidden_rows].any()
    assert [int(votes.votes[row]) for row in observed_cross] == [12, 12, 12]
    assert np.allclose(votes.weight[observed_cross], 0.0)
    internal = [row for row in range(adjacency.shape[0]) if not cross[row]]
    assert votes.observed[internal].all()
    assert np.allclose(votes.weight[internal], 1.0)

    labels = fusion.fuse_view_segments(
        cube_mesh.vertices,
        cube_mesh.faces,
        view_segment,
        _all_visible(view_segment),
        angle_deg=120.0,
        min_faces=1,
    )
    assert _n_parts(labels) == 2
    assert _partition(labels) == [frozenset(_CUBE_A), frozenset(_CUBE_B)]

    # v3 の1パス設計の反実仮想: 観測辺は投票、未観測辺は幾何プライアで union。
    one_pass_edges = (votes.observed & (votes.weight >= 0.5)) | (
        ~votes.observed & prior
    )
    one_pass = normalize_labels(
        union_find_labels(_CUBE_FACES, adjacency[one_pass_edges])
    )
    assert _n_parts(one_pass) == 1, "the v3 single-pass design must collapse to P==1"


# ---------------------------------------------------------------------------
# ゲート10: 未観測面の吸収(パス2-2)
# ---------------------------------------------------------------------------

# A = {+X, +Y, +Z, -Y}(8 面)/ B = {-X}(2 面)/ -Z(2 面)は全視点で未割当。
_GATE10_A = (0, 1, 4, 5, 6, 7, 8, 9)
_GATE10_B = (2, 3)
_GATE10_UNOBSERVED = (10, 11)


def test_gate10_unobserved_component_is_absorbed_by_the_prior_majority(
    cube_mesh: MeshData,
) -> None:
    """`-Z` が `prior` 境界辺の多い A へ吸収され、`assigned_ratio` も警告(ゲート10)。

    `-Z` の `prior=True` 境界辺は A 側 3 本(+X=0, +Y=4, -Y=7)と B 側 1 本(-X=3)
    なので tie 無しで A へ吸収される。内部対角線辺 (10, 11) も未観測なので、
    パス2-1 が先に `-Z` の 2 面を 1 つの未観測成分にまとめている必要がある。
    """
    adjacency = _adjacency(cube_mesh)
    view_segment = _stack(
        [_row_from_groups(_CUBE_FACES, (_GATE10_A, _GATE10_B))] * _N_VIEWS
    )
    votes = fusion.edge_vote_statistics(view_segment, adjacency, min_votes=2)
    # 前提: `-Z` に触る辺(内部対角線を含む)はすべて未観測。
    touches_unobserved = np.isin(
        adjacency, np.array(_GATE10_UNOBSERVED, dtype=np.int64)
    ).any(axis=1)
    assert not votes.observed[touches_unobserved].any()
    assert votes.observed[~touches_unobserved].all()

    with pytest.warns(UserWarning, match="assigned_ratio") as recorded:
        labels = fusion.fuse_view_segments(
            cube_mesh.vertices,
            cube_mesh.faces,
            view_segment,
            _all_visible(view_segment),
            angle_deg=120.0,
            min_faces=1,
        )
    assert _n_parts(labels) == 2
    assert _partition(labels) == [
        frozenset(_GATE10_A) | frozenset(_GATE10_UNOBSERVED),
        frozenset(_GATE10_B),
    ]
    assert int(labels[10]) == int(labels[0]) and int(labels[11]) == int(labels[0])
    # 10/12 の面が割当を得た(= 0.8333 < assigned_warn 0.90)。
    assert any("0.8333" in str(item.message) for item in recorded)


def test_gate10_unobserved_component_without_prior_edges_stays_its_own_part(
    cube_mesh: MeshData,
) -> None:
    """`prior=True` の境界を 1 本も持たない未観測成分は**残置**される(パス2-3)。

    ゲート10 と同じ入力を `angle_deg=30` で回す。cube の facet 間は二面角 90 度なので
    `prior=False` になり、`-Z` の境界辺 4 本(A 側 3 本・B 側 1 本)はすべて
    「幾何的にも繋がっていない」辺になる。残るのは facet 内の対角線
    (二面角 0 度 → `prior=True`)だけなので、パス2-1 が `-Z` の 2 面を 1 成分に
    まとめ、**パス2-2 は吸収先を 1 つも見つけられない** → 独立した部位として残る。

    **WHY このケースが要るか**(2026-07-30 反証レビュー N1): ゲート10 は
    `angle_deg=120` で cube の全 18 辺が `prior=True` なので、パス2-2 の境界集計から
    `prior &` を落としても結果が変わらなかった(= 仕様「どの観測成分とも
    `prior == True` で繋がらない未観測成分はそのまま残す」を検査するものが皆無)。
    ここでは `prior &` を落とすと `-Z` が A に吸収されて P==2 になり、落ちる。
    """
    adjacency = _adjacency(cube_mesh)
    prior = smooth_edge_mask(
        cube_mesh.vertices, cube_mesh.faces, adjacency, angle_deg=30.0
    )
    touches_unobserved = np.isin(
        adjacency, np.array(_GATE10_UNOBSERVED, dtype=np.int64)
    ).any(axis=1)
    internal = np.array(
        [
            set(row) == set(_GATE10_UNOBSERVED)
            for row in adjacency[touches_unobserved].tolist()
        ]
    )
    # 前提: `-Z` に触る 5 本のうち `prior=True` は内部対角線 1 本だけ。
    assert int(prior.sum()) == 6  # facet 内の対角線 6 本のみ
    assert np.array_equal(prior[touches_unobserved], internal)
    assert int(internal.sum()) == 1

    view_segment = _stack(
        [_row_from_groups(_CUBE_FACES, (_GATE10_A, _GATE10_B))] * _N_VIEWS
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # assigned_ratio はゲート10 で検査済み。
        labels = fusion.fuse_view_segments(
            cube_mesh.vertices,
            cube_mesh.faces,
            view_segment,
            _all_visible(view_segment),
            angle_deg=30.0,
            min_faces=1,
        )
    assert _n_parts(labels) == 3
    assert _partition(labels) == [
        frozenset(_GATE10_A),
        frozenset(_GATE10_B),
        frozenset(_GATE10_UNOBSERVED),
    ]
    # `-Z` の 2 面は 1 つの部位にまとまっている(パス2-1)が、A にも B にも
    # 吸収されていない(パス2-3)。
    assert int(labels[10]) == int(labels[11])
    assert int(labels[10]) != int(labels[0]) and int(labels[10]) != int(labels[2])


# ---------------------------------------------------------------------------
# ゲート11: 幾何プライアへの完全劣化
# ---------------------------------------------------------------------------

# `peanut_mesh` の二面角の最大は 7.62 度なので、これ**未満**のしきい値では表面が
# 細かく割れる(実測: `min_faces=1` で P==338)。`min_faces=None` の自動値
# `max(2, M // 100) = 60` が実際にマージを起こす唯一の設定であり、
# **自動値がずれたら結果が変わる**ことを保証するために使う(2026-07-30 反証
# レビュー B1: 60 度では 3 fixture すべてマージが no-op で、自動値の式を変えても
# ゲート11 が落ちなかった)。
_SPLITTING_ANGLE_DEG = 5.0


@pytest.mark.parametrize(
    "fixture_name", ["cube_mesh", "capped_cylinder_mesh", "peanut_mesh"]
)
@pytest.mark.parametrize(
    ("angle_deg", "min_faces"),
    [(60.0, 1), (60.0, None), (90.0, 3), (_SPLITTING_ANGLE_DEG, None)],
)
def test_gate11_all_unassigned_degrades_to_the_dihedral_segmenter(
    request: pytest.FixtureRequest,
    fixture_name: str,
    angle_deg: float,
    min_faces: int | None,
) -> None:
    """全 `-1` の `view_segment` の出力が `DihedralSegmenter` と一致する(ゲート11)。

    どう壊れたら落ちるか: パス2-1 が `prior` 以外の辺を繋いだ / 幾何プライアを
    `smooth_edge_mask` 以外で組み直した / `min_faces=None` の自動値が
    `DihedralSegmenter` とずれた、のいずれかが起きた瞬間。
    """
    mesh: MeshData = request.getfixturevalue(fixture_name)
    n_faces = len(mesh.faces)
    view_segment = np.full((3, n_faces), fusion.UNASSIGNED, dtype=np.int32)
    with warnings.catch_warnings():
        # 全面未割当なので被覆率の警告は出る(それ自体は別ゲートで検査する)。
        warnings.simplefilter("ignore")
        fused = fusion.fuse_view_segments(
            mesh.vertices,
            mesh.faces,
            view_segment,
            np.ones((3, n_faces), dtype=bool),
            angle_deg=angle_deg,
            min_faces=min_faces,
        )
    geometric = DihedralSegmenter(angle_deg=angle_deg, min_faces=min_faces).segment(
        mesh
    )
    assert np.array_equal(fused, geometric)


def test_gate11_auto_min_faces_is_the_same_formula_as_the_dihedral_segmenter(
    peanut_mesh: MeshData,
) -> None:
    """`min_faces=None` の自動値が **実際にマージを起こす** 設定で一致を固定する。

    **WHY この 1 本が要るか**(2026-07-30 反証レビュー B1): ゲート11 の他の組
    (`angle_deg=60`)では、cube(M=12 → 自動値 2、全部位が 2 面)・
    capped_cylinder(M=128 → 自動値 2、部位 64/32/32)・peanut(P==1 の単一部位)の
    いずれも**小部位マージが no-op** なので、`fusion.resolve_min_faces` の式を
    `M // 100` から `M // 50` に変えても 3 fixture すべて green のままだった。
    つまり `fusion.py` が `geometric.py` の private 定数を import してまで
    「式を 1 箇所に持つ」と主張している根拠を、何も検証していなかった。

    ここでは (a) 自動値そのものを固定し、(b) その値が結果を**変える**こと
    (`min_faces=1` との差)を確認し、(c) その状況で `DihedralSegmenter` と
    ビット一致することを見る。除数や下限を片側だけ変えると (a) か (c) が落ちる。
    """
    n_faces = len(peanut_mesh.faces)
    assert n_faces == 6016
    # (a) 共有された自動値そのもの。`DihedralSegmenter` 側の式(private)と同値。
    assert fusion.resolve_min_faces(None, n_faces) == 60
    assert fusion.resolve_min_faces(None, 100) == 2  # 下限 2 が効く側の点
    assert fusion.resolve_min_faces(7, n_faces) == 7  # 明示値は素通し

    view_segment = np.full((3, n_faces), fusion.UNASSIGNED, dtype=np.int32)
    visible = np.ones((3, n_faces), dtype=bool)

    def _fuse(min_faces: int | None) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fusion.fuse_view_segments(
                peanut_mesh.vertices,
                peanut_mesh.faces,
                view_segment,
                visible,
                angle_deg=_SPLITTING_ANGLE_DEG,
                min_faces=min_faces,
            )

    unmerged = _fuse(1)
    automatic = _fuse(None)
    # (b) 自動値が空回りしていないこと。マージ前 338 部位 → 自動値 60 で 19 部位。
    assert _n_parts(unmerged) == 338
    assert _n_parts(automatic) == 19
    # (c) その状況で幾何バックエンドとビット一致。
    assert np.array_equal(
        automatic,
        DihedralSegmenter(angle_deg=_SPLITTING_ANGLE_DEG, min_faces=None).segment(
            peanut_mesh
        ),
    )


# ---------------------------------------------------------------------------
# ゲート12: `min_votes` の対照
# ---------------------------------------------------------------------------


def test_gate12_min_votes_routes_single_vote_edges_to_the_prior(
    cube_mesh: MeshData,
) -> None:
    """votes==1 の辺だけの入力で、`min_votes=2` は幾何経路・`min_votes=1` は投票経路。

    視点が 1 つだけなので、全面が割当済みでも全辺の `votes` は 1。既定
    `min_votes=2` ではすべて未観測 → 幾何プライアの連結成分(`angle_deg=60` の
    cube は facet ごとに割れて P==6)。`min_votes=1` では投票が効いて A|B の
    P==2 になる。**v3 のゲートは `votes==0` しか見ていなかった**(NB 反映)。
    """
    view_segment = _stack([_row_from_groups(_CUBE_FACES, (_CUBE_A, _CUBE_B))])
    assert view_segment.shape == (1, _CUBE_FACES)
    adjacency = _adjacency(cube_mesh)
    single = fusion.edge_vote_statistics(view_segment, adjacency, min_votes=1)
    assert np.array_equal(single.votes, np.ones(18, dtype=np.int64))

    # 観測辺が 1 本も残らない = ML が寄与しない、という結末は必ず告知される。
    with pytest.warns(UserWarning, match="no adjacency edge reached min_votes=2"):
        prior_path = fusion.fuse_view_segments(
            cube_mesh.vertices,
            cube_mesh.faces,
            view_segment,
            _all_visible(view_segment),
            angle_deg=60.0,
            min_faces=1,
            min_votes=2,
        )
    geometric = DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(cube_mesh)
    assert _n_parts(prior_path) == 6
    assert np.array_equal(prior_path, geometric)

    vote_path = fusion.fuse_view_segments(
        cube_mesh.vertices,
        cube_mesh.faces,
        view_segment,
        _all_visible(view_segment),
        angle_deg=60.0,
        min_faces=1,
        min_votes=1,
    )
    assert _n_parts(vote_path) == 2
    assert _partition(vote_path) == [frozenset(_CUBE_A), frozenset(_CUBE_B)]


def test_min_votes_above_n_views_is_rejected_at_construction(
    static_renderer: type, static_mask_proposer: type
) -> None:
    """`min_votes > n_views` は構築時に `ValueError`(反証レビュー B2)。

    1 本の辺が得られる票の上限は視点数なので、この組み合わせでは**どの辺も観測辺に
    なれず**、出力は必ず `DihedralSegmenter` と同一になる。既定 `min_votes=2` の
    まま `n_views=1` を渡す経路が公開コンストラクタから素通しだった。
    """
    with pytest.raises(ValueError, match="min_votes=2 exceeds n_views=1"):
        MultiViewSegmenter(
            static_mask_proposer([]), lambda _mesh: static_renderer([]), n_views=1
        )
    with pytest.raises(ValueError, match="min_votes=5 exceeds n_views=3"):
        MultiViewSegmenter(
            static_mask_proposer([]),
            lambda _mesh: static_renderer([]),
            n_views=3,
            min_votes=5,
        )
    # 1 視点そのものは禁止ではない — `min_votes=1` なら投票経路が成立する。
    MultiViewSegmenter(
        static_mask_proposer([]),
        lambda _mesh: static_renderer([]),
        n_views=1,
        min_votes=1,
    )


def test_segment_warns_when_no_edge_reaches_min_votes(
    cube_mesh: MeshData,
    build_block_view,
    masks_from_face_groups,
    static_renderer: type,
    static_mask_proposer: type,
) -> None:
    """票が `min_votes` に届かず幾何プライアだけになったら警告する(反証レビュー B2)。

    視点 0 は A だけ、視点 1 は B だけをマスクに入れる。`min_votes=2 <= n_views=2`
    なので構築時ガードには掛からないが、**どの辺も両端そろって 2 票に届かない**ので
    結果は `DihedralSegmenter` とビット同一になる。全面が可視かつどこかの視点で
    割当済みなので `visible_warn` / `assigned_warn` は鳴らない — この経路が
    **警告ゼロ**だったのが B2 の指摘で、K=0 経路と同じ結末には同じ告知を出す。
    """
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        labels = _run_static_segmenter(
            cube_mesh,
            [(_CUBE_A,), (_CUBE_B,)],
            static_renderer,
            static_mask_proposer,
            build_block_view,
            masks_from_face_groups,
            min_faces=1,
        )
    messages = [str(item.message) for item in recorded]
    assert any("no adjacency edge reached min_votes=2" in m for m in messages)
    assert any("geometric prior alone" in m for m in messages)
    # 被覆率の警告は鳴らない(全面可視・全面どこかで割当済み)。
    assert not any("visible_ratio" in m or "assigned_ratio" in m for m in messages)
    assert np.array_equal(
        labels, DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(cube_mesh)
    )


# ---------------------------------------------------------------------------
# ゲート13: 被覆率の分離
# ---------------------------------------------------------------------------


def _fuse_and_record(
    mesh: MeshData, view_segment: np.ndarray, view_visible: np.ndarray
) -> list[str]:
    """融合を回し、出た警告メッセージを文字列のリストで返す。"""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fusion.fuse_view_segments(
            mesh.vertices, mesh.faces, view_segment, view_visible, min_faces=1
        )
    return [str(item.message) for item in recorded]


def test_gate13_visible_and_assigned_ratios_warn_independently(
    cube_mesh: MeshData,
) -> None:
    """(i) 不可視面 → `visible_ratio` のみ、(ii) 未割当面 → `assigned_ratio` のみ。"""
    # (i) 面 10/11 が全視点で face_id に現れない。残る 10 面はすべて割当済み。
    #
    # **不可視を 2 枚にしてある WHY**(2026-07-30 反証レビュー N2): 1 枚だと
    # `assigned_ratio` の分母を「可視面」から「全面」へ変える変異でも
    # `11/12 = 0.9167 >= assigned_warn(0.90)` で第2警告が鳴らず、判別できなかった。
    # 2 枚なら全面分母では `10/12 = 0.8333 < 0.90` になって第2警告が鳴るので、
    # 「分母は可視面」という設計判断(`fusion._warn_low_coverage` の WHY)がここで
    # 初めてゲートに固定される。実装では可視 10 面中 10 面が割当なので 1.0 = 無警告。
    invisible_segment = _stack([_row_from_groups(_CUBE_FACES, (tuple(range(10)),))] * 4)
    invisible_visible = np.ones((4, _CUBE_FACES), dtype=bool)
    invisible_visible[:, 10] = False
    invisible_visible[:, 11] = False
    messages = _fuse_and_record(cube_mesh, invisible_segment, invisible_visible)
    assert any("visible_ratio=0.8333" in message for message in messages)
    assert not any("assigned_ratio" in message for message in messages)

    # (ii) 全面が見えているが、面 10/11 はどのマスクにも入らない。
    unassigned_segment = _stack(
        [_row_from_groups(_CUBE_FACES, (tuple(range(10)),))] * 4
    )
    messages = _fuse_and_record(
        cube_mesh, unassigned_segment, np.ones((4, _CUBE_FACES), dtype=bool)
    )
    assert any("assigned_ratio=0.8333" in message for message in messages)
    assert not any("visible_ratio" in message for message in messages)


def test_assigned_face_that_is_invisible_is_rejected(cube_mesh: MeshData) -> None:
    """割当済みなのに全視点で不可視、という矛盾入力は `ValueError`(不変条件)。"""
    view_segment = _stack([_row_from_groups(_CUBE_FACES, (_CUBE_A, _CUBE_B))] * 2)
    view_visible = np.ones((2, _CUBE_FACES), dtype=bool)
    view_visible[:, 0] = False
    with pytest.raises(
        ValueError, match="carry a mask assignment while being invisible"
    ):
        fusion.fuse_view_segments(
            cube_mesh.vertices, cube_mesh.faces, view_segment, view_visible
        )


# ---------------------------------------------------------------------------
# ゲート14: 決定性
# ---------------------------------------------------------------------------


def test_gate14_fusion_and_segment_are_deterministic(
    cube_mesh: MeshData,
    build_block_view,
    masks_from_face_groups,
    static_renderer: type,
    static_mask_proposer: type,
) -> None:
    """同一入力を 2 回通すと `np.array_equal`(融合レベルと E2E の両方)。"""
    view_segment = _gate09_view_segment()
    first = fusion.fuse_view_segments(
        cube_mesh.vertices,
        cube_mesh.faces,
        view_segment,
        _all_visible(view_segment),
        angle_deg=120.0,
        min_faces=1,
    )
    second = fusion.fuse_view_segments(
        cube_mesh.vertices,
        cube_mesh.faces,
        view_segment,
        _all_visible(view_segment),
        angle_deg=120.0,
        min_faces=1,
    )
    assert np.array_equal(first, second)

    groups = [(_CUBE_A, _CUBE_B)] * _N_VIEWS
    runs = [
        _run_static_segmenter(
            cube_mesh,
            groups,
            static_renderer,
            static_mask_proposer,
            build_block_view,
            masks_from_face_groups,
        )
        for _ in range(2)
    ]
    assert np.array_equal(runs[0], runs[1])


# ---------------------------------------------------------------------------
# ゲート15: `peanut_mesh` の前提
# ---------------------------------------------------------------------------


def test_gate15_peanut_degenerates_to_one_part_under_the_geometric_prior(
    peanut_mesh: MeshData, peanut_truth_labels
) -> None:
    """`DihedralSegmenter(angle_deg=60, min_faces=1)` → P == 1(ゲート15)。

    Step 2-5 の主 ML ゲートの**前提条件**: 接合部に鋭い折れ目が無いので幾何だけでは
    割れない。fixture が閉多様体(E == 3F/2)で、真値ラベルがくびれで二分される
    ことも併せて実証する(前提が崩れたら ML ゲートが空虚になる)。
    """
    adjacency = _adjacency(peanut_mesh)
    n_faces = len(peanut_mesh.faces)
    assert n_faces == 6016
    assert adjacency.shape[0] == 3 * n_faces // 2  # 閉多様体
    assert len(np.unique(weld_vertices(np.asarray(peanut_mesh.vertices)))) == 3010

    labels = DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(peanut_mesh)
    assert _n_parts(labels) == 1

    truth = peanut_truth_labels(peanut_mesh)
    assert np.array_equal(np.unique(truth), np.array([0, 1]))
    assert int((truth == 0).sum()) == int((truth == 1).sum()) == n_faces // 2


# ---------------------------------------------------------------------------
# ゲート16: sabotage(heavy-artillery)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broken",
    [
        np.array([0, 2, 0, 2] * 3, dtype=np.int64),  # 非連番
        np.array([-1] + [0] * 11, dtype=np.int64),  # 負値
        np.zeros(_CUBE_FACES, dtype=np.float64),  # float dtype
    ],
)
def test_gate16a_label_contract_rejects_corrupted_labels(broken: np.ndarray) -> None:
    """`labels.validate_labels` が非連番/負値/float を弾く(検証が落ちられる証明)。"""
    with pytest.raises(ValueError):
        validate_labels(broken, _CUBE_FACES)


def test_gate16b_corrupted_fusion_output_is_caught(
    cube_mesh: MeshData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """段階E の後段に故意の未ラベル面を作ると最終検証が落ちる(ゲート16(b))。

    `fusion.normalize_labels` を「1 面だけ `-1` にする」実装へ差し替える。
    `fuse_view_segments` 末尾の `validate_labels` が無ければ、この壊れたラベルが
    そのまま pack へ流れる。
    """

    def _sabotage(labels: np.ndarray) -> np.ndarray:
        corrupted = normalize_labels(labels).copy()
        corrupted[0] = -1
        return corrupted

    monkeypatch.setattr(fusion, "normalize_labels", _sabotage)
    view_segment = _stack([_row_from_groups(_CUBE_FACES, (_CUBE_A, _CUBE_B))] * 2)
    with pytest.raises(ValueError, match="consecutive 0..P-1"):
        fusion.fuse_view_segments(
            cube_mesh.vertices,
            cube_mesh.faces,
            view_segment,
            _all_visible(view_segment),
        )


# ---------------------------------------------------------------------------
# ゲート17: 入口検証
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"assign_ratio": 0.0}, r"assign_ratio must be in \(0, 1\]"),
        ({"assign_ratio": -0.1}, r"assign_ratio must be in \(0, 1\]"),
        ({"assign_ratio": 1.5}, r"assign_ratio must be in \(0, 1\]"),
        ({"assign_ratio": float("nan")}, "assign_ratio must be finite"),
        ({"merge_threshold": -0.01}, r"merge_threshold must be in \[0, 1\]"),
        ({"merge_threshold": 1.01}, r"merge_threshold must be in \[0, 1\]"),
        ({"merge_threshold": float("inf")}, "merge_threshold must be finite"),
        ({"min_votes": 0}, "min_votes must be >= 1"),
        ({"min_votes": -3}, "min_votes must be >= 1"),
        ({"min_votes": 1.5}, "min_votes must be an int"),
        ({"min_faces": 0}, "min_faces must be >= 1"),
        ({"angle_deg": 0.0}, r"angle_deg must be in \(0, 180\]"),
        ({"angle_deg": 181.0}, r"angle_deg must be in \(0, 180\]"),
        ({"visible_warn": 1.2}, r"visible_warn must be in \[0, 1\]"),
        ({"assigned_warn": -0.5}, r"assigned_warn must be in \[0, 1\]"),
        ({"max_masks_per_view": 0}, "max_masks_per_view must be >= 1"),
        ({"n_views": 0}, "n_views must be >= 1"),
        ({"projection": "fisheye"}, "unknown projection"),
    ],
)
def test_gate17_segmenter_rejects_out_of_range_parameters(
    kwargs: dict[str, object],
    match: str,
    static_renderer: type,
    static_mask_proposer: type,
) -> None:
    """`MultiViewSegmenter.__init__` が全パラメータを構築時に検証する(ゲート17)。"""
    with pytest.raises(ValueError, match=match):
        MultiViewSegmenter(
            static_mask_proposer([]),
            lambda _mesh: static_renderer([]),
            **kwargs,  # type: ignore[arg-type]
        )


def test_gate17_view_segment_shape_and_dtype_are_checked(cube_mesh: MeshData) -> None:
    """`view_segment` / `view_visible` の shape/dtype 違反は `ValueError`(ゲート17)。"""
    good = _stack([_row_from_groups(_CUBE_FACES, (_CUBE_A, _CUBE_B))] * 2)
    visible = _all_visible(good)
    with pytest.raises(ValueError, match="dtype int32"):
        fusion.fuse_view_segments(
            cube_mesh.vertices, cube_mesh.faces, good.astype(np.int64), visible
        )
    with pytest.raises(ValueError, match=r"one column per face"):
        fusion.fuse_view_segments(
            cube_mesh.vertices, cube_mesh.faces, good[:, :5].copy(), visible[:, :5]
        )
    with pytest.raises(ValueError, match=r"shape \(V, M\)"):
        fusion.fuse_view_segments(
            cube_mesh.vertices, cube_mesh.faces, good[0], visible[0]
        )
    with pytest.raises(ValueError, match="at least one view"):
        fusion.fuse_view_segments(
            cube_mesh.vertices,
            cube_mesh.faces,
            good[:0].copy(),
            visible[:0].copy(),
        )
    with pytest.raises(ValueError, match="view_visible must have dtype bool"):
        fusion.fuse_view_segments(
            cube_mesh.vertices, cube_mesh.faces, good, visible.astype(np.int8)
        )
    with pytest.raises(ValueError, match="same shape as view_segment"):
        fusion.fuse_view_segments(
            cube_mesh.vertices, cube_mesh.faces, good, visible[:1]
        )


def test_gate17_too_many_masks_is_rejected() -> None:
    """`K > max_masks_per_view` は `ValueError`(ゲート17)。"""
    masks = np.zeros((5, 4, 4), dtype=bool)
    masks[:, 0, 0] = True
    with pytest.raises(ValueError, match="exceeds max_masks_per_view=4"):
        fusion.normalize_masks(masks, max_masks_per_view=4)
    # 上限そのものは通る(境界は閉)。
    assert fusion.normalize_masks(masks, max_masks_per_view=5).shape[0] == 1


def test_gate17_broken_view_buffers_are_rejected() -> None:
    """段階A が `coverage <=> face_id >= 0` の破れと面ID超過を弾く(§2.6)。"""
    face_id, coverage = _single_face_view(interior=4)
    masks = _masks_over_face_pixels(face_id, (16,))
    broken_coverage = coverage.copy()
    broken_coverage[0, 0] = True  # 背景画素に前景フラグ
    with pytest.raises(ValueError, match="coverage <=> "):
        fusion.assign_view_faces(masks, face_id, broken_coverage, n_faces=1)
    with pytest.raises(ValueError, match="but the mesh has only 0 face"):
        fusion.assign_view_faces(masks, face_id, coverage, n_faces=0)
    with pytest.raises(ValueError, match="masks must have shape"):
        fusion.assign_view_faces(masks[:, :2, :2], face_id, coverage, n_faces=1)
    with pytest.raises(ValueError, match="masks must have dtype bool"):
        fusion.normalize_masks(masks.astype(np.int8))


# ---------------------------------------------------------------------------
# 追加の契約テスト(寿命・Protocol 適合・非破壊)
# ---------------------------------------------------------------------------


def test_segment_without_entering_the_context_manager_is_allowed(
    cube_mesh: MeshData,
    build_block_view,
    masks_from_face_groups,
    static_renderer: type,
    static_mask_proposer: type,
) -> None:
    """未 `__enter__` の `segment()` は許され、`__exit__` 後は `RuntimeError`(裁定2)。

    `MeshRenderer` とは非対称であることの固定: renderer は `__enter__` で GL を
    作るので未入場では動けないが、segmenter は `__init__` で受け取った proposer
    だけで動ける。`rebake(segmentation=make_sam2_segmenter())` を `with` 無しで
    呼ぶ経路(「`rebake` は注入 backend を閉じない」契約)が現実に踏まれるため。
    """
    view = build_block_view(_CUBE_FACES)
    masks = [masks_from_face_groups(view.face_id, (_CUBE_A, _CUBE_B))] * _N_VIEWS
    renderers: list[object] = []

    def _factory(_mesh: MeshData) -> object:
        renderer = static_renderer([view] * _N_VIEWS)
        renderers.append(renderer)
        return renderer

    proposer = static_mask_proposer(masks * 2)
    segmenter = MultiViewSegmenter(proposer, _factory)  # type: ignore[arg-type]
    unentered = segmenter.segment(cube_mesh)
    assert _n_parts(unentered) == 2
    assert proposer.close_count == 0  # 未入場なので誰も閉じていない

    with segmenter as entered:
        assert entered is segmenter
        assert np.array_equal(entered.segment(cube_mesh), unentered)
    assert proposer.close_count == 1
    # renderer は segment() ごとに生成・破棄される(所有していない)。
    assert len(renderers) == 2
    assert all(
        renderer.enter_count == 1 and renderer.exit_count == 1  # type: ignore[attr-defined]
        for renderer in renderers
    )

    with pytest.raises(RuntimeError, match="has been closed"):
        segmenter.segment(cube_mesh)
    with pytest.raises(RuntimeError, match="has been closed"):
        with segmenter:
            pass
    segmenter.close()  # 冪等
    assert proposer.close_count == 1


def test_renderer_is_closed_even_when_the_proposer_raises(
    cube_mesh: MeshData,
    build_block_view,
    static_renderer: type,
) -> None:
    """提案器が例外を投げても renderer の `__exit__` は必ず通る。"""

    class _Boom:
        def propose(self, view: object) -> np.ndarray:
            raise RuntimeError("proposer exploded")

        def close(self) -> None:
            return None

    renderer = static_renderer([build_block_view(_CUBE_FACES)] * _N_VIEWS)
    segmenter = MultiViewSegmenter(_Boom(), lambda _mesh: renderer)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="proposer exploded"):
        segmenter.segment(cube_mesh)
    assert renderer.exit_count == 1


def test_segment_warns_when_no_mask_is_returned_at_all(
    cube_mesh: MeshData,
    build_block_view,
    static_renderer: type,
    static_mask_proposer: type,
) -> None:
    """全視点で K==0 なら「ML が寄与しなかった」警告 + 幾何バックエンド一致(§2.6)。"""
    view = build_block_view(_CUBE_FACES)
    empty = np.zeros((0, *view.face_id.shape), dtype=bool)
    proposer = static_mask_proposer([empty] * _N_VIEWS)
    renderer = static_renderer([view] * _N_VIEWS)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with MultiViewSegmenter(
            proposer, lambda _mesh: renderer, min_faces=1
        ) as segmenter:
            labels = segmenter.segment(cube_mesh)
    messages = [str(item.message) for item in recorded]
    assert any("returned no masks for any of the 24 views" in m for m in messages)
    assert any("assigned_ratio=0.0000" in m for m in messages)
    assert not any("visible_ratio" in m for m in messages)  # 全面が見えている
    assert np.array_equal(
        labels, DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(cube_mesh)
    )


def test_one_mask_over_disconnected_faces_still_yields_separate_parts(
    capped_cylinder_mesh: MeshData,
    build_block_view,
    masks_from_face_groups,
    static_renderer: type,
    static_mask_proposer: type,
) -> None:
    """同一マスクに入れても隣接していない面集合は別部位になる(段階C は隣接上の union)。

    円筒の 2 つのキャップを 1 枚のマスクに入れ、側面を別マスクにする。キャップ
    どうしは面隣接を 1 本も共有しないので、投票がどれだけ一致しても union は
    起こらない — BL-1 が `two_cubes_mesh` を融合ゲートから外した理由そのもの
    (「マスクが同じなら繋がる」実装に変えるとここが落ちる)。
    """
    corners = np.asarray(capped_cylinder_mesh.vertices)[
        np.asarray(capped_cylinder_mesh.faces)
    ]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    axis = normals[:, 2] / np.linalg.norm(normals, axis=1)
    top = tuple(np.flatnonzero(axis >= 0.5).tolist())
    bottom = tuple(np.flatnonzero(axis <= -0.5).tolist())
    side = tuple(np.flatnonzero(np.abs(axis) < 0.5).tolist())
    assert (len(side), len(top), len(bottom)) == (64, 32, 32)

    labels = _run_static_segmenter(
        capped_cylinder_mesh,
        [(side, top + bottom)] * _N_VIEWS,
        static_renderer,
        static_mask_proposer,
        build_block_view,
        masks_from_face_groups,
        min_faces=1,
    )
    assert _n_parts(labels) == 3
    assert sorted(np.bincount(labels).tolist()) == [32, 32, 64]
    assert int(labels[top[0]]) != int(labels[bottom[0]])


def test_multiview_segmenter_satisfies_the_backend_protocol(
    static_renderer: type, static_mask_proposer: type
) -> None:
    """`SegmentationBackend`(Protocol)への構造的適合。

    `SegmentationBackend` は `@runtime_checkable` ではないので `isinstance` では
    見られない(既存 `test_segmentation_geometric.py:66-68` と同じ形にする)。
    """
    backend: SegmentationBackend = MultiViewSegmenter(
        static_mask_proposer([]), lambda _mesh: static_renderer([])
    )
    assert callable(backend.segment)


def test_empty_mesh_returns_empty_labels_without_building_a_renderer(
    static_renderer: type, static_mask_proposer: type
) -> None:
    """面数 0 のメッシュは renderer を起こさずに `(0,)` を返す。"""
    calls: list[MeshData] = []

    def _factory(mesh: MeshData) -> object:
        calls.append(mesh)
        raise AssertionError("the renderer must not be built for an empty mesh")

    # `MeshData` は `N >= 1` を要求する(`types.py:63`)ので、頂点は 1 個残して
    # 面だけを空にする = 「描くものが無いメッシュ」。
    mesh = MeshData(
        vertices=np.zeros((1, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
        source_vertex=np.zeros(1, dtype=np.int64),
    )
    segmenter = MultiViewSegmenter(static_mask_proposer([]), _factory)  # type: ignore[arg-type]
    labels = segmenter.segment(mesh)
    assert labels.shape == (0,) and labels.dtype == np.int64
    assert calls == []


def test_fusion_does_not_mutate_its_arguments(
    cube_mesh: MeshData,
    build_block_view,
    masks_from_face_groups,
) -> None:
    """引数非破壊: 頂点・面・`view_segment`・マスクを 1 要素も書き換えない。"""
    view = build_block_view(_CUBE_FACES)
    masks = masks_from_face_groups(view.face_id, (_CUBE_A, _CUBE_B))
    view_segment = _stack([_row_from_groups(_CUBE_FACES, (_CUBE_A, _CUBE_B))] * 2)
    visible = _all_visible(view_segment)
    snapshots = [
        np.asarray(cube_mesh.vertices).copy(),
        np.asarray(cube_mesh.faces).copy(),
        view_segment.copy(),
        visible.copy(),
        masks.copy(),
        view.face_id.copy(),
    ]
    fusion.normalize_masks(masks)
    fusion.assign_view_faces(masks, view.face_id, view.coverage, n_faces=_CUBE_FACES)
    fusion.fuse_view_segments(
        cube_mesh.vertices, cube_mesh.faces, view_segment, visible
    )
    live = [
        np.asarray(cube_mesh.vertices),
        np.asarray(cube_mesh.faces),
        view_segment,
        visible,
        masks,
        view.face_id,
    ]
    for before, after in zip(snapshots, live):
        assert np.array_equal(before, after)
