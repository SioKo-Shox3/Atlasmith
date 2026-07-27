"""`segmentation/geometric.py` — `DihedralSegmenter` のテスト。

計画v4 §5 Step 2-2 の合否基準のうち、ラベル契約(項目2)・連結成分(3)・
二面角カット(4)・退化保証(5)・マージ(6)・決定性(7)・パラメータ検証(9)・
負の対照(10)をここで閉じる(項目1 と 8 は `test_segmentation_adjacency.py`)。
"""

from __future__ import annotations

import numpy as np
import pytest

from atlasmith.segmentation import DihedralSegmenter, SegmentationBackend
from atlasmith.types import MeshData

_ALL_FIXTURES = [
    "cube_mesh",
    "two_cubes_mesh",
    "capped_cylinder_mesh",
    "sphere_mesh",
    "torus_mesh",
    "nonmanifold_mesh",
    "zero_area_mesh",
]


def _canonical_partition(values: np.ndarray) -> list[frozenset[int]]:
    """同値類を「最小要素の昇順」で並べる(production を使わない独立実装)。

    ラベル値そのものではなく **面の分割の仕方** だけを比較するための正規形。
    """
    groups: dict[int, set[int]] = {}
    for index, value in enumerate(values.tolist()):
        groups.setdefault(value, set()).add(index)
    return [frozenset(group) for group in sorted(groups.values(), key=min)]


def _independent_face_normals(mesh: MeshData) -> np.ndarray:
    """テスト側の独立法線計算(`adjacency.face_normals` を import しない)。"""
    corners = mesh.vertices[mesh.faces]
    raw = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 合否基準2: ラベル契約
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", _ALL_FIXTURES)
def test_labels_satisfy_the_backend_contract(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """全 fixture で shape (M,) / dtype int64 / 値集合 == {0..P-1} の連番。"""
    mesh: MeshData = request.getfixturevalue(fixture_name)
    labels = DihedralSegmenter().segment(mesh)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(mesh.faces),)
    assert labels.dtype == np.int64
    n_parts = len(np.unique(labels))
    assert np.array_equal(np.unique(labels), np.arange(n_parts, dtype=np.int64))
    assert n_parts >= 1


def test_dihedral_segmenter_satisfies_the_protocol() -> None:
    """`SegmentationBackend`(Protocol)への構造的適合。"""
    backend: SegmentationBackend = DihedralSegmenter()
    assert callable(backend.segment)


def test_segment_does_not_mutate_the_input_mesh(cube_mesh: MeshData) -> None:
    vertices = cube_mesh.vertices.copy()
    faces = cube_mesh.faces.copy()
    uv = cube_mesh.uv.copy()
    DihedralSegmenter().segment(cube_mesh)
    assert np.array_equal(cube_mesh.vertices, vertices)
    assert np.array_equal(cube_mesh.faces, faces)
    assert np.array_equal(cube_mesh.uv, uv)


def test_segment_of_face_less_mesh_returns_empty_labels() -> None:
    mesh = MeshData(
        vertices=np.zeros((1, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    labels = DihedralSegmenter().segment(mesh)
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


# ---------------------------------------------------------------------------
# 合否基準3: 連結成分の分離
# ---------------------------------------------------------------------------


def test_two_cubes_split_into_two_parts_at_120_degrees(
    two_cubes_mesh: MeshData,
) -> None:
    labels = DihedralSegmenter(angle_deg=120.0, min_faces=1).segment(two_cubes_mesh)
    assert len(np.unique(labels)) == 2
    assert _canonical_partition(labels) == [
        frozenset(range(0, 12)),
        frozenset(range(12, 24)),
    ]


def test_single_cube_stays_one_part_at_120_degrees(cube_mesh: MeshData) -> None:
    """立方体の稜線は 90 度なので 120 度なら切れない(しきい値の上側対照)。"""
    labels = DihedralSegmenter(angle_deg=120.0, min_faces=1).segment(cube_mesh)
    assert len(np.unique(labels)) == 1


def test_threshold_is_inclusive_at_exactly_ninety_degrees(cube_mesh: MeshData) -> None:
    """`角度 <= angle_deg` の `<=` 意味論を固定する(90 度ちょうどで繋がる)。"""
    assert (
        len(
            np.unique(DihedralSegmenter(angle_deg=90.0, min_faces=1).segment(cube_mesh))
        )
        == 1
    )
    assert (
        len(
            np.unique(DihedralSegmenter(angle_deg=89.9, min_faces=1).segment(cube_mesh))
        )
        == 6
    )


# ---------------------------------------------------------------------------
# 合否基準4: 二面角カット
# ---------------------------------------------------------------------------


def test_cube_splits_into_six_facets_at_60_degrees(cube_mesh: MeshData) -> None:
    labels = DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(cube_mesh)
    assert len(np.unique(labels)) == 6
    # 期待部位 = conftest の facet 定義そのもの(面 2i と 2i+1 が同一 facet)。
    assert _canonical_partition(labels) == [
        frozenset({2 * facet, 2 * facet + 1}) for facet in range(6)
    ]


def test_capped_cylinder_splits_into_side_and_two_caps_at_60_degrees(
    capped_cylinder_mesh: MeshData,
) -> None:
    """側面 / +Z キャップ / -Z キャップ の 3 部位に、面法線分類と完全一致で割れる。"""
    labels = DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(
        capped_cylinder_mesh
    )
    assert len(np.unique(labels)) == 3

    # 期待部位を独立に構成する: 面法線と軸 (z) の内積で 3 分類。
    axis_dot = _independent_face_normals(capped_cylinder_mesh)[:, 2]
    expected = np.where(axis_dot >= 0.5, 0, np.where(axis_dot <= -0.5, 1, 2))
    assert np.bincount(expected, minlength=3).tolist() == [32, 32, 64]
    assert _canonical_partition(labels) == _canonical_partition(expected)


# ---------------------------------------------------------------------------
# 合否基準5: 退化保証(滑らかな閉曲面は P == 1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["sphere_mesh", "torus_mesh"])
def test_smooth_closed_surfaces_degrade_to_a_single_part(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    mesh: MeshData = request.getfixturevalue(fixture_name)
    labels = DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(mesh)
    assert len(np.unique(labels)) == 1


# ---------------------------------------------------------------------------
# 合否基準6: 小部位マージ
# ---------------------------------------------------------------------------


def test_cube_facets_merge_into_one_part_when_min_faces_is_four(
    cube_mesh: MeshData,
) -> None:
    """facet はどれも 2 面なので `min_faces=4` では全部が吸収されて P == 1。"""
    labels = DihedralSegmenter(angle_deg=60.0, min_faces=4).segment(cube_mesh)
    assert len(np.unique(labels)) == 1


def test_min_faces_one_disables_merging(capped_cylinder_mesh: MeshData) -> None:
    """`min_faces=1` はマージ無効(4 の期待値がマージ由来でないことの対照)。"""
    strict = DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(
        capped_cylinder_mesh
    )
    # 自動 min_faces は max(2, 128 // 100) = 2。最小部位は 32 面なので変化しない。
    automatic = DihedralSegmenter(angle_deg=60.0).segment(capped_cylinder_mesh)
    assert np.array_equal(strict, automatic)


def test_min_faces_larger_than_mesh_collapses_to_one_part(
    capped_cylinder_mesh: MeshData,
) -> None:
    labels = DihedralSegmenter(angle_deg=60.0, min_faces=1000).segment(
        capped_cylinder_mesh
    )
    assert len(np.unique(labels)) == 1


# ---------------------------------------------------------------------------
# 合否基準7: 決定性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", _ALL_FIXTURES)
def test_segmentation_is_bit_deterministic(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    mesh: MeshData = request.getfixturevalue(fixture_name)
    first = DihedralSegmenter(angle_deg=60.0).segment(mesh)
    second = DihedralSegmenter(angle_deg=60.0).segment(mesh)
    assert np.array_equal(first, second)
    # 同一インスタンスの再呼び出しでも同じ(内部状態を持ち越さない)。
    segmenter = DihedralSegmenter(angle_deg=60.0)
    assert np.array_equal(segmenter.segment(mesh), segmenter.segment(mesh))


# ---------------------------------------------------------------------------
# 合否基準9: パラメータ検証
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle_deg", [float("nan"), 0.0, -1.0, 181.0, float("inf")])
def test_invalid_angle_deg_raises_value_error(angle_deg: float) -> None:
    with pytest.raises(ValueError, match="angle_deg"):
        DihedralSegmenter(angle_deg=angle_deg)


@pytest.mark.parametrize("min_faces", [0, -1])
def test_invalid_min_faces_raises_value_error(min_faces: int) -> None:
    with pytest.raises(ValueError, match="min_faces must be >= 1"):
        DihedralSegmenter(min_faces=min_faces)


def test_non_integer_min_faces_raises_value_error() -> None:
    with pytest.raises(ValueError, match="min_faces must be None or an int"):
        DihedralSegmenter(min_faces=2.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("angle_deg", [1e-9, 60.0, 180.0])
def test_boundary_angles_are_accepted(angle_deg: float) -> None:
    assert DihedralSegmenter(angle_deg=angle_deg).angle_deg == angle_deg


def test_defaults_match_the_plan() -> None:
    segmenter = DihedralSegmenter()
    assert segmenter.angle_deg == 60.0
    assert segmenter.min_faces is None


# ---------------------------------------------------------------------------
# 合否基準10: 負の対照
# ---------------------------------------------------------------------------


def test_faces_across_a_nonmanifold_edge_land_in_different_parts(
    nonmanifold_mesh: MeshData,
) -> None:
    """(a) 3 面が共有する辺は、最も緩い 180 度でも部位を繋がない。"""
    labels = DihedralSegmenter(angle_deg=180.0, min_faces=1).segment(nonmanifold_mesh)
    assert labels.tolist() == [0, 1, 2]
    # 面 0 と 面 1 は同一法線(角度 0 度)。非多様体カットが無ければ必ず結合する。
    normals = _independent_face_normals(nonmanifold_mesh)
    assert np.allclose(normals[0], normals[1])


def test_zero_area_face_becomes_an_isolated_part(zero_area_mesh: MeshData) -> None:
    """(b) 零面積面は隣接があっても **二面角段では** 孤立部位になる。

    **`min_faces=1` が必須である理由(2つの「必ずカット」規約の非対称性)**:
    小部位マージには *カット前* の全 manifold 隣接を渡す設計なので(`segment` の
    コメント参照 — これを変えると
    `test_cube_facets_merge_into_one_part_when_min_faces_is_four` が落ちる)、
    零面積面は「隣接はあるが二面角段で切られた 1 面の部位」として残り、既定
    `min_faces`(ここでは `max(2, 3 // 100) = 2`)ではマージ段が拾って吸収し直す:

        DihedralSegmenter().segment(zero_area_mesh)            -> [0, 0, 0]
        DihedralSegmenter(min_faces=1).segment(zero_area_mesh) -> [0, 0, 1]

    非多様体辺のほうは *隣接そのものが作られない* ので同じことが起きない。
    つまり2つの規約は既定パラメータ下で非対称に振る舞う。パックへの実害は薄い
    (零面積面は UV 面積も 0)ため挙動は変えず、ここに記録するに留める。
    """
    labels = DihedralSegmenter(angle_deg=180.0, min_faces=1).segment(zero_area_mesh)
    assert labels.tolist() == [0, 0, 1]
    # 上記の非対称性そのものを固定する(既定では吸収されて P == 1 になる)。
    assert DihedralSegmenter(angle_deg=180.0).segment(zero_area_mesh).tolist() == [
        0,
        0,
        0,
    ]
