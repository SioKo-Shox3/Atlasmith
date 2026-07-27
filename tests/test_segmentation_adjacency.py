"""`segmentation/adjacency.py` — 位置weld・面隣接・二面角のテスト。

計画v4 §5 Step 2-2 の合否基準のうち、weld 正の対照(項目1)と weld 隣接構造の
明示検査(項目8)をここで閉じる。
"""

from __future__ import annotations

import numpy as np
import pytest

from atlasmith.segmentation.adjacency import (
    build_face_adjacency,
    dihedral_angles,
    face_normals,
    smooth_edge_mask,
    weld_vertices,
)
from atlasmith.types import MeshData

# ---------------------------------------------------------------------------
# 位置weld
# ---------------------------------------------------------------------------


def test_weld_groups_of_cube_are_exactly_eight(cube_mesh: MeshData) -> None:
    """合否基準1(weld 正の対照): 24 頂点の cube fixture が 8 グループに畳まれる。"""
    weld_map = weld_vertices(cube_mesh.vertices)
    assert weld_map.shape == (24,)
    assert weld_map.dtype == np.int64
    assert len(np.unique(weld_map)) == 8


def test_weld_representative_is_the_minimum_index_of_its_group(
    cube_mesh: MeshData,
) -> None:
    """代表は「グループ内で最小の元頂点 index」であること。"""
    weld_map = weld_vertices(cube_mesh.vertices)
    for representative in np.unique(weld_map):
        members = np.flatnonzero(weld_map == representative)
        assert members.min() == representative
        # 代表自身は自分を指す(写像が冪等)。
        assert weld_map[representative] == representative


def test_weld_groups_have_bit_identical_positions(cube_mesh: MeshData) -> None:
    """同一グループの頂点は座標が厳密一致し、別グループとは一致しないこと。"""
    vertices = cube_mesh.vertices
    weld_map = weld_vertices(vertices)
    for index, representative in enumerate(weld_map.tolist()):
        assert np.array_equal(vertices[index], vertices[representative])
    representatives = np.unique(weld_map)
    unique_positions = {tuple(vertices[r].tolist()) for r in representatives}
    assert len(unique_positions) == len(representatives)


def test_weld_groups_of_two_cubes_are_sixteen(two_cubes_mesh: MeshData) -> None:
    """2 個の立方体は位置を共有しないので 8 + 8 = 16 グループ。"""
    assert len(np.unique(weld_vertices(two_cubes_mesh.vertices))) == 16


def test_weld_folds_unrolled_cylinder_back_to_66_vertices(
    capped_cylinder_mesh: MeshData,
) -> None:
    """面ごとにアンロールした 384 頂点が trimesh 出力の 66 頂点へ畳み戻ること。"""
    assert capped_cylinder_mesh.vertices.shape == (384, 3)
    assert len(np.unique(weld_vertices(capped_cylinder_mesh.vertices))) == 66


def test_weld_is_deterministic_and_non_destructive(cube_mesh: MeshData) -> None:
    original = cube_mesh.vertices.copy()
    first = weld_vertices(cube_mesh.vertices)
    second = weld_vertices(cube_mesh.vertices)
    assert np.array_equal(first, second)
    assert np.array_equal(cube_mesh.vertices, original)


def test_weld_of_empty_and_single_vertex_inputs() -> None:
    assert weld_vertices(np.zeros((0, 3), dtype=np.float64)).shape == (0,)
    assert np.array_equal(
        weld_vertices(np.zeros((1, 3), dtype=np.float64)),
        np.array([0], dtype=np.int64),
    )


def test_weld_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        weld_vertices(np.zeros((4, 2), dtype=np.float64))


# ---------------------------------------------------------------------------
# 面隣接
# ---------------------------------------------------------------------------


def _adjacency_of(mesh: MeshData) -> np.ndarray:
    return build_face_adjacency(mesh.faces, weld_vertices(mesh.vertices))


def test_cube_has_exactly_18_welded_edges(cube_mesh: MeshData) -> None:
    """合否基準8(前半): cube の weld 隣接辺数 E == 18。

    導出(conftest の Step 2-2 節にも記録): 閉じた三角形メッシュのオイラー標数
    V - E + F = 2 に、位置weld 後の V=8 と F=12 を代入して E = 18。内訳は立方体の
    稜線 12 本 + 各 facet を 2 三角形に割る対角線 6 本。
    """
    adjacency = _adjacency_of(cube_mesh)
    assert adjacency.shape == (18, 2)
    assert adjacency.dtype == np.int64
    # 内訳の直接確認: facet 内対角線 6 本 = 同一 facet の面ペア (2i, 2i+1)。
    facet_internal = [
        pair for pair in adjacency.tolist() if pair[0] // 2 == pair[1] // 2
    ]
    assert len(facet_internal) == 6
    assert len(adjacency) - len(facet_internal) == 12


def test_two_cubes_have_no_edge_between_the_components(
    two_cubes_mesh: MeshData,
) -> None:
    """合否基準8(後半): 立方体間の辺が 0 本(全 36 本が各立方体に閉じる)。"""
    adjacency = _adjacency_of(two_cubes_mesh)
    assert adjacency.shape == (36, 2)
    # 面 0-11 が 1 個目、面 12-23 が 2 個目。
    crossing = [pair for pair in adjacency.tolist() if (pair[0] < 12) != (pair[1] < 12)]
    assert crossing == []


def test_capped_cylinder_is_closed_and_manifold(
    capped_cylinder_mesh: MeshData,
) -> None:
    """円筒は閉多様体なので辺数 = 3M/2 = 192(全辺がちょうど 2 面共有)。"""
    adjacency = _adjacency_of(capped_cylinder_mesh)
    assert len(capped_cylinder_mesh.faces) == 128
    assert adjacency.shape == (192, 2)


def test_adjacency_rows_are_ordered_and_deterministic(cube_mesh: MeshData) -> None:
    adjacency = _adjacency_of(cube_mesh)
    assert np.all(adjacency[:, 0] < adjacency[:, 1])
    keys = [tuple(pair) for pair in adjacency.tolist()]
    assert keys == sorted(keys)
    assert np.array_equal(adjacency, _adjacency_of(cube_mesh))


def test_nonmanifold_edge_yields_no_adjacency(nonmanifold_mesh: MeshData) -> None:
    """3 面が共有する辺は隣接に数えない(負の対照)。他の辺は 1 面しか持たない。"""
    assert _adjacency_of(nonmanifold_mesh).shape == (0, 2)


def test_zero_area_face_still_forms_a_manifold_edge(zero_area_mesh: MeshData) -> None:
    """零面積面は *隣接段では* 切られない(切るのは二面角段の役目)。"""
    adjacency = _adjacency_of(zero_area_mesh)
    assert adjacency.tolist() == [[0, 1], [0, 2]]


def test_positionally_degenerate_face_loses_all_adjacency() -> None:
    """weld で corner が潰れる面は全隣接カット。相手側の辺も同時に落ちる。"""
    # v3 は v1 と同一座標 → 面 1 の welded corner は (0, 1, 1) に潰れる。
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    adjacency = build_face_adjacency(faces, weld_vertices(vertices))
    assert adjacency.shape == (0, 2)


def test_degenerate_face_does_not_become_adjacent_to_itself() -> None:
    """corner が潰れた面は **自分自身と隣接しない**(重複辺による自己ループの防止)。

    weld で 2 corner が同一代表になると、その面は同一の辺を 2 回持つ。位置的縮退面を
    除外しないと「ちょうど 2 面が共有する辺」の判定にその面だけで合致してしまい、
    `(i, i)` という自己隣接が生まれる。上の 2 面版は非多様体(3 本目の重複で辺の
    共有数が 3 になる)側の規則でも切れてしまうため、縮退除外そのものを固定する
    対照としてこの 1 面版が要る。
    """
    # v1 と v2 が同一座標 → welded corner は (0, 1, 1)、辺 (0,1) を 2 回持つ。
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    adjacency = build_face_adjacency(faces, weld_vertices(vertices))
    assert adjacency.shape == (0, 2)


def test_degenerate_face_removal_can_expose_a_manifold_edge() -> None:
    """縮退面の除外は共有数を 3 → **2** に落とし、残り2面を新たに隣接にもする。

    「非多様体辺は必ずカット」だけを読むと予測できない経路なので固定する
    (`build_face_adjacency` の docstring と対で読むこと)。AI 生成メッシュの
    「内部辺に貼り付いたゼロ幅スライバ三角形」がこの形。
    """
    # v4 は v1 と同一座標 → 面 2 は weld で (0, 1, 1) に潰れて除外される。
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    # 生の入力では辺 (0,1) を 3 面が共有する = 非多様体。縮退面 2 を先に落とすので
    # 残る面 0 と面 1 が多様体ペアとして成立する。
    assert build_face_adjacency(faces, weld_vertices(vertices)).tolist() == [[0, 1]]


def test_adjacency_of_face_less_mesh_is_empty() -> None:
    adjacency = build_face_adjacency(
        np.zeros((0, 3), dtype=np.int64), np.zeros(3, dtype=np.int64)
    )
    assert adjacency.shape == (0, 2)
    assert adjacency.dtype == np.int64


def test_adjacency_rejects_bad_shapes_and_out_of_range_indices() -> None:
    weld_map = np.arange(3, dtype=np.int64)
    with pytest.raises(ValueError, match=r"faces must have shape \(M, 3\)"):
        build_face_adjacency(np.zeros((2, 4), dtype=np.int64), weld_map)
    with pytest.raises(ValueError, match=r"weld_map must have shape \(N,\)"):
        build_face_adjacency(np.zeros((1, 3), dtype=np.int64), np.zeros((3, 1)))
    with pytest.raises(ValueError, match="outside weld_map"):
        build_face_adjacency(np.array([[0, 1, 9]], dtype=np.int64), weld_map)


def test_adjacency_is_not_index_based(cube_mesh: MeshData) -> None:
    """weld を通さない頂点 index 隣接では cube の辺が 0 本になること(退行防止)。

    計画v2 §2.1 の前提そのもの: `load_mesh` は weld しないので、頂点 index の
    ままでは 6 facet が完全に孤立する。weld を外すと本層が壊れることを固定する。
    """
    identity = np.arange(len(cube_mesh.vertices), dtype=np.int64)
    index_based = build_face_adjacency(cube_mesh.faces, identity)
    assert index_based.shape == (6, 2)  # facet 内対角線のみ、稜線 12 本が消える


# ---------------------------------------------------------------------------
# 面法線と二面角
# ---------------------------------------------------------------------------


def test_face_normals_are_unit_and_axis_aligned_on_cube(cube_mesh: MeshData) -> None:
    normals, zero_area = face_normals(cube_mesh.vertices, cube_mesh.faces)
    assert normals.shape == (12, 3)
    assert not zero_area.any()
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
    # 立方体の面法線は 6 方向の軸並行ベクトルのみ。
    rounded = {tuple(float(v) for v in np.round(n, 12)) for n in normals.tolist()}
    assert rounded == {
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    }


def test_face_normals_flag_zero_area_faces(zero_area_mesh: MeshData) -> None:
    normals, zero_area = face_normals(zero_area_mesh.vertices, zero_area_mesh.faces)
    assert zero_area.tolist() == [False, False, True]
    assert np.array_equal(normals[2], np.zeros(3))


def test_dihedral_angles_of_cube_are_zero_or_ninety(cube_mesh: MeshData) -> None:
    adjacency = _adjacency_of(cube_mesh)
    normals, _ = face_normals(cube_mesh.vertices, cube_mesh.faces)
    angles = dihedral_angles(normals, adjacency)
    assert angles.shape == (18,)
    assert np.allclose(np.sort(angles)[:6], 0.0, atol=1e-9)  # facet 内対角線
    assert np.allclose(np.sort(angles)[6:], 90.0, atol=1e-9)  # 稜線


def test_dihedral_angles_of_empty_adjacency() -> None:
    angles = dihedral_angles(np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int64))
    assert angles.shape == (0,)


def test_dihedral_angles_rejects_bad_adjacency_shape() -> None:
    with pytest.raises(ValueError, match=r"adjacency must have shape \(E, 2\)"):
        dihedral_angles(np.zeros((3, 3)), np.zeros((3, 3), dtype=np.int64))


def test_dihedral_angle_is_180_when_winding_is_inconsistent() -> None:
    """既知の限界(docstring 記載)を実測で固定する: 巻き順反転で同一平面が 180 度。"""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    aligned = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    flipped = np.array([[0, 1, 2], [0, 3, 2]], dtype=np.int64)  # 面 1 を反転
    for faces, expected in ((aligned, 0.0), (flipped, 180.0)):
        adjacency = build_face_adjacency(faces, weld_vertices(vertices))
        normals, _ = face_normals(vertices, faces)
        assert dihedral_angles(normals, adjacency)[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# smooth_edge_mask(二面角しきい値 + 零面積カットの合成 — fusion.py と共有する部品)
# ---------------------------------------------------------------------------


def test_smooth_edge_mask_cuts_cube_creases_but_keeps_facet_diagonals(
    cube_mesh: MeshData,
) -> None:
    adjacency = _adjacency_of(cube_mesh)
    mask = smooth_edge_mask(
        cube_mesh.vertices, cube_mesh.faces, adjacency, angle_deg=60.0
    )
    assert mask.dtype == bool
    assert mask.shape == (18,)
    assert int(mask.sum()) == 6  # facet 内対角線(0 度)だけが残る
    # 境界は閉区間: 90 度ちょうどの稜線は angle_deg=90 で滑らか扱いになる。
    wide = smooth_edge_mask(
        cube_mesh.vertices, cube_mesh.faces, adjacency, angle_deg=90.0
    )
    assert int(wide.sum()) == 18


def test_smooth_edge_mask_cuts_zero_area_faces_even_at_180_degrees(
    zero_area_mesh: MeshData,
) -> None:
    """零面積カットが角度しきい値と **組で** 効くこと(この関数の存在理由)。

    零面積面の法線は零ベクトルなので内積 0 = 90 度と評価される。角度判定だけを
    書くと `angle_deg >= 90` で零面積面が滑らかな辺として繋がってしまう。
    """
    adjacency = _adjacency_of(zero_area_mesh)
    assert adjacency.tolist() == [[0, 1], [0, 2]]  # 面 2 が零面積
    mask = smooth_edge_mask(
        zero_area_mesh.vertices, zero_area_mesh.faces, adjacency, angle_deg=180.0
    )
    assert mask.tolist() == [True, False]


@pytest.mark.parametrize("angle_deg", [float("nan"), float("inf"), 0.0, -1.0, 181.0])
def test_smooth_edge_mask_rejects_out_of_range_angles(
    cube_mesh: MeshData, angle_deg: float
) -> None:
    """入口で `ValueError`(`DihedralSegmenter.__init__` と同じ範囲・同じ例外)。

    **`nan` を素通しさせない WHY**: `nan <= angle_deg` は常に False なので、検証が
    無いと「全辺カット = 全面が別部位」が無言で起きる。この関数は公開関数で
    Step 2-4 の `fusion.py` が直接呼ぶ予定なので、`DihedralSegmenter` を経由しない
    経路にも同じ番人が要る。
    """
    with pytest.raises(ValueError, match="angle_deg"):
        smooth_edge_mask(
            cube_mesh.vertices,
            cube_mesh.faces,
            _adjacency_of(cube_mesh),
            angle_deg=angle_deg,
        )


def test_smooth_edge_mask_handles_empty_adjacency_and_bad_shape() -> None:
    mask = smooth_edge_mask(
        np.zeros((1, 3)),
        np.zeros((0, 3), dtype=np.int64),
        np.zeros((0, 2), dtype=np.int64),
        angle_deg=60.0,
    )
    assert mask.shape == (0,)
    assert mask.dtype == bool
    with pytest.raises(ValueError, match=r"adjacency must have shape \(E, 2\)"):
        smooth_edge_mask(
            np.zeros((1, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros(2), angle_deg=1
        )


# ---------------------------------------------------------------------------
# fixture の UV 健全性
#
# Step 2-6/2-7 の焼き直しオラクル(内部点 PSNR)がこの fixture を使うため、UV 領域が
# 辺を共有していると `u=0.5` / `v=0.5` の直線上でバイリニア滲みが起き、原因の
# 切り分けが難しい形で PSNR が落ちる。既存 `_build_cube_geometry`(conftest:34)が
# 10% インセットを入れているのと同じ理由なので、ここで性質として固定する。
# ---------------------------------------------------------------------------

# 領域どうしの最小すきま。外枠 0.5 幅に 10% インセットを両側入れると 0.1 空く。
_MIN_UV_PATCH_GAP = 0.05


def _uv_bounding_box(uv: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(uv[:, 0].min()),
        float(uv[:, 0].max()),
        float(uv[:, 1].min()),
        float(uv[:, 1].max()),
    )


def _boxes_are_disjoint(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    gap: float,
) -> bool:
    """2 つの軸並行矩形が `gap` 以上離れているか(u 方向か v 方向のどちらかで)。"""
    u0a, u1a, v0a, v1a = first
    u0b, u1b, v0b, v1b = second
    return u0b - u1a >= gap or u0a - u1b >= gap or v0b - v1a >= gap or v0a - v1b >= gap


def _cylinder_face_classes(mesh: MeshData) -> np.ndarray:
    """面法線と z 軸の内積で 0=+Z キャップ / 1=-Z キャップ / 2=側面 に分ける。"""
    corners = mesh.vertices[mesh.faces]
    raw = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    axis_dot = (raw / np.linalg.norm(raw, axis=1, keepdims=True))[:, 2]
    return np.where(axis_dot >= 0.5, 0, np.where(axis_dot <= -0.5, 1, 2))


def test_capped_cylinder_uv_is_finite_and_inside_the_unit_square(
    capped_cylinder_mesh: MeshData,
) -> None:
    uv = capped_cylinder_mesh.uv
    assert uv is not None
    assert uv.shape == (384, 2)
    assert np.all(np.isfinite(uv))
    assert float(uv.min()) >= 0.0
    assert float(uv.max()) <= 1.0


def test_capped_cylinder_uv_regions_do_not_touch(
    capped_cylinder_mesh: MeshData,
) -> None:
    """側面 / +Z キャップ / -Z キャップ の UV 領域が互いに離れていること。"""
    uv = capped_cylinder_mesh.uv
    assert uv is not None
    classes = _cylinder_face_classes(capped_cylinder_mesh)
    corner_uv = uv[capped_cylinder_mesh.faces]  # (M, 3, 2)
    boxes = [
        _uv_bounding_box(corner_uv[classes == part].reshape(-1, 2)) for part in range(3)
    ]
    for first in range(3):
        for second in range(first + 1, 3):
            assert _boxes_are_disjoint(
                boxes[first], boxes[second], _MIN_UV_PATCH_GAP
            ), (
                f"UV regions {first} and {second} are closer than "
                f"{_MIN_UV_PATCH_GAP}: {boxes[first]} vs {boxes[second]}"
            )


def test_capped_cylinder_uv_is_inset_from_the_atlas_border(
    capped_cylinder_mesh: MeshData,
) -> None:
    """外枠にも 10% インセットが効いていること(アトラス端の滲み対策)。"""
    uv = capped_cylinder_mesh.uv
    assert uv is not None
    assert float(uv.min()) >= _MIN_UV_PATCH_GAP
    assert float(uv.max()) <= 1.0 - _MIN_UV_PATCH_GAP


def test_two_cubes_uv_halves_do_not_touch(two_cubes_mesh: MeshData) -> None:
    """2 個の立方体の UV が u=0.5 を挟んで重ならないこと。

    `_build_cube_geometry` の 10% インセットを u 方向に半分へ縮めた結果、
    すきまは 0.5 * (0.1/3) * 2 に縮む。ゼロでないことを固定する。
    """
    uv = two_cubes_mesh.uv
    assert uv is not None
    assert np.all(np.isfinite(uv))
    left_max = float(uv[:24, 0].max())
    right_min = float(uv[24:, 0].min())
    assert left_max < 0.5 < right_min
    assert right_min - left_max > 0.0
