"""`_part_unwrap_and_pack` の合否基準(計画v2 §5 Step 2-2 / 計画v4 §5 Step 2-6)。

検証の柱は「各 UV アイランドが単一部位に収まる」こと。ラベルの出どころに依存しないよう
**合成ラベル 4 ケース**(cube 手動2分割 / two_cubes / capped_cylinder / torus 半分割)を
主系統にし、`DihedralSegmenter` 由来ラベルで 1 系統だけ実地の裏取りをする。

アイランド分解は `tests/conftest.py` の `uv_island_labels`(production とは別実装の
独立オラクル)を使う。production の `_face_islands` を呼ぶと「実装が自分自身に一致する」
ことしか言えないため。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import xatlas

from atlasmith.pack import _naive_unwrap_and_pack, _part_unwrap_and_pack
from atlasmith.pack.part_pack import (
    _MAX_PARTS,
    AtlasDims,
    _check_atlas_dims,
    _check_face_map_bijection,
    _check_island_part_consistency,
    _check_part_face_structure,
    _check_uv_bounds,
    _validate_part_labels,
)
from atlasmith.segmentation import DihedralSegmenter
from atlasmith.segmentation.adjacency import build_face_adjacency, weld_vertices
from atlasmith.types import MeshData

_RESOLUTION = 256
_PADDING = 4
_UV_TOL = 1e-6  # UV 値域の許容(Phase 1 `tests/test_pack_naive.py:23` と同値)。
_MIN_TRI_AREA = 1e-12  # チャート三角形の下限面積(縮退チャート検出)。

# torus_split2 のアイランド数上界。根拠は
# `test_c3_island_count_stays_far_below_fragmentation` の docstring の実測表
# (実測 14 の約 4.5 倍 / 完全断片化 2048 の 1/32)。
_TORUS_MAX_ISLANDS = 64


# ---------------------------------------------------------------------------
# 合成ラベル(segmentation 非依存)
# ---------------------------------------------------------------------------


def _labels_cube_split2(mesh: MeshData) -> np.ndarray:
    """cube_mesh(12 面 = 6 facet x 2 三角形)を facet 境界で手動 2 分割する。

    `_build_cube_geometry`(`tests/conftest.py:38-45`)は facet を +X/-X/+Y/-Y/+Z/-Z の
    順に 2 三角形ずつ並べるので、前半 6 面 = {+X, -X, +Y}、後半 6 面 = {-Y, +Z, -Z}。
    どちらも facet を割らないので、部位境界は必ず立方体の稜線上に来る。
    """
    labels = np.zeros(len(mesh.faces), dtype=np.int64)
    labels[6:] = 1
    return labels


def _labels_two_cubes(mesh: MeshData) -> np.ndarray:
    """two_cubes_mesh を立方体ごとに分ける(前半 12 面 / 後半 12 面)。"""
    labels = np.zeros(len(mesh.faces), dtype=np.int64)
    labels[12:] = 1
    return labels


def _labels_capped_cylinder(mesh: MeshData) -> np.ndarray:
    """capped_cylinder_mesh を 側面(0)/+Z キャップ(1)/-Z キャップ(2) に分ける。

    fixture 自身が UV 領域を決めるのに使っているのと同じ規則(面法線と z 軸の内積)を
    テスト側で独立に組み直す。`DihedralSegmenter` を通さないので分割器に依存しない。
    """
    corners = mesh.vertices[mesh.faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    axis_dot = normals[:, 2]
    labels = np.zeros(len(mesh.faces), dtype=np.int64)
    labels[axis_dot >= 0.5] = 1
    labels[axis_dot <= -0.5] = 2
    return labels


def _labels_torus_split2(mesh: MeshData) -> np.ndarray:
    """torus_mesh を主角度 theta の符号で 2 つの半リングに分ける。"""
    centroid = mesh.vertices[mesh.faces].mean(axis=1)
    return (np.arctan2(centroid[:, 1], centroid[:, 0]) >= 0.0).astype(np.int64)


# ケース名 -> (mesh fixture 名, ラベル生成関数, 期待部位数 P)
_CASES: dict[str, tuple[str, Callable[[MeshData], np.ndarray], int]] = {
    "cube_split2": ("cube_mesh", _labels_cube_split2, 2),
    "two_cubes": ("two_cubes_mesh", _labels_two_cubes, 2),
    "capped_cylinder": ("capped_cylinder_mesh", _labels_capped_cylinder, 3),
    "torus_split2": ("torus_mesh", _labels_torus_split2, 2),
}
_CASE_NAMES = tuple(_CASES)


def _load_case(
    case: str, request: pytest.FixtureRequest
) -> tuple[MeshData, np.ndarray]:
    """ケース名から `(mesh, labels)` を組み、ラベル契約と期待 P を確かめてから返す。"""
    fixture_name, label_fn, expected_parts = _CASES[case]
    mesh = request.getfixturevalue(fixture_name)
    labels = label_fn(mesh)
    assert labels.shape == (len(mesh.faces),)
    assert labels.dtype == np.int64
    assert np.array_equal(np.unique(labels), np.arange(expected_parts, dtype=np.int64))
    return mesh, labels


# ---------------------------------------------------------------------------
# テスト側の独立道具
# ---------------------------------------------------------------------------


def _direct_xatlas_measurement(
    mesh: MeshData, labels: np.ndarray, *, resolution: int, padding_px: int
) -> tuple[int, int, int, list[np.ndarray]]:
    """テスト側で部位サブメッシュを組み直し、xatlas の生の出力を直接読む(実測)。

    production を呼ばずに `atlas.width` / `atlas.height` / `atlas.atlas_count` と
    **正規化前の生 UV** を得る。xatlas は同一幾何・同一オプションに対して決定的
    (Phase 1 `tests/test_pack_naive.py:102` の実測記録)なので、production と同じ
    アトラスが再現される。

    Returns:
        `(width, height, atlas_count, 部位ごとの生 UV のリスト)`。
    """
    atlas = xatlas.Atlas()
    for part in range(int(labels.max()) + 1):
        faces_part = mesh.faces[np.flatnonzero(labels == part)]
        vertex_ids = np.unique(faces_part)
        local_faces = np.searchsorted(vertex_ids, faces_part)
        atlas.add_mesh(
            np.ascontiguousarray(mesh.vertices[vertex_ids], dtype=np.float32),
            np.ascontiguousarray(local_faces, dtype=np.uint32),
        )
    pack_options = xatlas.PackOptions()
    pack_options.resolution = resolution
    pack_options.padding = padding_px
    atlas.generate(chart_options=xatlas.ChartOptions(), pack_options=pack_options)
    raw_uv = [
        np.asarray(atlas[part][2], dtype=np.float64)
        for part in range(int(labels.max()) + 1)
    ]
    return int(atlas.width), int(atlas.height), int(atlas.atlas_count), raw_uv


def _label_crossing_pairs(mesh: MeshData, labels: np.ndarray) -> np.ndarray:
    """weld 隣接のうち、両側の部位ラベルが異なるペア `(K, 2)` を返す。"""
    adjacency = build_face_adjacency(mesh.faces, weld_vertices(mesh.vertices))
    crossing = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    return adjacency[crossing]


def _mixed_islands(islands: np.ndarray, face_labels: np.ndarray) -> list[int]:
    """2 つ以上の部位ラベルを含むアイランドの id を返す(空なら違反 0 件)。"""
    return [
        island
        for island in range(int(islands.max()) + 1)
        if np.unique(face_labels[islands == island]).shape[0] != 1
    ]


def _inverse_face_map(face_map: np.ndarray) -> np.ndarray:
    """旧面 -> 新面 の逆写像(`face_map` が全単射であることを前提にする)。"""
    inverse = np.empty(face_map.shape[0], dtype=np.int64)
    inverse[face_map] = np.arange(face_map.shape[0], dtype=np.int64)
    return inverse


def _planar_parts_mesh() -> tuple[MeshData, np.ndarray]:
    """3 枚の平面矩形(面数 2 / 8 / 18・縦横比が別々)を 1 メッシュ 3 部位に組む。

    平面パッチは xatlas が等長に展開するので「UV 辺長 / 3D 辺長」が理論上ちょうど
    一定になり、**等方スケール規約の検証に量化域を与える**(曲面 fixture では展開の
    歪みと規約違反を区別できない)。probe (b) と同じ入力。
    """

    def grid(x0: float, width: float, height: float, nx: int, ny: int):
        xs = np.linspace(x0, x0 + width, nx + 1)
        ys = np.linspace(0.0, height, ny + 1)
        vertices = [(xs[i], ys[j], 0.0) for j in range(ny + 1) for i in range(nx + 1)]
        faces = []
        for j in range(ny):
            for i in range(nx):
                corner = j * (nx + 1) + i
                faces.append((corner, corner + 1, corner + nx + 2))
                faces.append((corner, corner + nx + 2, corner + nx + 1))
        return (
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
        )

    patches = [
        grid(0.0, 4.0, 1.0, 1, 1),
        grid(10.0, 1.0, 1.0, 2, 2),
        grid(20.0, 2.0, 3.0, 3, 3),
    ]
    all_vertices, all_faces, all_labels, offset = [], [], [], 0
    for part, (vertices, faces) in enumerate(patches):
        all_vertices.append(vertices)
        all_faces.append(faces + offset)
        all_labels.append(np.full(len(faces), part, dtype=np.int64))
        offset += len(vertices)
    mesh = MeshData(
        vertices=np.concatenate(all_vertices),
        faces=np.concatenate(all_faces),
        source_vertex=np.arange(offset, dtype=np.int64),
    )
    return mesh, np.concatenate(all_labels)


def _edge_length_ratios(mesh: MeshData) -> np.ndarray:
    """各三角形の全辺について `|UV 辺| / |3D 辺|` を返す(等方性の指標)。"""
    assert mesh.uv is not None
    corners3d = mesh.vertices[mesh.faces]
    corners_uv = mesh.uv[mesh.faces].astype(np.float64)
    ratios = []
    for k in range(3):
        edge3d = corners3d[:, (k + 1) % 3] - corners3d[:, k]
        edge_uv = corners_uv[:, (k + 1) % 3] - corners_uv[:, k]
        length3d = np.linalg.norm(edge3d, axis=1)
        length_uv = np.linalg.norm(edge_uv, axis=1)
        nonzero = length3d > 0
        ratios.append(length_uv[nonzero] / length3d[nonzero])
    return np.concatenate(ratios)


# ---------------------------------------------------------------------------
# 合否基準 1: 面対応・全単射・面数保存
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c1_face_correspondence_is_exact_and_bijective(case, request) -> None:
    """全新面・全 corner の 3D 座標が `old_faces[face_map]` と厳密一致し、全単射。

    新頂点は元頂点 float64 値の複製なので、丸め誤差の入る余地が無い = `array_equal`
    の厳密一致で assert する(Phase 1 の atol より強い基準)。
    """
    mesh, labels = _load_case(case, request)
    new_mesh, face_map, _dims = _part_unwrap_and_pack(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert len(new_mesh.faces) == len(mesh.faces)  # M_new == M_old
    assert face_map.dtype == np.int64
    assert face_map.shape == (len(mesh.faces),)
    # 全単射: 旧面 index の集合がちょうど 0..M-1 を 1 回ずつ覆う。
    assert np.array_equal(np.sort(face_map), np.arange(len(mesh.faces), dtype=np.int64))
    assert np.array_equal(
        new_mesh.vertices[new_mesh.faces], mesh.vertices[mesh.faces[face_map]]
    )


# ---------------------------------------------------------------------------
# 合否基準 2: アイランド–部位整合
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c2_every_island_lies_inside_a_single_part(
    case, request, uv_island_labels
) -> None:
    """全 UV アイランドの `labels[face_map]` が単一値 — 違反 0 件。

    **この 0 件は「xatlas が部位をまたがなかった」証拠ではない**(2026-08-05 反証
    レビュー B1)。`_part_unwrap_and_pack` は部位 p の新頂点 index を
    `[offset_p, offset_p + N_p)` という互いに素なレンジへ置いてから連結するので、
    xatlas が何を返そうとアイランドは部位をまたげない — つまり本テストは**構成上の
    性質(連結オフセット算術)を守る回帰テスト**であって、パッキング器の振る舞いを
    測るゲートではない。実際に発火させられる唯一の経路はそのオフセットの破壊で、
    識別力の実証は `test_c2_oracle_has_teeth_on_the_naive_path` が担う。
    """
    mesh, labels = _load_case(case, request)
    new_mesh, face_map, _dims = _part_unwrap_and_pack(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    islands = uv_island_labels(new_mesh.faces)
    face_labels = labels[face_map]
    mixed = _mixed_islands(islands, face_labels)
    assert mixed == []


def test_c2_island_oracle_can_report_a_violation(uv_island_labels) -> None:
    """独立オラクル側の sabotage: 上の判定手続きが「落ちられる」ことを示す。

    `test_c2_...` は全称命題なので、`uv_island_labels` か `_mixed_islands` の
    どちらかが壊れていても静かに green になり得る。手組みの違反レイアウトを同じ
    手続きに通し、**違反として報告されること**をここで固定する。

    面 0/1 は新頂点 2 を共有するので 1 アイランド、面 2 は誰とも共有しないので
    別アイランド。ラベルを `[0, 1, 1]` にすると前者だけが混在になる。
    """
    faces = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]], dtype=np.int64)
    islands = uv_island_labels(faces)
    # 独立オラクルの素性: 共有あり 2 面が同一 id、孤立面は別 id、連番は 0..K-1。
    assert islands[0] == islands[1]
    assert islands[2] != islands[0]
    assert sorted(set(islands.tolist())) == [0, 1]
    assert _mixed_islands(islands, np.array([0, 1, 1], dtype=np.int64)) == [
        int(islands[0])
    ]
    # 対照: 共有アイランドのラベルが揃っていれば違反 0 件(アイランドを跨ぐラベル差は
    # 違反ではない)。
    assert _mixed_islands(islands, np.array([0, 0, 1], dtype=np.int64)) == []


def test_c2_oracle_has_teeth_on_the_naive_path(torus_mesh, uv_island_labels) -> None:
    """識別力の実証: **部位分割を取り除くと**同じ手続きが実データで違反を報告する。

    `_naive_unwrap_and_pack`(メッシュ全体を 1 回で展開)に torus の 2 分割ラベルを
    当てると、アイランドが部位境界を無視して跨ぐ。合成データではなく production の
    もう一方の経路の実出力なので、「オラクル+判定手続きが実データで落ちられる」ことの
    証拠になる(実測 2026-08-05: 混在アイランド **8 個** / 全 15 アイランド)。
    """
    labels = _labels_torus_split2(torus_mesh)
    naive_mesh, face_map = _naive_unwrap_and_pack(
        torus_mesh, resolution=_RESOLUTION, padding_px=_PADDING
    )
    islands = uv_island_labels(naive_mesh.faces)
    mixed = _mixed_islands(islands, labels[face_map])
    assert len(mixed) > 0


# ---------------------------------------------------------------------------
# 合否基準 3: アイランド数 >= P / 各部位のアイランド数 >= 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c3_island_counts(case, request, uv_island_labels) -> None:
    """アイランド数 >= P、かつ各部位が少なくとも 1 つのアイランドを持つ。

    **「アイランド数 >= P」は非識別ゲートである**(2026-08-05 反証レビュー B3)。
    部位分割を取り除いた `_naive_unwrap_and_pack` でも 4 fixture すべてが満たす
    (実測 `6>=2 / 12>=2 / 128>=3 / 15>=2`)。契約の記述としては正しいので残すが、
    これが green でも「部位分割が効いている」証拠にはならない。**「各部位が >= 1」の
    方は安価な健全性検査**として意味がある(部位が丸ごと消えたら落ちる)。
    断片化側(上界)は `test_c3_island_count_stays_far_below_fragmentation` が見る。
    """
    mesh, labels = _load_case(case, request)
    n_parts = int(labels.max()) + 1
    new_mesh, face_map, _dims = _part_unwrap_and_pack(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    islands = uv_island_labels(new_mesh.faces)
    n_islands = int(islands.max()) + 1
    assert n_islands >= n_parts
    face_labels = labels[face_map]
    per_part = np.zeros(n_parts, dtype=np.int64)
    for island in range(n_islands):
        values = np.unique(face_labels[islands == island])
        assert values.shape[0] == 1
        per_part[values[0]] += 1
    assert (per_part >= 1).all()


def _islands_within_limit(islands: np.ndarray, limit: int) -> bool:
    """アイランド数が `limit` 以下か(断片化の上界述語)。"""
    return int(islands.max()) + 1 <= limit


def test_c3_island_count_stays_far_below_fragmentation(
    torus_mesh, uv_island_labels
) -> None:
    """★ 上界ゲート: torus のチャートが 1 三角形単位まで断片化していないこと。

    `>= P` の下界だけでは「全部の三角形が別チャート」でも green になる(実際
    capped_cylinder は 128 面 -> 128 アイランドでその状態)。チャート化が壊れたことを
    検出できる唯一の fixture が torus なのでここに置く。

    **閾値 `_TORUS_MAX_ISLANDS` の根拠(実測。発明していない)**:

    | 状況 | アイランド数 |
    |---|---|
    | 部位経路・`(res, pad)` = (128,2)/(256,4)/(512,8)/(1024,16) | **14 で不変** |
    | naive 経路(部位分割なし)`res=256, pad=4` | 15 |
    | 完全断片化(1 三角形 = 1 チャート) | 2048 (= M) |

    実測 14 に対して約 4.5 倍、完全断片化 2048 に対して 1/32 の位置に置いた。
    xatlas の chart 分割が版差でいくらか動いても誤検出せず、断片化(2 桁の増加)は
    確実に捕まえる幅である。
    """
    labels = _labels_torus_split2(torus_mesh)
    new_mesh, _face_map, _dims = _part_unwrap_and_pack(
        torus_mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    islands = uv_island_labels(new_mesh.faces)
    n_islands = int(islands.max()) + 1
    assert _islands_within_limit(islands, _TORUS_MAX_ISLANDS), (
        f"UV charts look fragmented: {n_islands} islands for "
        f"{len(torus_mesh.faces)} faces (limit {_TORUS_MAX_ISLANDS})"
    )


def test_c3_upper_bound_predicate_rejects_a_fragmented_pack(
    capped_cylinder_mesh, uv_island_labels
) -> None:
    """識別力の実証: 上界述語が**実在の断片化出力**を拒否する。

    capped_cylinder は面ごとに頂点をアンロールした fixture(384 頂点 / 128 面)なので、
    部位経路の出力は 1 三角形 = 1 チャートの**完全断片化**そのものになる(実測 128
    アイランド)。合成データではなく production の実出力を同じ述語に通し、拒否される
    ことを固定する — 上界ゲートが「常に真」でないことの証拠。
    """
    labels = _labels_capped_cylinder(capped_cylinder_mesh)
    new_mesh, _face_map, _dims = _part_unwrap_and_pack(
        capped_cylinder_mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    islands = uv_island_labels(new_mesh.faces)
    assert int(islands.max()) + 1 == len(capped_cylinder_mesh.faces)  # 完全断片化。
    assert not _islands_within_limit(islands, _TORUS_MAX_ISLANDS)


# ---------------------------------------------------------------------------
# 合否基準 4: 部位境界 = シーム(+ 量化域の非空)
# ---------------------------------------------------------------------------


def _seam_violations(
    mesh: MeshData, labels: np.ndarray, new_mesh: MeshData, face_map: np.ndarray
) -> list[tuple[int, int]]:
    """ラベル異隣接ペアのうち、対応する新面が新頂点を共有してしまうものを返す。"""
    inverse = _inverse_face_map(face_map)
    return [
        (int(old_a), int(old_b))
        for old_a, old_b in _label_crossing_pairs(mesh, labels)
        if set(new_mesh.faces[inverse[old_a]].tolist())
        & set(new_mesh.faces[inverse[old_b]].tolist())
    ]


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c4_part_boundaries_become_seams(case, request) -> None:
    """ラベルが異なる weld 隣接ペアの新面が、新頂点を 1 つも共有しない — 違反 0 件。

    **fixture ごとの識別力**(2026-08-05 反証レビュー B2。同じ述語を「部位分割を
    取り除いた」`_naive_unwrap_and_pack` の出力に当てた実測):

    | fixture | 交差ペア | naive での違反 | 判定 |
    |---|---|---|---|
    | cube_split2 | 8 | 0 | 非識別(facet 単位アンロールで境界が既存分割線に乗る) |
    | two_cubes | 0 | 0 | **空虚**(2 立方体が位置を共有しない) |
    | capped_cylinder | 64 | 0 | **非識別**(下記) |
    | torus_split2 | 64 | **63** | **唯一の識別 fixture** |

    capped_cylinder が非識別なのは、fixture が面ごとに頂点をアンロールしている
    (384 頂点 / 128 面)ため、**入力時点で全三角形が index 上孤立**しており、部位分割の
    有無に関わらず新面が頂点を共有しないから。
    つまり torus 以外の 3 つが green でも「部位分割がシームを作った」証拠にはならない。
    量化域の非空と識別力は `test_c4_torus_is_the_only_discriminating_fixture` が担う。
    fixture を足す人は、まずこの表と同じ実験(naive 経路への述語当て)をすること。
    """
    mesh, labels = _load_case(case, request)
    new_mesh, face_map, _dims = _part_unwrap_and_pack(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert _seam_violations(mesh, labels, new_mesh, face_map) == []


def test_c4_torus_is_the_only_discriminating_fixture(torus_mesh) -> None:
    """★ 量化域の非空 **かつ** 識別力: torus で述語が実データを落とせることを示す。

    2 段構え:

    1. **量化域が非空**: torus 2 分割のラベル異隣接ペアは 64 本(実測)。0 本なら
       `test_c4_part_boundaries_become_seams` は空虚に green になる。
    2. **識別力**: 部位分割を取り除いた `_naive_unwrap_and_pack` の出力に同じ述語を
       当てると **違反が出る**(実測 2026-08-05: 64 本中 **63** 本)。ここが落ちる
       fixture は 4 つのうち torus だけで、**計画が非空 assert を課した
       capped_cylinder は naive でも違反 0 = 非識別**だった(上の表)。

    期待値 64 の導出: trimesh の torus は主角度 32 分割 x 副角度 32 分割で、theta の
    符号で 2 分割すると境界は theta = 0 と theta = pi の 2 本の副円。各副円は 32 本の
    辺を持つので 32 x 2 = 64。
    """
    labels = _labels_torus_split2(torus_mesh)
    assert int(labels.max()) + 1 == 2
    crossing = _label_crossing_pairs(torus_mesh, labels)
    assert crossing.shape[0] == 64  # 量化域の非空。

    naive_mesh, naive_face_map = _naive_unwrap_and_pack(
        torus_mesh, resolution=_RESOLUTION, padding_px=_PADDING
    )
    naive_violations = _seam_violations(torus_mesh, labels, naive_mesh, naive_face_map)
    assert len(naive_violations) > 0  # 識別力: 部位分割が無ければ落ちる。


def test_c4_capped_cylinder_crossing_domain_is_nonempty(
    capped_cylinder_mesh,
) -> None:
    """capped_cylinder(P==3)のラベル異隣接ペアが 64 本あること(量化域の記録)。

    **ただしこの fixture は識別力を持たない**(2026-08-05 反証レビュー B2): 面ごとに
    頂点をアンロールした fixture(384 頂点 / 128 面)なので入力時点で全三角形が index
    上孤立しており、**部位分割を取り除いた naive 経路でも違反 0 件**になる。計画が
    非空 assert を課したのはこの fixture だが、非空であることと識別力があることは別。
    交差ペア数の記録として残し、識別力は torus のテストが担う。

    期待値 64 の導出: `sections=32` の円筒は rim を 32 分割するので、キャップ 1 枚と
    側面が共有する境界辺は 32 本。キャップは +Z / -Z の 2 枚なので 32 x 2 = 64。
    """
    labels = _labels_capped_cylinder(capped_cylinder_mesh)
    assert int(labels.max()) + 1 == 3
    crossing = _label_crossing_pairs(capped_cylinder_mesh, labels)
    assert crossing.shape[0] == 64
    # 非識別であることの実測を固定する(将来ここを識別ゲートと誤解しないため)。
    naive_mesh, naive_face_map = _naive_unwrap_and_pack(
        capped_cylinder_mesh, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert (
        _seam_violations(capped_cylinder_mesh, labels, naive_mesh, naive_face_map) == []
    )


# ---------------------------------------------------------------------------
# 合否基準 4 の実体側(N7): 異部位のチャートが UV 空間で重ならない
#
# C4 は「実装が xatlas 出力に施す変換」をピン止めしているだけで、bleed を防ぐ実体的な
# 性質(異部位が同じテクセルを取り合わない)は測っていない。ここで直接測る。
# ---------------------------------------------------------------------------


def _part_coverage_masks(
    uv: np.ndarray, faces: np.ndarray, part_of_face: np.ndarray, n_parts: int, size: int
) -> np.ndarray:
    """部位ごとのテクセル被覆 `(P, size, size) bool` を独立にラスタライズする。

    conftest の `rasterize_coverage` は「先着の face_id が勝つ」tie-break を持つため、
    **重なりが原理的に見えない**。ここは重なりの検出が目的なので、部位ごとに独立した
    マスクへ書き、あとで重ね合わせる。テクセル中心 = `((c+0.5)/size, (r+0.5)/size)`。
    """
    masks = np.zeros((n_parts, size, size), dtype=bool)
    uv_px = np.asarray(uv, dtype=np.float64) * size - 0.5
    for face_index, face in enumerate(faces):
        ax, ay = uv_px[face[0]]
        bx, by = uv_px[face[1]]
        cx, cy = uv_px[face[2]]
        area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area2) < 1e-12:
            continue  # 縮退三角形はテクセルを占めない。
        col_lo = max(0, int(np.floor(min(ax, bx, cx))))
        col_hi = min(size - 1, int(np.ceil(max(ax, bx, cx))))
        row_lo = max(0, int(np.floor(min(ay, by, cy))))
        row_hi = min(size - 1, int(np.ceil(max(ay, by, cy))))
        if col_hi < col_lo or row_hi < row_lo:
            continue
        px, py = np.meshgrid(
            np.arange(col_lo, col_hi + 1, dtype=np.float64),
            np.arange(row_lo, row_hi + 1, dtype=np.float64),
        )
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area2
        w1 = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / area2
        w2 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / area2
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        masks[part_of_face[face_index], row_lo : row_hi + 1, col_lo : col_hi + 1] |= (
            inside
        )
    return masks


def _dilate4(mask: np.ndarray) -> np.ndarray:
    """4 近傍・非循環の二値膨張(境界外は False 扱い)。"""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _cross_part_overlap_counts(masks: np.ndarray) -> tuple[int, int]:
    """`(重なるテクセル数, 4 近傍で接するテクセル数)` を返す。"""
    overlap = int((masks.sum(axis=0) > 1).sum())
    n_parts = masks.shape[0]
    touching = 0
    for part in range(n_parts):
        others = masks[np.arange(n_parts) != part].any(axis=0)
        touching += int((_dilate4(masks[part]) & others).sum())
    return overlap, touching


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_n7_parts_do_not_share_or_touch_texels(case, request) -> None:
    """異なる部位のチャートが同じテクセルを占めず、4 近傍でも接しない。

    C4(新頂点を共有しない)は index の話であって、UV 空間で重なっていないことは
    言っていない。焼き直しの bleed を実際に防ぐのはこちらなので、アトラス実寸法
    `D = max(width, height)` のテクセル格子で直接測る(実測 2026-08-05: 4 fixture とも
    重なり 0 / 4 近傍接触 0、被覆テクセルは 5.2 万〜5.3 万)。

    **この判定は `padding_px` には反応しない**(実測。当初その理由づけを書いたが誤り
    だったので訂正して残す): `padding_px=0` でも異部位の最小間隔は既に 2 テクセル
    あり、接触 0 のままである。

    | `padding_px` | 最小間隔(接触までの 4 近傍膨張回数) cube / cylinder / torus |
    |---|---|
    | 0 | 2 / 2 / 3 |
    | 1 | 3 / 4 / 6 |
    | 4 | >8 / >8 / >8 |

    このゲートが実際に捕まえるのは**部位分離そのものの喪失**である(実証: 部位を 1 つに
    畳むミューテーションで torus が `touching=452` を出して落ちる)。ガター幅そのものの
    規約は `rebake` 側(Step 2-7 の `g = min(padding_px, ...)`)の担当。
    """
    mesh, labels = _load_case(case, request)
    n_parts = int(labels.max()) + 1
    new_mesh, face_map, dims = _part_unwrap_and_pack(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert new_mesh.uv is not None
    size = max(dims.width, dims.height)
    masks = _part_coverage_masks(
        new_mesh.uv.astype(np.float64), new_mesh.faces, labels[face_map], n_parts, size
    )
    assert int(masks.any(axis=0).sum()) > 0  # 量化域の非空(被覆が空なら測っていない)。
    overlap, touching = _cross_part_overlap_counts(masks)
    assert (overlap, touching) == (0, 0)


def test_n7_overlap_detector_has_teeth() -> None:
    """識別力の実証: 手組みで重ねた UV を重なりゲートが検出する。

    部位 0 と部位 1 に**同一の**三角形 UV を与える(完全重複)。検出器が 0 を返すなら
    上のゲートは空虚。接触のみ(重なりゼロで隣り合う)ケースも別に確かめる。
    """
    triangle = np.array([[0.1, 0.1], [0.6, 0.1], [0.1, 0.6]], dtype=np.float64)
    uv = np.concatenate([triangle, triangle])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    masks = _part_coverage_masks(uv, faces, np.array([0, 1], dtype=np.int64), 2, 32)
    overlap, touching = _cross_part_overlap_counts(masks)
    assert overlap > 0
    assert touching > 0

    # 接触のみ: 2 つの三角形を隣接する列に置く(重なり 0 だがガターが無い)。
    left = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]], dtype=np.float64)
    right = np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    masks = _part_coverage_masks(
        np.concatenate([left, right]), faces, np.array([0, 1], np.int64), 2, 32
    )
    _overlap, touching = _cross_part_overlap_counts(masks)
    assert touching > 0


# ---------------------------------------------------------------------------
# 合否基準 5: アトラス寸法規約(承認事項 D = 案 (b))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c5_atlas_dims_match_direct_measurement(case, request) -> None:
    """戻り値の width/height が実測と一致し、atlas_count == 1。

    さらに UV が規約 (b) の変換 `u' = u*width/D, v' = v*height/D` そのものであることを、
    テスト側で直接読んだ**正規化前の生 UV** と突き合わせて確かめる。
    """
    mesh, labels = _load_case(case, request)
    new_mesh, _face_map, dims = _part_unwrap_and_pack(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    width, height, atlas_count, raw_uv = _direct_xatlas_measurement(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert isinstance(dims, AtlasDims)
    assert (dims.width, dims.height) == (width, height)
    assert atlas_count == 1
    largest = float(max(width, height))
    expected_uv = np.concatenate(raw_uv) * np.array(
        [width / largest, height / largest], dtype=np.float64
    )
    assert new_mesh.uv is not None
    assert new_mesh.uv.shape == expected_uv.shape
    assert np.allclose(new_mesh.uv.astype(np.float64), expected_uv, atol=1e-7)


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c5_uv_stays_inside_the_atlas_rectangle(case, request) -> None:
    """UV が `[0, w/D] x [0, h/D]` に内包される(規約 (b) の値域)。"""
    mesh, labels = _load_case(case, request)
    new_mesh, _face_map, dims = _part_unwrap_and_pack(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert new_mesh.uv is not None
    largest = float(max(dims.width, dims.height))
    limit = np.array([dims.width / largest, dims.height / largest])
    assert (new_mesh.uv.min(axis=0) >= -_UV_TOL).all()
    assert (new_mesh.uv.max(axis=0) <= limit + _UV_TOL).all()


def test_c5_isotropic_density_on_a_non_square_atlas() -> None:
    """非正方形アトラスでも、等方スケール後のテクセル密度が両軸で一致する。

    probe (b) の回帰テスト。平面パッチ 3 枚(面数 2/8/18)は等長展開されるので、
    「UV 辺長 / 3D 辺長」は規約 (b) が正しければ辺の向きに依らず一定になる。per-axis
    正規化のまま(= `1/D` を掛け忘れる)だと u と v で `width/height` 倍ずれる。
    """
    mesh, labels = _planar_parts_mesh()
    new_mesh, _face_map, dims = _part_unwrap_and_pack(
        mesh, labels, resolution=64, padding_px=2
    )
    # 量化域の非空: アトラスが正方形だと u/v のずれが原理的に生じず空虚に green。
    assert dims.width != dims.height, (
        "this fixture must produce a non-square atlas for the test to have teeth, "
        f"got {dims.width}x{dims.height}"
    )
    ratios = _edge_length_ratios(new_mesh)
    spread = float((ratios.max() - ratios.min()) / ratios.max())
    assert spread < 1e-4, f"anisotropic UV density: spread={spread:.3e}"


# ---------------------------------------------------------------------------
# 合否基準 6: 入口検証の負の対照(オーケストレーター裁定1)
# ---------------------------------------------------------------------------


def _bad_label_cases(n_faces: int) -> list[tuple[str, object]]:
    """`_part_unwrap_and_pack` が拒むべき labels の一覧。"""
    return [
        ("not_ndarray", [0] * n_faces),
        ("wrong_ndim", np.zeros((n_faces, 1), dtype=np.int64)),
        ("wrong_length", np.zeros(n_faces - 1, dtype=np.int64)),
        ("float_dtype", np.zeros(n_faces, dtype=np.float64)),
        ("int32_dtype", np.zeros(n_faces, dtype=np.int32)),
        (
            "non_consecutive",
            np.array([0] * (n_faces // 2) + [2] * (n_faces - n_faces // 2), np.int64),
        ),
        (
            "negative",
            np.array([-1] * (n_faces // 2) + [0] * (n_faces - n_faces // 2), np.int64),
        ),
    ]


@pytest.mark.parametrize("name", [case[0] for case in _bad_label_cases(12)])
def test_c6_entry_validation_rejects_bad_labels(name, cube_mesh) -> None:
    """shape 違い・float dtype・非連番・負値がすべて `ValueError`。"""
    bad = dict(_bad_label_cases(len(cube_mesh.faces)))[name]
    with pytest.raises(ValueError):
        _part_unwrap_and_pack(
            cube_mesh, bad, resolution=_RESOLUTION, padding_px=_PADDING
        )


def test_c6_entry_validation_rejects_empty_mesh() -> None:
    """`M == 0`(= `P == 0`)が `ValueError`(計画v2 §2.6 の表)。"""
    mesh = MeshData(
        vertices=np.zeros((3, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="at least one face and one part"):
        _part_unwrap_and_pack(
            mesh,
            np.zeros(0, dtype=np.int64),
            resolution=_RESOLUTION,
            padding_px=_PADDING,
        )


def test_c6_entry_validation_rejects_too_many_parts() -> None:
    """`P > _MAX_PARTS` が `ValueError` で、対処が案内される。

    xatlas に届く前に落ちることも同時に確かめる(幾何は退化したままで通す)。
    """
    n_faces = _MAX_PARTS + 1
    mesh = MeshData(
        vertices=np.zeros((3 * n_faces, 3), dtype=np.float64),
        faces=np.arange(3 * n_faces, dtype=np.int64).reshape(-1, 3),
    )
    labels = np.arange(n_faces, dtype=np.int64)
    with pytest.raises(ValueError, match="too many parts") as excinfo:
        _part_unwrap_and_pack(mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING)
    message = str(excinfo.value)
    assert "--seg-angle" in message
    assert "--seg-min-faces" in message
    assert "--granularity naive" in message


def test_c6_part_limit_boundary_is_inclusive() -> None:
    """`P == _MAX_PARTS` は通り、`P == _MAX_PARTS + 1` で落ちる(境界の位置)。"""
    assert _validate_part_labels(np.arange(_MAX_PARTS, dtype=np.int64), _MAX_PARTS) == (
        _MAX_PARTS
    )
    with pytest.raises(ValueError, match="too many parts"):
        _validate_part_labels(np.arange(_MAX_PARTS + 1, dtype=np.int64), _MAX_PARTS + 1)


# ---------------------------------------------------------------------------
# 合否基準 7: sabotage(不変条件の検査が「落ちられる」証明)
# ---------------------------------------------------------------------------


def test_c7_sabotage_island_check_rejects_mixed_labels() -> None:
    """1 アイランドに 2 部位が混在する手組みデータで検査関数が `ValueError`。

    面 0/1 は頂点 2 を共有するので 1 つのアイランドを成す。そこへ故意に別ラベルを
    与える。**この assert が緑にならなければ、production の不変条件は「常に真」を
    返すだけの飾りだったことになる。**
    """
    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    with pytest.raises(ValueError, match="spans 2 parts"):
        _check_island_part_consistency(faces, np.array([0, 1], dtype=np.int64))


def test_c7_island_check_accepts_valid_layouts() -> None:
    """識別力の対照: 同一ラベル、および頂点を共有しない別アイランドは通す。"""
    shared = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    _check_island_part_consistency(shared, np.array([1, 1], dtype=np.int64))
    # 新頂点を共有しない 2 アイランドなら、ラベルが違っても不変条件は破れない。
    disjoint = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    _check_island_part_consistency(disjoint, np.array([0, 1], dtype=np.int64))


def test_c7_island_check_rejects_label_length_mismatch() -> None:
    """検査関数自身の入口(面数と face_labels の長さ)も fail-loud。"""
    faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    with pytest.raises(ValueError, match="face_labels must have shape"):
        _check_island_part_consistency(faces, np.array([0], dtype=np.int64))


def test_c7_face_map_bijection_check_rejects_broken_maps() -> None:
    """全単射検査の sabotage。現行実装では構造上到達しないので手組みで直接叩く。

    `_build_face_map` と部位ごとの面数検査を通り抜けた後に残る最後の砦なので、
    「常に通る飾り」でないことを 3 通りの壊れ方で固定する。
    """
    with pytest.raises(ValueError, match="face count is not preserved"):
        _check_face_map_bijection(np.array([0, 1], dtype=np.int64), 3)
    with pytest.raises(ValueError, match="out-of-range old face"):
        _check_face_map_bijection(np.array([0, 1, 3], dtype=np.int64), 3)
    with pytest.raises(ValueError, match="not a bijection"):
        _check_face_map_bijection(np.array([0, 1, 1], dtype=np.int64), 3)
    # 対照: 並べ替えは正当な全単射なので通る。
    _check_face_map_bijection(np.array([2, 0, 1], dtype=np.int64), 3)


def test_c7_atlas_dims_check_rejects_bad_atlases() -> None:
    """アトラス寸法規約の sabotage(非正寸法・複数アトラス)。"""
    with pytest.raises(ValueError, match="empty atlas"):
        _check_atlas_dims(0, 16, 1)
    with pytest.raises(ValueError, match="empty atlas"):
        _check_atlas_dims(16, 0, 1)
    with pytest.raises(ValueError, match="packed into 2 atlases"):
        _check_atlas_dims(16, 16, 2)
    assert _check_atlas_dims(78, 60, 1) == AtlasDims(width=78, height=60)


def test_c7_uv_bounds_check_rejects_out_of_rectangle_uv() -> None:
    """UV 値域検査の sabotage: `[0, w/D] x [0, h/D]` を出た UV を弾く。

    `height/D = 60/78` なので v=0.9 は矩形外(u では合法な値)— 「片軸だけ見て
    いる」実装では通ってしまう位置を選んである。
    """
    dims = AtlasDims(width=78, height=60)
    _check_uv_bounds(np.array([[0.0, 0.0], [0.99, 0.76]]), dims)
    with pytest.raises(ValueError, match="leaves the atlas rectangle"):
        _check_uv_bounds(np.array([[0.0, 0.0], [0.5, 0.9]]), dims)
    with pytest.raises(ValueError, match="leaves the atlas rectangle"):
        _check_uv_bounds(np.array([[-0.01, 0.0], [0.5, 0.5]]), dims)


# ---------------------------------------------------------------------------
# N1: fail-loud のメッセージが **大域**面 index を報告する
# ---------------------------------------------------------------------------


def _two_part_mesh(faces: np.ndarray, n_vertices: int) -> MeshData:
    """任意の面配列から、部位 0 = 面 0、部位 1 = 残り、の 2 部位メッシュを作る。"""
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    vertices = np.stack(
        [np.cos(angles), np.sin(angles), np.zeros(n_vertices)], axis=1
    ).astype(np.float64)
    return MeshData(
        vertices=vertices,
        faces=faces,
        source_vertex=np.arange(n_vertices, dtype=np.int64),
    )


def test_n1_duplicate_face_reports_global_face_indices() -> None:
    """部位内の重複面が **大域**面 index で報告される(反証レビュー N1)。

    面 0 を部位 0、面 1/2 を部位 1 に置く。重複しているのは**大域面 1 と 2** だが、
    部位ローカルではこれらは 0 と 1 になる。修正前は `old faces 0 and 1` と報告して
    いて、利用者が実際の面へ辿り着けなかった。
    """
    faces = np.array([[0, 1, 2], [1, 2, 3], [3, 2, 1]], dtype=np.int64)
    mesh = _two_part_mesh(faces, 4)
    labels = np.array([0, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate face vertex-set") as excinfo:
        _part_unwrap_and_pack(mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING)
    message = str(excinfo.value)
    assert "old faces 1 and 2" in message, message
    assert "part 1" in message
    assert "global" in message


def test_n1_degenerate_face_reports_global_face_index() -> None:
    """corner の index が重複した退化面が **大域**面 index で報告される。

    退化しているのは大域面 1(部位 1 のローカル 0)。修正前は `old face 0` だった。
    """
    faces = np.array([[0, 1, 2], [0, 1, 1], [1, 2, 3]], dtype=np.int64)
    mesh = _two_part_mesh(faces, 4)
    labels = np.array([0, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="is degenerate") as excinfo:
        _part_unwrap_and_pack(mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING)
    message = str(excinfo.value)
    assert "old face 1 is degenerate" in message, message
    assert "part 1" in message
    assert "global" in message


def test_n1_structure_check_accepts_parts_that_split_a_duplicate() -> None:
    """挙動不変の対照: 重複面が**別の部位に分かれている**なら拒否しない。

    BL-7 裁定「重複面の対応は約束しないが、部位が分かれた場合に偶発的に通ることは
    あり得る」を守る。検査を大域に広げるとここが挙動変化になるので、部位内に閉じて
    いることを固定する(`_check_part_face_structure` を直接叩く)。
    """
    faces_part = np.array([[0, 1, 2]], dtype=np.int64)
    _check_part_face_structure(faces_part, np.array([1], dtype=np.int64), 0)
    _check_part_face_structure(faces_part, np.array([2], dtype=np.int64), 1)


# ---------------------------------------------------------------------------
# 合否基準 8: 異常系(零面積面)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels_list", [[0, 0, 0], [0, 0, 1]], ids=["one_part", "two_parts"]
)
def test_c8_zero_area_face_is_rejected_or_fully_preserved(
    labels_list, zero_area_mesh, uv_island_labels
) -> None:
    """零面積面入りメッシュ: **通過して M 保存 + 全不変条件 green**(実測の終端挙動)。

    §2.6 の表は「`ValueError` か M 保存のどちらか」を許すが、**実測(2026-08-05)では
    通過する側に落ちる**ので、そちらを期待値として固定する。裸の `except ValueError:
    return` は入口検疫のエラーでも緑になり反証不能だった(反証レビュー N4)ので、
    `ValueError` 側に落ちた場合は**理由を「xatlas が面数を変えた」に限定**する —
    零面積面と無関係なエラーで緑にしない。

    **黙った面欠落が起こらない**ことが本題。面数保存・全単射・アイランド–部位整合まで
    確かめる。
    """
    labels = np.array(labels_list, dtype=np.int64)
    try:
        new_mesh, face_map, dims = _part_unwrap_and_pack(
            zero_area_mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
        )
    except ValueError as error:
        # 許容されるもう一方の終端は「xatlas が零面積面を落として面数が変わった」場合
        # だけ。それ以外の ValueError はテストの前提が崩れているので通さない。
        assert "changed the face count" in str(error), f"unexpected ValueError: {error}"
        return
    assert len(new_mesh.faces) == len(zero_area_mesh.faces)
    assert np.array_equal(
        np.sort(face_map), np.arange(len(zero_area_mesh.faces), dtype=np.int64)
    )
    assert dims.width > 0 and dims.height > 0
    islands = uv_island_labels(new_mesh.faces)
    face_labels = labels[face_map]
    for island in range(int(islands.max()) + 1):
        assert np.unique(face_labels[islands == island]).shape[0] == 1


# ---------------------------------------------------------------------------
# 合否基準 9: source_vertex 合成契約 / UV 値域 / チャート面積
# ---------------------------------------------------------------------------


def _with_source_vertex(mesh: MeshData, source_vertex: np.ndarray | None) -> MeshData:
    """同一幾何で `source_vertex` だけ差し替えた `MeshData` を作る。"""
    return MeshData(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uv=mesh.uv,
        maps={},
        source_vertex=source_vertex,
    )


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c9_source_vertex_composition_through_parts(case, request) -> None:
    """裁定6 の合成契約を部位経由で: `new.sv == old.sv[origin]`。

    `origin`(新頂点 -> 元頂点)は internal だが、恒等 `source_vertex` で 1 回走らせると
    `new.sv == origin` になるので復元できる。非恒等な既知の並べ替え(反転)を据えた
    走行と突き合わせることで、実装が `source_vertex` を無視して `origin` を返す誤りを
    識別できる(Phase 1 `tests/test_pack_naive.py:113-145` と同型)。
    """
    mesh, labels = _load_case(case, request)
    n_vertices = len(mesh.vertices)
    identity_mesh, _fm1, _d1 = _part_unwrap_and_pack(
        _with_source_vertex(mesh, np.arange(n_vertices, dtype=np.int64)),
        labels,
        resolution=_RESOLUTION,
        padding_px=_PADDING,
    )
    origin = identity_mesh.source_vertex
    assert origin is not None
    permutation = np.arange(n_vertices, dtype=np.int64)[::-1].copy()
    permuted_mesh, _fm2, _d2 = _part_unwrap_and_pack(
        _with_source_vertex(mesh, permutation),
        labels,
        resolution=_RESOLUTION,
        padding_px=_PADDING,
    )
    assert np.array_equal(permuted_mesh.source_vertex, permutation[origin])
    # 識別力: 正解は `origin` 自身と異なる(恒等実装では通らない)。
    assert not np.array_equal(permutation[origin], origin)


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c9_source_vertex_none_adopts_origin(case, request) -> None:
    """`source_vertex=None` の分岐は「新頂点 -> 元頂点」自体を採用する。"""
    mesh, labels = _load_case(case, request)
    n_vertices = len(mesh.vertices)
    none_mesh, _fm1, _d1 = _part_unwrap_and_pack(
        _with_source_vertex(mesh, None),
        labels,
        resolution=_RESOLUTION,
        padding_px=_PADDING,
    )
    source_vertex = none_mesh.source_vertex
    assert source_vertex is not None
    assert source_vertex.dtype == np.int64
    assert source_vertex.shape == (len(none_mesh.vertices),)
    assert int(source_vertex.min()) >= 0
    assert int(source_vertex.max()) < n_vertices
    identity_mesh, _fm2, _d2 = _part_unwrap_and_pack(
        _with_source_vertex(mesh, np.arange(n_vertices, dtype=np.int64)),
        labels,
        resolution=_RESOLUTION,
        padding_px=_PADDING,
    )
    assert np.array_equal(source_vertex, identity_mesh.source_vertex)
    # 新頂点位置 == 元頂点位置[source_vertex]。
    assert np.array_equal(none_mesh.vertices, mesh.vertices[source_vertex])


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_c9_uv_range_and_chart_area(case, request) -> None:
    """UV は `[0,1]` に収まり(±1e-6)、各 UV 三角形の面積が正(縮退チャートなし)。"""
    mesh, labels = _load_case(case, request)
    new_mesh, _face_map, _dims = _part_unwrap_and_pack(
        mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    uv = new_mesh.uv
    assert uv is not None
    assert uv.dtype == np.float32
    assert uv.shape == (len(new_mesh.vertices), 2)
    assert float(uv.min()) >= -_UV_TOL
    assert float(uv.max()) <= 1.0 + _UV_TOL
    triangles = uv[new_mesh.faces].astype(np.float64)
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    area = 0.5 * np.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    assert (area > _MIN_TRI_AREA).all()


# ---------------------------------------------------------------------------
# `DihedralSegmenter` 由来ラベルでの 1 系統 + 引数非破壊
# ---------------------------------------------------------------------------


def test_dihedral_segmenter_labels_satisfy_all_invariants(
    capped_cylinder_mesh, uv_island_labels
) -> None:
    """実分割器の出力(P==3)でも全不変条件が成立する(合成ラベルへの依存を断つ)。

    **識別力の注記**(2026-08-05 反証レビュー B2): この fixture は面ごとに頂点を
    アンロールしているので、**下の C4 相当部分(異隣接ペアが新頂点を共有しない)は
    空虚**である — 部位分割を取り除いた naive 経路でも違反 0 件になる。ここで意味が
    あるのは「実分割器のラベルが入口検疫を通り、面対応・全単射・寸法規約が成立する」
    ことの方で、シーム化の識別は torus のテストが担う。
    """
    labels = DihedralSegmenter(angle_deg=60.0, min_faces=1).segment(
        capped_cylinder_mesh
    )
    assert int(labels.max()) + 1 == 3
    new_mesh, face_map, dims = _part_unwrap_and_pack(
        capped_cylinder_mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert len(new_mesh.faces) == len(capped_cylinder_mesh.faces)
    assert np.array_equal(
        np.sort(face_map), np.arange(len(capped_cylinder_mesh.faces), dtype=np.int64)
    )
    assert np.array_equal(
        new_mesh.vertices[new_mesh.faces],
        capped_cylinder_mesh.vertices[capped_cylinder_mesh.faces[face_map]],
    )
    islands = uv_island_labels(new_mesh.faces)
    face_labels = labels[face_map]
    for island in range(int(islands.max()) + 1):
        assert np.unique(face_labels[islands == island]).shape[0] == 1
    crossing = _label_crossing_pairs(capped_cylinder_mesh, labels)
    assert crossing.shape[0] > 0  # 量化域の非空。
    inverse = _inverse_face_map(face_map)
    for old_a, old_b in crossing:
        assert not (
            set(new_mesh.faces[inverse[old_a]].tolist())
            & set(new_mesh.faces[inverse[old_b]].tolist())
        )
    largest = float(max(dims.width, dims.height))
    limit = np.array([dims.width / largest, dims.height / largest])
    assert new_mesh.uv is not None
    assert (new_mesh.uv.max(axis=0) <= limit + _UV_TOL).all()


@pytest.mark.parametrize("case", _CASE_NAMES)
def test_inputs_are_not_mutated(case, request) -> None:
    """引数非破壊(横断規約): `mesh` の配列も `labels` も書き換えない。"""
    mesh, labels = _load_case(case, request)
    vertices_before = mesh.vertices.copy()
    faces_before = mesh.faces.copy()
    labels_before = labels.copy()
    _part_unwrap_and_pack(mesh, labels, resolution=_RESOLUTION, padding_px=_PADDING)
    assert np.array_equal(mesh.vertices, vertices_before)
    assert np.array_equal(mesh.faces, faces_before)
    assert np.array_equal(labels, labels_before)
