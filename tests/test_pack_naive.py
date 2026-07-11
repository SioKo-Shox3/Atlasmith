"""_naive_unwrap_and_pack の面対応表・UV・weld mapping の検証(test-design P1-P3)。

P1 face_map corner 整合(全面全 corner の 3D 座標一致)/ P2 UV 値域 [0,1]+チャート面積 /
P3 source_vertex 整合+面数保存。3 メッシュ(cube/sphere/torus)で parametrize する。

_naive_unwrap_and_pack は internal だが、面対応表(裁定3)と合成契約(裁定6)は
bake の整列済み入力契約の土台なので、境界を直接テストする(計画 Step 1-2)。
"""

from __future__ import annotations

import numpy as np
import pytest

from atlasmith.pack import _naive_unwrap_and_pack

MESHES = ("cube_mesh", "sphere_mesh", "torus_mesh")

_RESOLUTION = 512
_PADDING = 8
_CORNER_ATOL = 1e-5  # 全面全 corner の 3D 一致許容(確定値表)。
_UV_TOL = 1e-6  # UV 値域 [0,1] の許容(確定値表)。
_MIN_TRI_AREA = 1e-12  # チャート三角形の下限面積(縮退チャート検出)。


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_p1_face_map_corner_correspondence(mesh_fixture, request) -> None:
    """P1: 新面 i の corner k の 3D 座標が old_faces[face_map[i], k] と一致する。

    裁定3 の corner 整列契約。位置が重複する頂点(cube の隅)でも index 由来の
    対応で崩れないことを、全面・全 corner の 3D 座標一致(atol=1e-5)で確認する。
    """
    mesh = request.getfixturevalue(mesh_fixture)
    new_mesh, face_map = _naive_unwrap_and_pack(
        mesh, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert face_map.dtype == np.int64
    assert face_map.shape == (len(mesh.faces),)
    # new_mesh.vertices[new_mesh.faces[i, k]] は corner k の新頂点位置、
    # mesh.vertices[mesh.faces[face_map[i], k]] は対応する旧頂点位置。
    new_corner_pos = new_mesh.vertices[new_mesh.faces]  # (M, 3, 3)
    old_corner_pos = mesh.vertices[mesh.faces[face_map]]  # (M, 3, 3)
    assert new_corner_pos.shape == old_corner_pos.shape
    assert np.allclose(new_corner_pos, old_corner_pos, atol=_CORNER_ATOL)


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_p2_uv_range_and_chart_area(mesh_fixture, request) -> None:
    """P2: 新 UV は [0,1](±1e-6)に収まり、各 UV 三角形の面積が正(縮退なし)。"""
    mesh = request.getfixturevalue(mesh_fixture)
    new_mesh, _face_map = _naive_unwrap_and_pack(
        mesh, resolution=_RESOLUTION, padding_px=_PADDING
    )
    uv = new_mesh.uv
    assert uv is not None
    assert uv.dtype == np.float32
    assert float(uv.min()) >= -_UV_TOL
    assert float(uv.max()) <= 1.0 + _UV_TOL
    # 各 UV 三角形の符号なし面積 = 0.5 * |edge1 x edge2|。縮退チャートが無いこと。
    tri = uv[new_mesh.faces].astype(np.float64)  # (M, 3, 2)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    assert (area > _MIN_TRI_AREA).all()


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_p3_source_vertex_and_face_count(mesh_fixture, request) -> None:
    """P3: 面数保存(M_new == M_old)と source_vertex 合成契約(裁定6)。

    fixtures の source_vertex は恒等なので new.source_vertex = vmapping。よって
    new_mesh.vertices[j] は mesh.vertices[new_mesh.source_vertex[j]] と一致するはず
    (weld mapping が位置を保つ = 新頂点は元頂点の複製)。atol=1e-5 で確認する。
    """
    mesh = request.getfixturevalue(mesh_fixture)
    new_mesh, _face_map = _naive_unwrap_and_pack(
        mesh, resolution=_RESOLUTION, padding_px=_PADDING
    )
    assert len(new_mesh.faces) == len(mesh.faces)  # M_new == M_old(確定値表)。
    source_vertex = new_mesh.source_vertex
    assert source_vertex is not None
    assert source_vertex.dtype == np.int64
    assert source_vertex.shape == (len(new_mesh.vertices),)
    # source_vertex は元頂点への有効 index。
    assert int(source_vertex.min()) >= 0
    assert int(source_vertex.max()) < len(mesh.vertices)
    # weld mapping の整合: 新頂点位置 == 元頂点位置[source_vertex]。
    assert np.allclose(
        new_mesh.vertices, mesh.vertices[source_vertex], atol=_CORNER_ATOL
    )
