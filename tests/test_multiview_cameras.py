"""決定的カメラ配置のゲート(計画v4 §5 Step 2-3 ゲート2・ゲート10 の一部)。

GPU 不要(numpy のみ)。ここが守るのは「同じ入力なら同じ行列」「既知点が視錐台に
収まる」「収まらない入力は黙って歪めずに落ちる」の 3 点。

**投影行列そのもののゲート(2周目レビュー B1)**: 面ID主ゲートのオラクル
(`tests/conftest.py` の `_project_to_ndc`)は production の `Camera.mvp` を消費する
ため、**投影行列の自己整合的な誤りはオラクルへ伝播して打ち消し合う**。実測:
`_perspective` / `_orthographic` の z 行を符号反転(= 深度規約の反転 = メッシュの
裏面が見える状態)させると、描かれる面が裏面 6 枚に総入れ替わっているのに主ゲートは
一致率 1.000000 で**通ってしまった**。したがって深度の向きと NDC のスケールは、
`mvp` を信用しない**手計算の期待値**で別途固定する必要がある。この節の 2 本
(`test_projection_maps_near_far_to_minus_one_plus_one` /
`test_lateral_ndc_scale_matches_hand_derivation`)がそれを担う。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlasmith.segmentation.multiview.cameras import (
    DEFAULT_N_VIEWS,
    MAX_PERSPECTIVE_FOV_DEG,
    build_camera,
    build_cameras,
    fibonacci_directions,
    validate_frustum,
)
from atlasmith.types import MeshData

_PROJECTIONS = ("perspective", "orthographic")

# 手計算の期待値と突き合わせるときに使う既知の視線方向(軸に平行でないもの)。
_PROBE_DIRECTION = np.array([0.55, 0.45, 0.70])
# 余裕係数(計画v4 §2.4.1 / 裁定1)。**テスト側で独立に持つ** — production の定数を
# import すると、係数を書き換えたときに期待値も一緒に動いてゲートが空虚になる。
_EXTENT_MARGIN = 1.1
_DEPTH_MARGIN = 1.2
_ORTHOGRAPHIC_DISTANCE_FACTOR = 2.2


def _bounding_sphere(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """AABB 中心と外接球半径(テスト側の独立実装)。"""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    centre = (lo + hi) * 0.5
    return centre, float(np.linalg.norm(vertices - centre, axis=1).max())


def _expected_distance(radius: float, projection: str, fov_deg: float) -> float:
    """計画の距離規約を**テスト側で書き下した**もの(production を呼ばない)。"""
    if projection == "perspective":
        return radius * _EXTENT_MARGIN / math.sin(math.radians(fov_deg) * 0.5)
    return radius * _ORTHOGRAPHIC_DISTANCE_FACTOR


def _cube_corners_and_center(mesh: MeshData) -> np.ndarray:
    """AABB の 8 隅と中心 `(9, 3)`(既知点ゲートの評価点)。"""
    lo = mesh.vertices.min(axis=0)
    hi = mesh.vertices.max(axis=0)
    corners = np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ],
        dtype=np.float64,
    )
    center = ((lo + hi) * 0.5)[np.newaxis, :]
    return np.concatenate([corners, center], axis=0)


def test_fibonacci_directions_are_unit_length_and_distinct() -> None:
    """24 方向が単位長(atol=1e-12)で相異なる(ゲート2)。"""
    dirs = fibonacci_directions(DEFAULT_N_VIEWS)

    assert dirs.shape == (DEFAULT_N_VIEWS, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)
    # 相異なること: 全ペアの距離が 0 でない(丸め誤差より十分大きい閾値で見る)。
    diffs = np.linalg.norm(dirs[:, None, :] - dirs[None, :, :], axis=2)
    off_diagonal = diffs[~np.eye(DEFAULT_N_VIEWS, dtype=bool)]
    assert off_diagonal.min() > 1e-6


def test_fibonacci_directions_are_bit_deterministic() -> None:
    """2 回呼びで `np.array_equal`(RNG 不使用の証明 — ゲート2)。"""
    assert np.array_equal(
        fibonacci_directions(DEFAULT_N_VIEWS), fibonacci_directions(DEFAULT_N_VIEWS)
    )


@pytest.mark.parametrize("projection", _PROJECTIONS)
def test_build_cameras_is_bit_deterministic(
    cube_mesh: MeshData, projection: str
) -> None:
    """同じメッシュから 2 回組んだカメラの行列がビット一致する(ゲート2)。"""
    first = build_cameras(cube_mesh.vertices, n_views=8, projection=projection)
    second = build_cameras(cube_mesh.vertices, n_views=8, projection=projection)

    assert len(first) == len(second) == 8
    for a, b in zip(first, second):
        assert a.index == b.index
        assert np.array_equal(a.eye, b.eye)
        assert np.array_equal(a.up, b.up)
        assert np.array_equal(a.view, b.view)
        assert np.array_equal(a.proj, b.proj)
        assert np.array_equal(a.mvp, b.mvp)


@pytest.mark.parametrize("projection", _PROJECTIONS)
def test_known_points_land_inside_the_ndc_cube(
    cube_mesh: MeshData, projection: str, project_to_ndc
) -> None:
    """bbox 中心と 8 隅が全視点で NDC `[-1, 1]^3` に入る(ゲート2)。

    「全頂点が視錐台内」という設計前提(near クリッピングを扱わない理由)を、
    カメラ側だけで独立に確かめる。
    """
    points = _cube_corners_and_center(cube_mesh)
    cameras = build_cameras(
        cube_mesh.vertices, n_views=DEFAULT_N_VIEWS, projection=projection
    )

    for camera in cameras:
        ndc = project_to_ndc(points, camera)
        assert np.all(np.abs(ndc) <= 1.0), (
            f"camera {camera.index} ({projection}) pushes a known point outside the "
            f"NDC cube: max |ndc| = {np.abs(ndc).max()}"
        )


@pytest.mark.parametrize("projection", _PROJECTIONS)
def test_all_mesh_vertices_are_inside_the_frustum(
    capped_cylinder_mesh: MeshData, projection: str
) -> None:
    """`build_cameras` が返すカメラは、元メッシュの全頂点を視錐台に収める。"""
    cameras = build_cameras(
        capped_cylinder_mesh.vertices, n_views=DEFAULT_N_VIEWS, projection=projection
    )
    for camera in cameras:
        validate_frustum(capped_cylinder_mesh.vertices, camera)


@pytest.mark.parametrize("projection", _PROJECTIONS)
def test_projection_maps_near_far_to_minus_one_plus_one(
    cube_mesh: MeshData, projection: str, project_to_ndc
) -> None:
    """**深度の向きを手計算で固定する**(2周目レビュー B1 の中核)。

    視線上の 3 点(near 平面上・中心・far 平面上)を、`mvp` を使わずに置く:
    カメラは `eye = c + u*d` にあり視線は `-u` なので、eye から距離 Z の点は
    `p = c + u*(d - Z)`。`Z = near` なら `p = c + u*1.2R`、`Z = far` なら
    `p = c - u*1.2R`、`Z = d` なら `p = c`。

    期待する `z_ndc`(投影行列の定義から手で導出したもの。`camera.proj` は見ない):
      - near 平面上 → **-1**、far 平面上 → **+1**(どちらの投影でも)
      - 中心 → 透視なら `1.2R/d`、正射影なら `0`
        (透視: `z_ndc(Z) = ((f+n)Z - 2fn) / ((f-n)Z)` に `n = d-1.2R`,
         `f = d+1.2R`, `Z = d` を入れると `2.88R^2 / (2.4Rd) = 1.2R/d`。
         正射影: `z_ndc(Z) = (2Z - (f+n)) / (f-n)` に `Z = d` を入れると `0`。)

    **どう壊れたら落ちるか**: 投影行列の z 行を符号反転する(= 手前と奥が入れ替わり、
    メッシュの内側が見える)と near が +1・far が -1 になって落ちる。この自己整合的な
    サボタージュは面ID主ゲートを通り抜ける(オラクルが同じ `mvp` を使うため)ので、
    **深度規約を検査しているのはこのテストだけ**。
    """
    fov_deg = 30.0
    centre, radius = _bounding_sphere(cube_mesh.vertices)
    unit = _PROBE_DIRECTION / np.linalg.norm(_PROBE_DIRECTION)
    distance = _expected_distance(radius, projection, fov_deg)
    camera = build_camera(
        cube_mesh.vertices, _PROBE_DIRECTION, projection=projection, fov_deg=fov_deg
    )

    offset = radius * _DEPTH_MARGIN
    points = np.array([centre + unit * offset, centre, centre - unit * offset])
    z_ndc = project_to_ndc(points, camera)[:, 2]
    expected_centre = (
        (radius * _DEPTH_MARGIN) / distance if projection == "perspective" else 0.0
    )

    assert z_ndc[0] == pytest.approx(-1.0, abs=1e-12), (
        f"the point on the near plane must map to z_ndc = -1, got {z_ndc[0]} "
        "(the depth convention is inverted: hidden faces would win the depth test)"
    )
    assert z_ndc[2] == pytest.approx(1.0, abs=1e-12), (
        f"the point on the far plane must map to z_ndc = +1, got {z_ndc[2]}"
    )
    assert z_ndc[1] == pytest.approx(expected_centre, abs=1e-12)
    # 手前が小さい(GL の depth_func='<' と組で「近い面が勝つ」を意味する)。
    assert z_ndc[0] < z_ndc[1] < z_ndc[2]


@pytest.mark.parametrize("projection", _PROJECTIONS)
def test_lateral_ndc_scale_matches_hand_derivation(
    cube_mesh: MeshData, projection: str, project_to_ndc
) -> None:
    """横方向の NDC スケールを手計算の期待値と突き合わせる(2周目レビュー B1)。

    中心から視線に**垂直**に距離 s だけずらした点は、view 空間で `(x, y)` の
    長さが s・深度が d になる。したがって NDC の半径 `hypot(x_ndc, y_ndc)` は
    (up の取り方に依らず)次の値になる:
      - 透視: `s * cos(fov/2) / (1.1R)`
        (`focal/d = (1/tan θ) * (sin θ / (1.1R)) = cos θ / (1.1R)`、θ = fov/2)
      - 正射影: `s / (1.1R)`

    **どう壊れたら落ちるか**: focal(`1/tan(fov/2)`)や half-extent(`1.1R`)を
    取り違えると落ちる。半径で見るので view 基底(side / true_up)の選び方には
    依存しない = カメラ内部実装から独立している。
    """
    fov_deg = 30.0
    centre, radius = _bounding_sphere(cube_mesh.vertices)
    unit = _PROBE_DIRECTION / np.linalg.norm(_PROBE_DIRECTION)
    camera = build_camera(
        cube_mesh.vertices, _PROBE_DIRECTION, projection=projection, fov_deg=fov_deg
    )

    # 視線に垂直な任意の方向(半径で評価するので向きは結果に効かない)。
    perpendicular = np.cross(unit, np.array([0.0, 0.0, 1.0]))
    perpendicular /= np.linalg.norm(perpendicular)
    offsets = np.array([0.25, 0.5, 1.0]) * radius
    points = centre + perpendicular[np.newaxis, :] * offsets[:, np.newaxis]
    ndc = project_to_ndc(points, camera)

    if projection == "perspective":
        scale = math.cos(math.radians(fov_deg) * 0.5) / (_EXTENT_MARGIN * radius)
    else:
        scale = 1.0 / (_EXTENT_MARGIN * radius)
    np.testing.assert_allclose(
        np.hypot(ndc[:, 0], ndc[:, 1]), offsets * scale, rtol=1e-12, atol=1e-12
    )
    # 外接球の表面(s = R)でも画面内に収まる(余裕係数 1.1 の意味)。
    assert np.hypot(ndc[-1, 0], ndc[-1, 1]) < 1.0


def test_validate_frustum_rejects_vertices_outside(cube_mesh: MeshData) -> None:
    """視錐台外の頂点があれば `ValueError`(距離の自動調整はしない — ゲート2)。

    どう壊れたら落ちるか: 「入らなければカメラを引く」という親切な実装に変えると、
    独立オラクル(near クリッピング非対応)の前提が黙って崩れるので、この
    テストが最初に落ちる。
    """
    camera = build_camera(
        cube_mesh.vertices, np.array([0.0, 0.0, 1.0]), projection="orthographic"
    )
    far_away = np.array([[0.0, 0.0, -1000.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="outside the view frustum"):
        validate_frustum(far_away, camera)


def test_orthographic_distance_follows_the_arbitration(cube_mesh: MeshData) -> None:
    """正射影のカメラ距離が `d = R * 2.2` である(2026-07-29 裁定1)。

    併せて `near = d - 1.2R = R > 0` を確認する。ここが 1.2R 以下に変わると
    near が非正になり、投影行列が壊れる。
    """
    vertices = cube_mesh.vertices
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    radius = float(np.linalg.norm(vertices - center, axis=1).max())
    camera = build_camera(
        vertices, np.array([1.0, 0.0, 0.0]), projection="orthographic"
    )

    distance = float(np.linalg.norm(camera.eye - center))
    assert distance == pytest.approx(radius * 2.2, rel=1e-12)
    assert distance - radius * 1.2 > 0.0


def test_perspective_distance_follows_the_plan(cube_mesh: MeshData) -> None:
    """透視のカメラ距離が `R / sin(fov/2) * 1.1`(計画v4 §2.4.1)。"""
    vertices = cube_mesh.vertices
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    radius = float(np.linalg.norm(vertices - center, axis=1).max())
    camera = build_camera(
        vertices, np.array([1.0, 0.0, 0.0]), projection="perspective", fov_deg=30.0
    )

    expected = radius / math.sin(math.radians(30.0) * 0.5) * 1.1
    assert float(np.linalg.norm(camera.eye - center)) == pytest.approx(
        expected, rel=1e-12
    )


def test_up_vector_switches_only_near_the_pole(cube_mesh: MeshData) -> None:
    """up は既定 `[0,0,1]`、視線がほぼ極方向のときだけ `[0,1,0]`(ゲート2)。

    この分岐は V 方向ゲート(`+Z` から見て up = `+Y`)の前提でもある。
    """
    polar = build_camera(
        cube_mesh.vertices, np.array([0.0, 0.0, 1.0]), projection="orthographic"
    )
    equatorial = build_camera(
        cube_mesh.vertices, np.array([1.0, 0.0, 0.0]), projection="orthographic"
    )

    assert np.array_equal(polar.up, np.array([0.0, 1.0, 0.0]))
    assert np.array_equal(equatorial.up, np.array([0.0, 0.0, 1.0]))


def test_entry_validation_rejects_bad_arguments(cube_mesh: MeshData) -> None:
    """`n_views` / `fov_deg` / `projection` の入口検証(ゲート10)。"""
    vertices = cube_mesh.vertices
    with pytest.raises(ValueError, match="n_views must be >= 1"):
        build_cameras(vertices, n_views=0, projection="perspective")
    with pytest.raises(ValueError, match="n_views must be an int"):
        build_cameras(vertices, n_views=2.5, projection="perspective")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"fov_deg must be finite and in \(0, 180\)"):
        build_cameras(vertices, n_views=4, projection="perspective", fov_deg=0.0)
    with pytest.raises(ValueError, match=r"fov_deg must be finite and in \(0, 180\)"):
        build_cameras(vertices, n_views=4, projection="perspective", fov_deg=180.0)
    with pytest.raises(ValueError, match="unknown projection"):
        build_cameras(vertices, n_views=4, projection="isometric")


def test_perspective_fov_upper_bound_is_the_real_one(cube_mesh: MeshData) -> None:
    """透視の画角上限が**実際に使える値**で検証される(2周目レビュー B5)。

    距離規約 `d = 1.1R/sin(fov/2)` と `near = d - 1.2R > 0` から
    `sin(fov/2) < 1.1/1.2` ⇒ `fov < 2*asin(1.1/1.2) = 132.887...` 度。
    以前は docstring も検証も `0 < fov < 180` を謳っており、132.9 度のような
    「文書どおりの入力」が `near plane must be positive`(= 内部不整合を示唆する
    メッセージ)で落ちて**原因を誤って指し示していた**。
    """
    assert MAX_PERSPECTIVE_FOV_DEG == pytest.approx(132.887071, abs=1e-5)
    vertices = cube_mesh.vertices
    # 上限の直下は通る(境界が実効的であることの非空虚性)。
    build_camera(
        vertices,
        _PROBE_DIRECTION,
        projection="perspective",
        fov_deg=MAX_PERSPECTIVE_FOV_DEG - 0.01,
    )

    for fov in (MAX_PERSPECTIVE_FOV_DEG, 133.0, 179.0):
        with pytest.raises(ValueError) as excinfo:
            build_camera(
                vertices, _PROBE_DIRECTION, projection="perspective", fov_deg=fov
            )
        message = str(excinfo.value)
        assert "too wide" in message, message
        assert "132.887" in message  # 上限を数値で示す
        assert "asin(1.1/1.2)" in message  # 導出が読み取れる
        assert "orthographic" in message  # 行動可能な代替


def test_wide_fov_is_allowed_for_orthographic(cube_mesh: MeshData) -> None:
    """正射影では広い画角が通る(`fov_deg` は使われないため — 裁定1)。

    上限(132.887 度)は透視の距離規約から出るもので、正射影の距離 `2.2R` には
    無関係。ここを一律に狭めると「使われない引数のせいで正射影が拒否される」
    という別の誤りになるので、通ることを固定しておく。
    """
    camera = build_camera(
        cube_mesh.vertices,
        _PROBE_DIRECTION,
        projection="orthographic",
        fov_deg=170.0,
    )
    centre, radius = _bounding_sphere(cube_mesh.vertices)
    assert float(np.linalg.norm(camera.eye - centre)) == pytest.approx(
        radius * _ORTHOGRAPHIC_DISTANCE_FACTOR, rel=1e-12
    )


def test_fov_is_validated_even_for_orthographic(cube_mesh: MeshData) -> None:
    """正射影でも `fov_deg` の範囲は検証する(入口検証に投影別の穴を作らない)。

    裁定1 のとおり `fov_deg` は透視専用だが、「正射影なら不正な画角が黙って通る」
    という状態は入口検証としては穴なので、値そのものは常に見る。
    """
    with pytest.raises(ValueError, match=r"fov_deg must be finite and in \(0, 180\)"):
        build_cameras(
            cube_mesh.vertices, n_views=4, projection="orthographic", fov_deg=-1.0
        )


def test_degenerate_geometry_is_rejected() -> None:
    """全頂点が同一点(半径 0)のメッシュは `ValueError`。"""
    same_point = np.zeros((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="bounding sphere radius"):
        build_cameras(same_point, n_views=4, projection="perspective")
