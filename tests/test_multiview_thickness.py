"""`segmentation/multiview/thickness.py` のゲート(計画v4 §5 Step 2-5・裁定C)。

厚み(Shape Diameter)は決定的な幾何量なので、GPU も ML も無しで厳密に
検査できる。中核は **BVH レイキャストを総当たり Möller–Trumbore のオラクルと
突き合わせる**こと(probe 2026-08-01 で実証済みの総当たり実装をテスト側へ
独立に写す — production の BVH を import して自分と比べる循環にしない)。
性能は壁時計の絶対値ではなく**スケーリング比**で見る
(`tests/test_segmentation_labels.py` の前例に倣う)。
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
import pytest

from atlasmith.segmentation.adjacency import face_normals
from atlasmith.segmentation.multiview.thickness import (
    compute_face_thickness01,
    raycast_first_hit,
    thickness_to_image,
)
from atlasmith.types import MeshData

# ---------------------------------------------------------------------------
# 総当たりオラクル(probe_samesh_parity.py の実証済み実装をテスト側へ独立に写す)
#
# 数値許容の**定数は production と同じ値を写す**(相対 eps の基準まで変えると
# 「同じ問題を解いているか」が崩れる)。**アルゴリズムは独立**: BVH も枝刈りも
# 使わず、レイ × 全三角形を chunk ごとに全対 broadcast する O(R*M)。
# ---------------------------------------------------------------------------

_ORACLE_ORIGIN_OFFSET_RATIO = 1e-4
_ORACLE_MIN_HIT_RATIO = 1e-6
_ORACLE_DET_EPS_RATIO = 1e-12
# (chunk, M, 3) float64 の中間配列が数本立つ。256 x 6016 x 3 x 8B ≒ 37MB/本。
_ORACLE_RAY_CHUNK = 256


def _oracle_first_hit(
    vertices: np.ndarray,
    faces: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """総当たり Möller–Trumbore(両面ヒット)の最小 t。ヒット無しは inf。"""
    verts = np.asarray(vertices, dtype=np.float64)
    triangles = verts[np.asarray(faces, dtype=np.int64)]
    tri_v0 = triangles[:, 0, :]
    tri_e1 = triangles[:, 1, :] - tri_v0
    tri_e2 = triangles[:, 2, :] - tri_v0
    diagonal = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    min_hit = _ORACLE_MIN_HIT_RATIO * diagonal
    det_eps = _ORACLE_DET_EPS_RATIO * diagonal * diagonal

    origin_array = np.asarray(origins, dtype=np.float64)
    direction_array = np.asarray(directions, dtype=np.float64)
    n_rays = origin_array.shape[0]
    result = np.empty(n_rays, dtype=np.float64)
    for start in range(0, n_rays, _ORACLE_RAY_CHUNK):
        stop = min(start + _ORACLE_RAY_CHUNK, n_rays)
        rays_o = origin_array[start:stop]
        rays_d = direction_array[start:stop]
        pvec = np.cross(rays_d[:, None, :], tri_e2[None, :, :])
        det = np.einsum("tk,rtk->rt", tri_e1, pvec)
        parallel = np.abs(det) < det_eps
        inv_det = 1.0 / np.where(parallel, 1.0, det)
        tvec = rays_o[:, None, :] - tri_v0[None, :, :]
        u = np.einsum("rtk,rtk->rt", tvec, pvec) * inv_det
        qvec = np.cross(tvec, tri_e1[None, :, :])
        v = np.einsum("rk,rtk->rt", rays_d, qvec) * inv_det
        t = np.einsum("tk,rtk->rt", tri_e2, qvec) * inv_det
        valid = ~parallel & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > min_hit)
        result[start:stop] = np.where(valid, t, np.inf).min(axis=1)
    return result


def _inward_rays(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    """`compute_face_thickness01` と同じ規約の内向きレイ(有効面のみ)を組む。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals, zero_area = face_normals(vertices, faces)
    valid = ~zero_area
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    centroids = vertices[faces].mean(axis=1)
    origins = centroids[valid] - normals[valid] * (
        _ORACLE_ORIGIN_OFFSET_RATIO * diagonal
    )
    return origins, -normals[valid]


# ---------------------------------------------------------------------------
# BVH vs 総当たりオラクル(裁定C の主ゲート)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name", ["cube_mesh", "capped_cylinder_mesh", "peanut_mesh"]
)
def test_bvh_matches_the_brute_force_oracle_on_inward_rays(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """BVH レイキャストが総当たりオラクルと一致する(厚みレイそのもので)。

    どう壊れたら落ちるか: BVH の分割・AABB 判定・枝刈りのどれかが交差を
    取りこぼした(または余計に拾った)瞬間、当該レイの最小 t が総当たりと
    食い違って落ちる。
    """
    mesh: MeshData = request.getfixturevalue(fixture_name)
    origins, directions = _inward_rays(mesh)
    got = raycast_first_hit(mesh.vertices, mesh.faces, origins, directions)
    want = _oracle_first_hit(mesh.vertices, mesh.faces, origins, directions)
    assert got.shape == want.shape
    # 非空の担保: ヒットが 1 本も無い fixture ではこのゲートは空虚。
    assert np.isfinite(want).any()
    assert np.array_equal(np.isinf(got), np.isinf(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=1e-9, atol=1e-12)


def test_bvh_matches_the_oracle_on_outward_rays_too(cube_mesh: MeshData) -> None:
    """外向きレイ(自面ヒットが最近傍になる系)でも一致する。

    内向きレイだけだと「反対側の壁」ばかりで、始点至近のヒットを拾う経路
    (`t > min_hit` の下限まわり)が検査されない。
    """
    origins, directions = _inward_rays(cube_mesh)
    got = raycast_first_hit(cube_mesh.vertices, cube_mesh.faces, origins, -directions)
    want = _oracle_first_hit(cube_mesh.vertices, cube_mesh.faces, origins, -directions)
    assert np.isfinite(want).any()
    assert np.array_equal(np.isinf(got), np.isinf(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=1e-9, atol=1e-12)


def test_raycast_is_deterministic(peanut_mesh: MeshData) -> None:
    """同一入力 2 回でビット同一(RNG 不使用・安定ソートの実証)。"""
    origins, directions = _inward_rays(peanut_mesh)
    first = raycast_first_hit(
        peanut_mesh.vertices, peanut_mesh.faces, origins, directions
    )
    second = raycast_first_hit(
        peanut_mesh.vertices, peanut_mesh.faces, origins, directions
    )
    assert np.array_equal(first, second)


def test_raycast_rejects_bad_shapes(cube_mesh: MeshData) -> None:
    """入口検証: 契約外の shape は `ValueError`。"""
    good_origins = np.zeros((2, 3))
    good_dirs = np.zeros((2, 3))
    with pytest.raises(ValueError, match="vertices"):
        raycast_first_hit(np.zeros((3, 2)), cube_mesh.faces, good_origins, good_dirs)
    with pytest.raises(ValueError, match="faces"):
        raycast_first_hit(
            cube_mesh.vertices,
            np.zeros((2, 4), dtype=np.int64),
            good_origins,
            good_dirs,
        )
    with pytest.raises(ValueError, match="integer"):
        raycast_first_hit(
            cube_mesh.vertices,
            np.zeros((2, 3), dtype=np.float64),
            good_origins,
            good_dirs,
        )
    with pytest.raises(ValueError, match="origins"):
        raycast_first_hit(
            cube_mesh.vertices, cube_mesh.faces, np.zeros((2,)), good_dirs
        )
    with pytest.raises(ValueError, match="directions"):
        raycast_first_hit(
            cube_mesh.vertices, cube_mesh.faces, good_origins, np.zeros((3, 3))
        )


# ---------------------------------------------------------------------------
# 反証レビュー B-4(2026-08-03): AABB の上面をかすめる軸平行レイ
# ---------------------------------------------------------------------------

# 立方体を並べる本数・間隔・半径。**間隔 > 2*half** にして箱の間に隙間を作るのが
# 肝: 隙間から撃つと「原点を含むノードは無い / 前方のノードへ進む必要がある」
# (= slab の near > 0)状況になり、far が 0 に張り付く欠陥が顕在化する。
_GRAZE_N_BOXES = 6
_GRAZE_PITCH = 2.0
_GRAZE_HALF = 0.5


def _axis_aligned_boxes(
    n_boxes: int = _GRAZE_N_BOXES,
    pitch: float = _GRAZE_PITCH,
    half: float = _GRAZE_HALF,
) -> tuple[np.ndarray, np.ndarray]:
    """x 軸上に等間隔で並べた閉じた立方体 `n_boxes` 個を 1 メッシュにする。

    箱ごとに頂点を独立に持つ(weld しない)ので、bbox は
    `[-half, (n_boxes-1)*pitch + half] x [-half, half] x [-half, half]`。
    **全ての箱が y/z 方向で bbox の面に接している**ため、BVH のどのノードも
    `hi[1] == hi[2] == half` を持つ — これが「上面かすめ」の再現条件。
    """
    unit_vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [-1.0, +1.0, +1.0],
        ],
        dtype=np.float64,
    )
    unit_faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # -Z
            [4, 5, 6], [4, 6, 7],  # +Z
            [0, 1, 5], [0, 5, 4],  # -Y
            [1, 2, 6], [1, 6, 5],  # +X
            [2, 3, 7], [2, 7, 6],  # +Y
            [3, 0, 4], [3, 4, 7],  # -X
        ],
        dtype=np.int64,
    )  # fmt: skip
    vertices = np.concatenate(
        [
            unit_vertices * half + np.array([index * pitch, 0.0, 0.0])
            for index in range(n_boxes)
        ]
    )
    faces = np.concatenate(
        [unit_faces + index * len(unit_vertices) for index in range(n_boxes)]
    )
    return vertices, faces


def test_bvh_finds_hits_for_rays_grazing_the_aabb_top_face() -> None:
    """反証レビュー B-4(2026-08-03)の再現ケース。

    `direction[k] == 0` かつ `origin[k]` がノード AABB の `hi[k]` と **厳密一致**
    する軸平行レイ。旧実装は方向 0 の軸を `1e-300` で置換していたため
    `(hi - origin) * 1e300 = 0.0 * 1e300 = 0.0` となり、slab の `far` が 0 に
    張り付いて前方(`near > 0`)のノードが軒並み棄却された。

    どう壊れたら落ちるか: 方向 0 軸の扱いが「スラブ内かどうか」以外の判定に
    戻った瞬間、`hi` 上のレイだけ `inf` を返して総当たりオラクルと食い違う
    (実測の壊れ方: bvh=[inf ...] / oracle=[0.5 ...])。
    """
    vertices, faces = _axis_aligned_boxes()
    hi_z = float(vertices[:, 2].max())
    lo_z = float(vertices[:, 2].min())
    # 原点は箱 0 と箱 1 の隙間のちょうど中央。+x へ進むと箱 1 の -X 面に当たる。
    gap = _GRAZE_PITCH - 2.0 * _GRAZE_HALF
    origin_x = _GRAZE_HALF + gap / 2.0
    expected = gap / 2.0
    # y は bbox の両端(= 角をかすめる)と内部の両方を採る。
    ys = np.array([-_GRAZE_HALF, -0.2, 0.0, 0.2, _GRAZE_HALF])
    for label, z in (
        ("z = bbox hi (the reported break)", hi_z),
        ("z = bbox hi - 1e-12", hi_z - 1e-12),
        ("z = bbox lo", lo_z),
        ("z = interior", 0.0),
    ):
        origins = np.stack(
            [np.full(ys.shape, origin_x), ys, np.full(ys.shape, z)], axis=1
        )
        directions = np.zeros_like(origins)
        directions[:, 0] = 1.0
        want = _oracle_first_hit(vertices, faces, origins, directions)
        # 空虚防止: オラクル自身が全レイでヒットを見つけていること。
        np.testing.assert_allclose(want, expected, rtol=0.0, atol=1e-12)
        got = raycast_first_hit(vertices, faces, origins, directions)
        np.testing.assert_allclose(
            got, want, rtol=1e-12, atol=0.0, err_msg=f"BVH missed hits at {label}"
        )


def test_raycast_with_no_faces_returns_inf() -> None:
    """`M == 0` は「何にも当たらない」= 全 inf(例外ではない)。"""
    result = raycast_first_hit(
        np.zeros((0, 3)),
        np.zeros((0, 3), dtype=np.int64),
        np.zeros((2, 3)),
        np.ones((2, 3)),
    )
    assert result.shape == (2,)
    assert np.isinf(result).all()


# ---------------------------------------------------------------------------
# compute_face_thickness01(正規化・充填・異常系)
# ---------------------------------------------------------------------------


def test_thickness_is_deterministic_and_in_unit_interval(
    peanut_mesh: MeshData,
) -> None:
    first = compute_face_thickness01(peanut_mesh.vertices, peanut_mesh.faces)
    second = compute_face_thickness01(peanut_mesh.vertices, peanut_mesh.faces)
    assert first.shape == (len(peanut_mesh.faces),)
    assert first.dtype == np.float64
    assert np.isfinite(first).all()
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
    assert np.array_equal(first, second)


def test_peanut_waist_is_thinner_than_the_lobes(peanut_mesh: MeshData) -> None:
    """くびれ近傍(|z| 小)の厚みが膨らみ極付近より薄い(緩い性質検査)。

    peanut は `r(theta) = 1 + 0.35 cos(2 theta)` の回転面: くびれの差し渡しは
    約 1.3、極を貫く軸方向は約 2.7。正規化後もこの大小関係は保たれるはず。
    この性質こそが SAM2 の SDF チャンネルがローブ境界を見つける根拠
    (probe 実測: SDF 単独で accuracy 0.9574)。
    """
    thickness01 = compute_face_thickness01(peanut_mesh.vertices, peanut_mesh.faces)
    centroids = np.asarray(peanut_mesh.vertices)[np.asarray(peanut_mesh.faces)].mean(
        axis=1
    )
    waist = np.abs(centroids[:, 2]) < 0.2
    polar = np.abs(centroids[:, 2]) > 1.0
    assert waist.any() and polar.any()
    assert float(thickness01[waist].mean()) < float(thickness01[polar].mean())


def test_uniform_thickness_collapses_to_half(cube_mesh: MeshData) -> None:
    """立方体は全面の厚みが同一(= 辺長)なので、退化した値域は一様 0.5 になる。

    **一様 0.5 = SDF 画像が真っ平らなグレー = SAM2 への信号ゼロ** なので、
    N-1(2026-08-03 反証レビュー)以降は必ず `UserWarning` を伴う。
    """
    with pytest.warns(UserWarning, match="degenerate"):
        thickness01 = compute_face_thickness01(cube_mesh.vertices, cube_mesh.faces)
    assert np.array_equal(thickness01, np.full(len(cube_mesh.faces), 0.5))


def test_degenerate_thickness_warning_names_the_lost_signal(
    cube_mesh: MeshData,
) -> None:
    """N-1: 退化警告が「SAM2 への信号が消えた」と行動可能な代替を告げる。

    どう壊れたら落ちるか: 一様 0.5 への崩壊を `LOG.info` に戻した(= 呼び出し側が
    気付けない)瞬間、または文面から代替経路が消えた瞬間に落ちる。
    """
    with pytest.warns(UserWarning) as captured:
        compute_face_thickness01(cube_mesh.vertices, cube_mesh.faces)
    message = str(captured[0].message)
    assert "degenerate" in message
    assert "no signal for SAM2" in message
    assert "--segmenter geometric" in message


def test_zero_area_face_is_filled_with_the_median(cube_mesh: MeshData) -> None:
    """零面積面はレイを張らず中央値で埋める(画像に穴を作らない)。"""
    faces = np.vstack([np.asarray(cube_mesh.faces), [[0, 0, 1]]]).astype(np.int64)
    with pytest.warns(UserWarning, match="degenerate"):
        thickness01 = compute_face_thickness01(cube_mesh.vertices, faces)
    assert thickness01.shape == (len(faces),)
    # 立方体の有効面はすべて同厚 → 中央値も同値 → 全体が一様 0.5 のまま。
    assert np.array_equal(thickness01, np.full(len(faces), 0.5))


def test_open_surface_with_no_hits_raises(cube_mesh: MeshData) -> None:
    """1 枚きりの三角形は**どちら向きの**レイも当たらない → `RuntimeError`。

    B-2(2026-08-03)で逆方向リトライが入ったので、ここは「両方向とも全滅」の
    経路になる。メッセージは**開いた曲面と巻き順不整合の両方**を挙げること
    (片方だけ断定すると、もう一方の利用者に嘘の診断を出す)。
    """
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(RuntimeError, match="watertight") as excinfo:
        compute_face_thickness01(vertices, faces)
    message = str(excinfo.value)
    assert "either direction" in message
    assert "winding" in message


def test_reversed_winding_closed_surface_is_rescued_by_the_retry(
    peanut_mesh: MeshData, caplog: pytest.LogCaptureFixture
) -> None:
    """★ 反証レビュー B-2(2026-08-03)の再現: 巻き順を反転した閉曲面。

    法線が一貫して内向きのメッシュ(AI 生成・インポート由来で頻出 = 本ツールの
    主対象入力)では `-normal` のレイが全て逃げる。旧実装はここで
    「not watertight」と **嘘の診断つきで** `RuntimeError` を投げた
    (対照: 幾何バックエンドは巻き順反転でも無傷)。

    検査するのは 3 点: (a) 例外が出ない (b) 厚みが正方向と一致する
    (c) 巻き順が反転している旨の警告ログが出る。
    """
    forward = compute_face_thickness01(peanut_mesh.vertices, peanut_mesh.faces)
    # 空虚防止: 正方向の厚みが本当に情報を持っている(一様 0.5 ではない)。
    assert float(forward.max()) > float(forward.min())
    reversed_faces = np.asarray(peanut_mesh.faces)[:, ::-1].copy()

    logger_name = "atlasmith.segmentation.multiview.thickness"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        reversed_thickness = compute_face_thickness01(
            peanut_mesh.vertices, reversed_faces
        )

    # (b) 面の並びは変わらない(各面の corner 順だけ反転)ので要素ごとに比べられる。
    np.testing.assert_allclose(reversed_thickness, forward, rtol=1e-9, atol=1e-9)
    # (c) 黙って向きを変えない。
    warnings_seen = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name == logger_name
    ]
    assert any("winding" in message for message in warnings_seen), warnings_seen
    assert any("+normal" in message for message in warnings_seen), warnings_seen


def test_all_zero_area_faces_raise(cube_mesh: MeshData) -> None:
    """全面が零面積ならレイを 1 本も張れない → `RuntimeError`。"""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    faces = np.array([[0, 0, 1]], dtype=np.int64)
    with pytest.raises(RuntimeError, match="zero-area"):
        compute_face_thickness01(vertices, faces)


def test_thickness_of_an_empty_mesh_is_empty() -> None:
    result = compute_face_thickness01(
        np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    )
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_thickness_does_not_mutate_its_inputs(cube_mesh: MeshData) -> None:
    """引数非破壊(coding-style の全体契約)。"""
    vertices = np.asarray(cube_mesh.vertices).copy()
    faces = np.asarray(cube_mesh.faces).copy()
    with pytest.warns(UserWarning, match="degenerate"):  # 立方体は一様厚み(N-1)
        compute_face_thickness01(vertices, faces)
    assert np.array_equal(vertices, np.asarray(cube_mesh.vertices))
    assert np.array_equal(faces, np.asarray(cube_mesh.faces))


# ---------------------------------------------------------------------------
# thickness_to_image(SDF 画像合成)
# ---------------------------------------------------------------------------


def test_thickness_image_maps_faces_and_background() -> None:
    """背景は 0、面は `round(t * 255)` の同値 3ch(uint8・C 連続)。"""
    face_id = np.array([[-1, 0], [1, 2]], dtype=np.int32)
    thickness01 = np.array([0.0, 0.5, 1.0])
    image = thickness_to_image(face_id, thickness01)
    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert image.flags["C_CONTIGUOUS"]
    expected = np.array([[0, 0], [128, 255]], dtype=np.uint8)
    for channel in range(3):
        assert np.array_equal(image[:, :, channel], expected)


def test_thickness_image_rejects_contract_violations() -> None:
    face_id = np.array([[0, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="face_id"):
        thickness_to_image(np.zeros((2, 2, 2), dtype=np.int32), np.array([0.5]))
    with pytest.raises(ValueError, match="integer"):
        thickness_to_image(np.zeros((2, 2)), np.array([0.5]))
    with pytest.raises(ValueError, match="thickness01"):
        thickness_to_image(face_id, np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        thickness_to_image(face_id, np.array([0.5, 1.5]))
    # メッシュ跨ぎの取り違え: face_id が厚み配列の範囲外の面を参照している。
    with pytest.raises(ValueError, match="different"):
        thickness_to_image(face_id, np.array([0.5]))


def test_thickness_image_of_pure_background_is_black() -> None:
    """全背景 + 空の厚み配列は合法(真っ黒な画像)。"""
    image = thickness_to_image(
        np.full((2, 3), -1, dtype=np.int32), np.zeros(0, dtype=np.float64)
    )
    assert image.shape == (2, 3, 3)
    assert not image.any()


# ---------------------------------------------------------------------------
# 性能: スケーリング比ゲート(壁時計の絶対値は使わない — labels の前例に倣う)
# ---------------------------------------------------------------------------

# 小/大 peanut の解像度。面数は n_phi * (2 * n_theta - 2) なので
# (24, 32) -> 1472 面、(48, 64) -> 6016 面 — 規模比 ~4.09 倍。
_SCALING_SMALL_RES = (24, 32)
_SCALING_LARGE_RES = (48, 64)
# 規模 ~4 倍での所要時間比の上限。総当たり O(M^2) はレイ数も面数も伸びるので
# ~16.7 倍、BVH(M log M 級)は ~4.5 倍になる。両者の間に置く。
_SCALING_MAX_RATIO = 10.0


def _fastest_seconds(call: Callable[[], object], repeats: int = 3) -> float:
    """最速値を採る(GC や OS のスケジューリング揺らぎを落とすため)。"""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def test_thickness_does_not_scale_like_the_brute_force(
    build_peanut_geometry: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """面数 ~4 倍で所要時間が総当たり級(~16 倍)に伸びないこと。

    **比を見る WHY**: 絶対値はマシンで変わるが、規模を k 倍にしたときの比は
    アルゴリズムの次数で決まる。厚みはレイ数 = 面数なので、総当たりに退行すると
    比が規模比の 2 乗(~16.7)に張り付く。
    """
    small_v, small_f, _ = build_peanut_geometry(*_SCALING_SMALL_RES)
    large_v, large_f, _ = build_peanut_geometry(*_SCALING_LARGE_RES)
    # 規模の前提(比の分母/分子が想定どおり伸びている)を先に固定する。
    assert len(large_f) > 3.5 * len(small_f)

    def run_small() -> np.ndarray:
        return compute_face_thickness01(small_v, small_f)

    def run_large() -> np.ndarray:
        return compute_face_thickness01(large_v, large_f)

    # 空虚でないことの担保: 両規模とも実際に意味のある厚みが出る。
    for result in (run_small(), run_large()):
        assert np.isfinite(result).all()
        assert float(result.max()) > float(result.min())

    ratio = _fastest_seconds(run_large) / _fastest_seconds(run_small)
    assert ratio <= _SCALING_MAX_RATIO, (
        f"compute_face_thickness01 scaled by {ratio:.1f}x when the face count "
        f"grew ~4x (limit {_SCALING_MAX_RATIO}) - the raycast looks brute-force "
        "again"
    )
