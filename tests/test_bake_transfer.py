"""bake_maps のテスト(test-design-phase1.md T1-T8)。

T1 恒等転写 / T2 ガター mask / T3 ガター値 / T4 端チャート回り込み /
T5 複数マップ一括 / T6-T8 負の対照(UV シフト / V 反転 / チャンネル入替)。

評価マスク・カバレッジ・形態処理は conftest の独立実装(production 非 import)を
fixture 経由で用いる(裁定9)。
"""

from __future__ import annotations

import numpy as np
import pytest

from atlasmith.bake import bake_maps
from atlasmith.metrics import masked_psnr

MESHES = ("cube_mesh", "sphere_mesh", "torus_mesh")

# 独立カバレッジ率の下限(確定値表): cube 0.50 / sphere・torus 0.30。
COVERAGE_MIN = {"cube_mesh": 0.50, "sphere_mesh": 0.30, "torus_mesh": 0.30}

_IDENTITY_ATOL = 1e-6  # 恒等転写の単一ゲート(裁定1)。
_NEG_CEILING = 25.0  # 負の対照の PSNR 天井。
_NEG_MARGIN = 15.0  # 正解 PSNR - 誤り PSNR の下限。

_DIRS8 = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _gather(arr: np.ndarray, dr: int, dc: int, fill: float | bool) -> np.ndarray:
    """近傍 arr[r+dr, c+dc] を (r, c) へ集める非循環シフト(テスト側ローカル道具)。"""
    out = np.full_like(arr, fill)
    height, width = arr.shape[:2]
    r_src0, r_src1 = max(0, dr), min(height, height + dr)
    c_src0, c_src1 = max(0, dc), min(width, width + dc)
    r_dst0, r_dst1 = max(0, -dr), min(height, height - dr)
    c_dst0, c_dst1 = max(0, -dc), min(width, width - dc)
    if r_src0 < r_src1 and c_src0 < c_src1:
        out[r_dst0:r_dst1, c_dst0:c_dst1] = arr[r_src0:r_src1, c_src0:c_src1]
    return out


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_t1_identity_transfer(
    mesh_fixture, request, make_texture, rasterize_coverage, erode8
) -> None:
    """T1: 新 UV = 旧 UV・同解像度 → 評価マスク内で max|Δ| ≤ 1e-6。"""
    mesh = request.getfixturevalue(mesh_fixture)
    size = (128, 128)
    tex = make_texture("gradient", size, 3)
    result = bake_maps(
        mesh.faces,
        mesh.uv,
        mesh.faces,
        mesh.uv,
        {"basecolor": tex},
        size=size,
        padding_px=0,
    )
    cov, _face_id, _bary = rasterize_coverage(mesh.faces, mesh.uv, size)
    assert cov.mean() >= COVERAGE_MIN[mesh_fixture]
    eval_mask = erode8(cov, 1)  # 独立カバレッジの 1px 侵食(裁定11)。
    assert eval_mask.any()
    # 評価マスクは独立カバレッジの内部。production も同テクセルを被覆しているはず。
    assert result.chart_coverage[eval_mask].all()
    baked = result.maps["basecolor"]
    diff = np.abs(baked[eval_mask] - tex[eval_mask])
    assert diff.max() <= _IDENTITY_ATOL


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_t2_gutter_mask(mesh_fixture, request, make_texture, dilate8) -> None:
    """T2: padding=0 は valid==coverage、padding=4 は valid==dilate(coverage,4)。"""
    mesh = request.getfixturevalue(mesh_fixture)
    size = (128, 128)
    tex = make_texture("gradient", size, 3)
    args = (mesh.faces, mesh.uv, mesh.faces, mesh.uv, {"basecolor": tex})
    r0 = bake_maps(*args, size=size, padding_px=0)
    assert np.array_equal(r0.valid_mask, r0.chart_coverage)
    r4 = bake_maps(*args, size=size, padding_px=4)
    expected = dilate8(r4.chart_coverage, 4)
    # 注意(検証限界): conftest の dilate8/_shift_bool は production の
    # _dilate_mask/_shift と実質同一ロジックのため、この assert は形状の内部一貫性
    # (valid_mask が chart_coverage の 4 回膨張と一致すること)の確認であり、膨張
    # アルゴリズム自体の正しさ(特に回り込み/np.roll 問題)の独立検証ではない。
    # 最高リスクの回り込み汚染は T4 が実テクスチャ出力で独立に担保している。
    assert np.array_equal(r4.valid_mask, expected)
    assert r4.valid_mask.sum() > r0.valid_mask.sum()  # 膨張が動いた証拠。


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_t3_gutter_values(mesh_fixture, request, make_texture) -> None:
    """T3: ガターテクセルの値は、被覆済み 8 近傍のいずれかの値の厳密コピー。"""
    mesh = request.getfixturevalue(mesh_fixture)
    size = (128, 128)
    tex = make_texture("gradient", size, 3)  # 近傍値が僅かに異なる → コピーを判別可能。
    result = bake_maps(
        mesh.faces,
        mesh.uv,
        mesh.faces,
        mesh.uv,
        {"basecolor": tex},
        size=size,
        padding_px=1,
    )
    gutter = result.valid_mask & ~result.chart_coverage
    assert gutter.any()
    baked = result.maps["basecolor"]
    cov = result.chart_coverage
    match = np.zeros(gutter.shape, dtype=bool)
    for dr, dc in _DIRS8:
        neigh_cov = _gather(cov, dr, dc, False)
        neigh_val = _gather(baked, dr, dc, 0.0)
        equal = np.all(np.abs(baked - neigh_val) <= 1e-6, axis=-1)
        match |= gutter & neigh_cov & equal
    # 平均補間なら「どの近傍とも一致しない」中間値になり、この assert が落ちる。
    assert np.array_equal(match & gutter, gutter)


def test_t4_left_chart_no_wraparound(cube_mesh, make_texture) -> None:
    """T4: 左端のみのチャートが右端 valid を汚染しない(np.roll 不使用の証明)。"""
    uv_left = cube_mesh.uv.copy()
    uv_left[:, 0] = uv_left[:, 0] * 0.12  # u を [0, 0.12] に押し込む。
    size = (64, 64)
    tex = make_texture("gradient", size, 3)
    result = bake_maps(
        cube_mesh.faces,
        uv_left,
        cube_mesh.faces,
        uv_left,
        {"basecolor": tex},
        size=size,
        padding_px=8,
    )
    assert result.chart_coverage.any()
    # 左端チャート(u ≤ 0.12 → c ≲ 7)+ padding 8 でも右半分は無効のまま。
    assert not result.valid_mask[:, size[1] // 2 :].any()
    assert not result.valid_mask[:, -1].any()


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_t5_multiple_maps(mesh_fixture, request, make_texture) -> None:
    """T5: 2 マップ一括焼き。対応表共用(個別焼きと bit 一致)・valid 一致・非同一。"""
    mesh = request.getfixturevalue(mesh_fixture)
    size = (128, 128)
    base = make_texture("gradient", size, 3)
    rough = make_texture("aperiodic", size, 1)  # (H, W, 1): 別チャンネル数を混在。
    common = (mesh.faces, mesh.uv, mesh.faces, mesh.uv)
    combined = bake_maps(
        *common, {"basecolor": base, "roughness": rough}, size=size, padding_px=4
    )
    only_base = bake_maps(*common, {"basecolor": base}, size=size, padding_px=4)
    only_rough = bake_maps(*common, {"roughness": rough}, size=size, padding_px=4)
    # 対応表を1度だけ計算し使い回す → 個別焼きと bit 一致。
    assert np.array_equal(combined.maps["basecolor"], only_base.maps["basecolor"])
    assert np.array_equal(combined.maps["roughness"], only_rough.maps["roughness"])
    assert np.array_equal(combined.valid_mask, only_base.valid_mask)
    assert combined.maps["basecolor"].shape[2] == 3
    assert combined.maps["roughness"].shape[2] == 1
    # 2 マップが偶然一致していないこと(内容が別物である証拠)。
    vm = combined.valid_mask
    b0 = combined.maps["basecolor"][..., 0][vm]
    r0 = combined.maps["roughness"][..., 0][vm]
    assert not np.allclose(b0, r0)


def _identity_and_wrong_psnr(
    mesh, tex_correct, tex_wrong, uv_old_wrong, size, rasterize_coverage, erode8
) -> tuple[float, float]:
    """恒等(正解)PSNR と誤対応 PSNR を、独立カバレッジ内部点で算出して返す。"""
    correct = bake_maps(
        mesh.faces,
        mesh.uv,
        mesh.faces,
        mesh.uv,
        {"m": tex_correct},
        size=size,
        padding_px=0,
    )
    wrong = bake_maps(
        mesh.faces,
        mesh.uv,
        mesh.faces,
        uv_old_wrong,
        {"m": tex_wrong},
        size=size,
        padding_px=0,
    )
    cov, _face_id, _bary = rasterize_coverage(mesh.faces, mesh.uv, size)
    eval_mask = erode8(cov, 1)
    assert eval_mask.any()
    psnr_correct = masked_psnr(correct.maps["m"], tex_correct, eval_mask)
    psnr_wrong = masked_psnr(wrong.maps["m"], tex_correct, eval_mask)
    return psnr_correct, psnr_wrong


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_t6_negative_uv_shift(
    mesh_fixture, request, make_texture, rasterize_coverage, erode8
) -> None:
    """T6 負の対照: 旧 UV を 0.1 シフトした誤対応は PSNR < 25 かつマージン ≥ 15。"""
    mesh = request.getfixturevalue(mesh_fixture)
    size = (128, 128)
    tex = make_texture("aperiodic", size, 3)  # 非周期 → シフトが偶然一致しない。
    uv_shift = mesh.uv.copy() + np.float32(0.1)
    psnr_correct, psnr_wrong = _identity_and_wrong_psnr(
        mesh, tex, tex, uv_shift, size, rasterize_coverage, erode8
    )
    assert psnr_wrong < _NEG_CEILING
    assert psnr_correct - psnr_wrong >= _NEG_MARGIN


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_t7_negative_v_flip(
    mesh_fixture, request, make_texture, rasterize_coverage, erode8
) -> None:
    """T7 負の対照: 旧 UV の V 反転は PSNR < 25 かつマージン ≥ 15。"""
    mesh = request.getfixturevalue(mesh_fixture)
    size = (128, 128)
    tex = make_texture("aperiodic", size, 3)
    uv_vflip = mesh.uv.copy()
    uv_vflip[:, 1] = 1.0 - uv_vflip[:, 1]
    psnr_correct, psnr_wrong = _identity_and_wrong_psnr(
        mesh, tex, tex, uv_vflip, size, rasterize_coverage, erode8
    )
    assert psnr_wrong < _NEG_CEILING
    assert psnr_correct - psnr_wrong >= _NEG_MARGIN


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_t8_negative_channel_swap(
    mesh_fixture, request, make_texture, rasterize_coverage, erode8
) -> None:
    """T8 負の対照: チャンネル入替した旧テクスチャは PSNR < 25 かつマージン ≥ 15。"""
    mesh = request.getfixturevalue(mesh_fixture)
    size = (128, 128)
    tex = make_texture("aperiodic", size, 3)
    tex_swapped = tex[..., [1, 0, 2]].copy()  # ch0 と ch1 を入替。
    # 新旧 UV は恒等(誤対応ではない)。誤りは「入替済みテクスチャを転写」した点のみ。
    psnr_correct, psnr_wrong = _identity_and_wrong_psnr(
        mesh, tex, tex_swapped, mesh.uv, size, rasterize_coverage, erode8
    )
    assert psnr_wrong < _NEG_CEILING
    assert psnr_correct - psnr_wrong >= _NEG_MARGIN


# ---------------------------------------------------------------------------
# 独立ラスタライザ(conftest)の巻き順・corner 順・tie-break 直接検証
#
# Step 1-1 二次レビュー(Codex)が発見した「CW 面で corner 1/2 を戻し忘れる」欠陥の
# 回帰テスト。小さな人工三角形で bary/face_id を直接検証する(bake_maps は経由しない
# — 欠陥は独立ラスタライザ側にあり、Step 1-2 オラクルが face_id+bary を旧 UV へ直接
# 適用する経路で顕在化するため)。
# ---------------------------------------------------------------------------


def test_rasterizer_winding_independent_bary(rasterize_coverage) -> None:
    """同一三角形を CCW/CW 両順でラスタライズし、位置と corner 別重心の一致を確認。

    同じ footprint を CCW 順 (A,B,C) と CW 順 (A,C,B) で走査。共通の内部テクセルで:
      - bary を各々の入力 corner へ適用した復元 UV が texel 中心に一致(位置は巻き順
        非依存)、
      - corner 別重心が入力順に追従する(A は両者一致・B は CCW pos1=CW pos2・C は
        CCW pos2=CW pos1)、
    を検証する。CW で corner 1/2 を戻し忘れる欠陥ではこれが破れる。
    """
    size = (16, 16)
    uv = np.array([[0.2, 0.2], [0.8, 0.3], [0.4, 0.8]], dtype=np.float64)  # A, B, C
    faces_ccw = np.array([[0, 1, 2]], dtype=np.int64)  # texel 空間で det>0(CCW)
    faces_cw = np.array([[0, 2, 1]], dtype=np.int64)  # 同一 footprint・det<0(CW)
    cov_ccw, _fid_ccw, bary_ccw = rasterize_coverage(faces_ccw, uv, size)
    cov_cw, _fid_cw, bary_cw = rasterize_coverage(faces_cw, uv, size)
    common = cov_ccw & cov_cw
    assert common.any()
    rr, cc = np.where(common)
    for r, c in zip(rr.tolist(), cc.tolist()):
        center = np.array([(c + 0.5) / size[1], (r + 0.5) / size[0]])
        w_ccw = bary_ccw[r, c]  # 入力 corner (A, B, C) に対応
        w_cw = bary_cw[r, c]  # 入力 corner (A, C, B) に対応
        # A の重みは巻き順不変。B は CCW pos1 = CW pos2、C は CCW pos2 = CW pos1。
        assert np.isclose(w_ccw[0], w_cw[0], atol=1e-9)
        assert np.isclose(w_ccw[1], w_cw[2], atol=1e-9)
        assert np.isclose(w_ccw[2], w_cw[1], atol=1e-9)
        # bary を各々の入力 corner へ適用 → 位置(texel 中心)に一致(巻き順非依存)。
        assert np.allclose(w_ccw @ uv[faces_ccw[0]], center, atol=1e-9)
        assert np.allclose(w_cw @ uv[faces_cw[0]], center, atol=1e-9)


def test_rasterizer_shared_edge_single_owner(rasterize_coverage) -> None:
    """共有辺(対角)を持つ 2 三角形で、共有テクセルが最小 face_id 単独所有になる。"""
    size = (24, 24)
    square = np.array(
        [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)  # 対角 0-2 を共有
    cov, face_id, _bary = rasterize_coverage(faces, square, size)
    cov0, _f0, _b0 = rasterize_coverage(faces[:1], square, size)
    cov1, _f1, _b1 = rasterize_coverage(faces[1:], square, size)
    shared = cov0 & cov1  # 対角上で両三角形が含むテクセル。
    assert shared.any()  # 共有辺テクセルが存在(検証が非自明であること)。
    # 統合ラスタでは共有テクセルは最小 face_id(=0)が単独所有(tie-break の直接検証)。
    assert (face_id[shared] == 0).all()
    # 被覆は 2 三角形の和に一致(取りこぼし・二重無し)。
    assert np.array_equal(cov, cov0 | cov1)
    # 各テクセルは単一 face_id(0 か 1)を持つ(-1 は非被覆のみ)。
    assert set(np.unique(face_id[cov]).tolist()) <= {0, 1}


def test_rasterizer_corner_correspondence_nonidentity_uv(rasterize_coverage) -> None:
    """CW 面 + 非恒等な旧 UV で、bary を旧 corner 順に適用すると既知の旧 UV に一致。

    新 UV は CW 巻き(det<0)。旧 UV は新 UV のアフィン像 O_i = T(N_i)。アフィン写像は
    重心結合を保つので、任意の内部テクセル P で bary · uv_old[face] == T(texel 中心) が
    厳密に成り立つ。CW で corner 1/2 を戻し忘れる欠陥があると O1/O2 の重みが入れ替わり、
    この一致が破れる(O1 != O2 のため検出可能)。
    """
    size = (32, 32)
    uv_new = np.array([[0.1, 0.1], [0.1, 0.6], [0.6, 0.1]], dtype=np.float64)
    face = np.array([[0, 1, 2]], dtype=np.int64)  # texel 空間で det<0(CW)

    def affine(p: np.ndarray) -> np.ndarray:
        # 非対称アフィン T(u,v) = (0.8 - 0.4v, 0.2 + 0.5u)。O_i は [0,1] に収まる。
        return np.stack([0.8 - 0.4 * p[..., 1], 0.2 + 0.5 * p[..., 0]], axis=-1)

    uv_old = affine(uv_new)
    cov, face_id, bary = rasterize_coverage(face, uv_new, size)
    assert cov.any()
    assert (face_id[cov] == 0).all()
    rr, cc = np.where(cov)
    centers = np.stack(
        [(cc + 0.5) / size[1], (rr + 0.5) / size[0]], axis=1
    )  # (K, 2) texel 中心 UV
    old_recon = bary[cov] @ uv_old[face[0]]  # w0*O0 + w1*O1 + w2*O2
    assert np.allclose(old_recon, affine(centers), atol=1e-9)
