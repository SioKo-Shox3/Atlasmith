"""再展開後の焼き直しを一方向オラクルで検証する(test-design O1-O4)。

一方向オラクル(裁定B1/9): 基準値 = 旧テクスチャを旧 UV で独立サンプル、被験値 =
焼き上がり新テクスチャを新 UV で同じ独立サンプラでサンプル。往復で相殺される誤り
(V 反転・面取り違え・座標規約ズレ)をこの一方向比較で検出する。呼び出し側は
`mesh.faces[face_map]` で旧面を行整列してから `bake_maps` を呼ぶ(裁定5)。

ゲート(いずれも確定値表・512^2 新 atlas / 256^2 旧 multisine / padding 8):
- O1 主ゲート: 内部点・multisine で PSNR >= 40 dB。3ch 単一 map(既存)に加え、
  2D 単チャンネル(scalar)map でも同じ内部点・同じ 40dB を課す(should-fix2 —
  production の (H,W) 入力を (H,W,1) へ昇格する scalar 分岐を保護する)。
- O2 シーム帯: 有効な non-wrap シーム点で PSNR >= 20 dB。wrap 点(周期シームを跨ぐ面の
  点)は物理的に無意味なので PSNR ゲートから除外し、分類の正しさ(非空・extent 分離)
  のみ検証する(should-fix1 — 下の WHY 参照)。
- O3 負の対照: UV 0.1 シフト / V 反転 / チャンネル入替 が PSNR < 25 dB かつ
  正解 - 誤り >= 15 dB(識別力の証明)。shift/vflip は非周期、swap は直交チャンネルの
  gradient を用いる(下の WHY 参照)。
- O4 往復(補助): 旧 -> 新 -> 旧で内部点 PSNR >= 35 dB。

シーム分類と wrap 面(実装時の必須知見 — 報告参照):
テクスチャ・ラスタライザ・サンプラ・形態処理は conftest の独立実装(production 非
import・裁定9)を fixture 経由で使う。分類は「内部 vs シーム帯」の二分。内部点は、
新 atlas と旧 atlas の両カバレッジ境界から seam_margin(=2 テクセル、8 近傍 2 回
erosion)以上離れ、**かつ旧 UV 側の周期シームを跨がない**面の点。cube/sphere/torus
の球面・トーラス fixture の解析 UV は周期的(theta/phi が [0,1] を wrap)で、シームを
跨ぐ面は旧 UV 三角形が [0,1] をほぼ丸ごと張る。これらは「シーム帯分類で除外される
前提」と conftest が明記する seam 面であり、旧 UV 三角形の軸別 extent > 0.5(周期の
半分超 = wrap の標準判定)で検出してシーム帯へ回す。この扱いをしないと、この wrap 面の
点(旧 UV が折り返して基準値が物理的に無意味)が内部点に混ざり、主ゲートを不当に
下げる(実測: 非 wrap 内部点は 55 dB でありながら wrap 点は ~19 dB、解像度を上げても
床が残る)。閾値・解像度は確定値のまま、内部点の membership のみを intent に沿って
精緻化している。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from atlasmith.bake import bake_maps
from atlasmith.metrics import masked_psnr
from atlasmith.pack import _naive_unwrap_and_pack

MESHES = ("cube_mesh", "sphere_mesh", "torus_mesh")

_NEW_RES = 512  # 新 atlas 解像度(確定値表)。
_OLD_RES = 256  # 旧テクスチャ解像度(確定値表)。
_PADDING = 8  # xatlas パッキング兼 bake ガター(確定値表・単一ソース同期)。
_K_SAMPLES = 8  # 面あたりの一様重心サンプル数(+重心で K+1)。
_SAMPLE_SEED = 12345  # サンプル重心の固定シード(決定的)。
_SEAM_ERODE = 2  # seam_margin = 2.0 テクセル → 8 近傍 2 回 erosion。
_WRAP_EXTENT = 0.5  # 旧 UV 三角形の軸別 extent がこれを超える面は周期シームを wrap。

# 内部点比率下限・数(確定値表)。総サンプル点は M*(K+1)、wrap 点も分母に含む。
_INTERIOR_RATIO_MIN = {"cube_mesh": 0.50, "sphere_mesh": 0.40, "torus_mesh": 0.40}
_INTERIOR_COUNT_MIN = 100

_GATE_O1 = 40.0  # 主ゲート(内部点・multisine)。
_GATE_O2 = 20.0  # シーム帯。
_GATE_O4 = 35.0  # 往復(補助)。
_NEG_CEILING = 25.0  # 負の対照の PSNR 天井。
_NEG_MARGIN = 15.0  # 正解 - 誤り の下限。

_MULTISINE_SEED = 1  # 主ゲート用平滑テクスチャ(freqs 3/7/13)のシード。
_APERIODIC_SEED = 2  # 負の対照用非周期テクスチャのシード。
_SCALAR_SEED = 3  # O1 scalar 版(2D 単チャンネル multisine)のシード。


def _point_psnr(test_vals: np.ndarray, ref_vals: np.ndarray, mask: np.ndarray) -> float:
    """点集合 (P, C) の PSNR。`mask` (P,) が選ぶ点の全チャンネルを一括評価する。

    masked_psnr は (H, W, C) + (H, W) mask を取るので、選択点を (n, 1, C) と (n, 1)
    に整形して渡す(空選択は masked_psnr が ValueError にする — 呼び出し側で非空 assert
    済み)。
    """
    sel_test = test_vals[mask][:, np.newaxis, :]
    sel_ref = ref_vals[mask][:, np.newaxis, :]
    full = np.ones(sel_test.shape[:2], dtype=bool)
    return masked_psnr(sel_test, sel_ref, full)


def _uv_to_texel(
    uv_pts: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """UV 点を、それを含むテクセルの (row, col) index に変換する(clamp 境界)。

    テクセル中心 UV = ((c+0.5)/W, (r+0.5)/H)(横断規約)なので、UV 点を含むセルは
    c = floor(u*W), r = floor(v*H)。
    """
    col = np.clip(np.floor(uv_pts[:, 0] * width).astype(np.int64), 0, width - 1)
    row = np.clip(np.floor(uv_pts[:, 1] * height).astype(np.int64), 0, height - 1)
    return row, col


def _build_context(
    mesh,
    *,
    make_texture,
    bilinear_sample,
    rasterize_coverage,
    erode8,
    face_barycentric_samples,
) -> SimpleNamespace:
    """1 メッシュ分のオラクル文脈を構築する(重いので mesh ごとに1回・キャッシュ前提)。

    再展開 → サンプル点生成 → シーム分類 → テクスチャ焼き(multisine/aperiodic の
    正解+3 種の負)→ 往復焼き → 独立サンプラで基準値/被験値を用意し、O1-O4 が使う
    配列を SimpleNamespace に詰めて返す。
    """
    new_mesh, face_map = _naive_unwrap_and_pack(
        mesh, resolution=_NEW_RES, padding_px=_PADDING
    )
    aligned_old_faces = mesh.faces[face_map]  # 行・corner 整列済み旧面(裁定5)。
    old_uv = np.asarray(mesh.uv, dtype=np.float32)

    # サンプル重心: K 点の sqrt 法一様 + 面重心。全面で同じ重心集合を使う(決定的)。
    bary = face_barycentric_samples(_K_SAMPLES, _SAMPLE_SEED)  # (K, 3)
    all_bary = np.vstack([bary, np.full((1, 3), 1.0 / 3.0)])  # (K+1, 3)
    n_per_face = all_bary.shape[0]

    # 新面 i(旧面 = face_map[i])で index を揃える。corner k が一致するので、同じ重心を
    # 新旧の corner に適用すれば同一の物理表面点になる。
    new_tri_uv = np.asarray(new_mesh.uv, dtype=np.float64)[new_mesh.faces]  # (M,3,2)
    old_tri_uv = old_uv.astype(np.float64)[aligned_old_faces]  # (M,3,2)
    new_pts = np.einsum("sk,mkc->msc", all_bary, new_tri_uv).reshape(-1, 2)
    old_pts = np.einsum("sk,mkc->msc", all_bary, old_tri_uv).reshape(-1, 2)

    # wrap 面: 旧 UV 三角形の軸別 extent が周期の半分(0.5)を超える面は周期シームを跨ぐ。
    old_extent = (old_tri_uv.max(axis=1) - old_tri_uv.min(axis=1)).max(axis=1)  # (M,)
    wrap_face = old_extent > _WRAP_EXTENT
    wrap_point = np.repeat(wrap_face, n_per_face)

    # 独立カバレッジ(テスト側ラスタライザ)を新旧で作り、2 テクセル erosion で内側取得。
    new_cov, _new_fid, _new_bary = rasterize_coverage(
        new_mesh.faces, new_mesh.uv, (_NEW_RES, _NEW_RES)
    )
    old_cov, _old_fid, _old_bary = rasterize_coverage(
        mesh.faces, old_uv, (_OLD_RES, _OLD_RES)
    )
    new_interior_cov = erode8(new_cov, _SEAM_ERODE)
    old_interior_cov = erode8(old_cov, _SEAM_ERODE)
    new_row, new_col = _uv_to_texel(new_pts, _NEW_RES, _NEW_RES)
    old_row, old_col = _uv_to_texel(old_pts, _OLD_RES, _OLD_RES)
    far_from_seam = (
        new_interior_cov[new_row, new_col] & old_interior_cov[old_row, old_col]
    )
    interior = far_from_seam & ~wrap_point
    seam = ~interior  # シーム帯 = 内部でない全点(境界近傍 or wrap 面)。
    # non-wrap シーム点 = 境界近傍だが周期シームを跨がない点。O2 の PSNR ゲート
    # (>=20dB)はこれにのみ課す(should-fix1)。wrap 点は物理的に無意味なので除外。
    nonwrap_seam = seam & ~wrap_point

    # --- テクスチャ焼き(全マップ 3ch。O1/O2/O4 は multisine、O3 は aperiodic)。---
    def bake_forward(uv_old_src: np.ndarray, tex: np.ndarray) -> np.ndarray:
        return bake_maps(
            new_mesh.faces,
            new_mesh.uv,
            aligned_old_faces,
            uv_old_src,
            {"m": tex},
            size=(_NEW_RES, _NEW_RES),
            padding_px=_PADDING,
        ).maps["m"]

    tex_ms = make_texture("multisine", (_OLD_RES, _OLD_RES), 3, seed=_MULTISINE_SEED)
    tex_ap = make_texture("aperiodic", (_OLD_RES, _OLD_RES), 3, seed=_APERIODIC_SEED)
    # WHY(swap のテクスチャ選定): channel-swap を強い負にするにはチャンネルが構造的に
    # 異なる必要がある。aperiodic はチャンネルが位相違いのみ(同一 freqs)で、seed 次第で
    # ch0/ch1 が似て PSNR > 25 に紛れ識別力が崩れる(実測 seed2 で 27.6 dB — test-design
    # 注4「25-40dB は識別力欠陥」に該当)。gradient は ch0=u 勾配 / ch1=v 勾配 と直交する
    # ので入替は seed に依らず強い負(~10 dB)。shift/vflip は周期エイリアス回避のため
    # 非周期を使う(既存 T6/T7 と同方針)。
    tex_grad = make_texture("gradient", (_OLD_RES, _OLD_RES), 3)

    new_tex_ms = bake_forward(old_uv, tex_ms)
    # O1 scalar 版(should-fix2): 2D 単チャンネル map。make_texture の channels=1 出力
    # (H,W,1)を [..., 0] で真の 2D (H,W) に落とし、production の (H,W)->(H,W,1) 昇格
    # 分岐(bake_maps の ndim!=3 経路)を経由させる。焼き上がりは (H,W,1) になる。
    tex_scalar = make_texture("multisine", (_OLD_RES, _OLD_RES), 1, seed=_SCALAR_SEED)[
        ..., 0
    ]
    new_tex_scalar = bake_forward(old_uv, tex_scalar)
    # O3 shift/vflip: 非周期の正対応 vs 旧 UV の誤対応(0.1 シフト / V 反転)。
    new_tex_correct_ap = bake_forward(old_uv, tex_ap)
    uv_shift = old_uv + np.float32(0.1)
    new_tex_shift = bake_forward(uv_shift, tex_ap)
    uv_vflip = old_uv.copy()
    uv_vflip[:, 1] = 1.0 - uv_vflip[:, 1]
    new_tex_vflip = bake_forward(uv_vflip, tex_ap)
    # O3 swap: gradient の正対応 vs チャンネル入替済み旧テクスチャ(UV は正対応、
    # 誤りはテクスチャのみ)。
    new_tex_correct_grad = bake_forward(old_uv, tex_grad)
    new_tex_swap = bake_forward(old_uv, tex_grad[..., [1, 0, 2]].copy())

    # O4 往復: 旧 -> 新(new_tex_ms)-> 旧。backward は non-wrap 面に限定する。
    # WHY: 旧 UV レイアウトへ焼き戻す際、wrap 面の巨大三角形が旧テクスチャ全域を
    # 覆って上書きし、backward bake を全域で破壊する(実測: 全 interior 点が wrap
    # カバレッジ内に入り往復 PSNR が 16-29 dB へ落ちる)。転写可能な non-wrap 面だけで
    # 焼き戻すと往復は 53-61 dB(補助ゲート >= 35 を満たす)。wrap 面を内部点から
    # 除外する分類と一貫した扱い。
    nonwrap = ~wrap_face
    old_recon = bake_maps(
        aligned_old_faces[nonwrap],
        old_uv,
        new_mesh.faces[nonwrap],
        new_mesh.uv,
        {"m": new_tex_ms},
        size=(_OLD_RES, _OLD_RES),
        padding_px=_PADDING,
    ).maps["m"]

    # --- 独立サンプラで基準値/被験値(点集合)を用意 ---
    def sample(tex: np.ndarray, pts: np.ndarray) -> np.ndarray:
        return bilinear_sample(tex, pts[:, 0], pts[:, 1])

    return SimpleNamespace(
        interior=interior,
        seam=seam,
        # should-fix1: wrap 点/ non-wrap シーム点/ 面別 wrap 分類を O2 が使う。
        wrap_point=wrap_point,
        nonwrap_seam=nonwrap_seam,
        wrap_face=wrap_face,
        old_extent=old_extent,
        n_total=len(interior),
        # O1/O2
        ref_ms=sample(tex_ms, old_pts),
        test_ms=sample(new_tex_ms, new_pts),
        # O1 scalar 版(should-fix2)。ref (K,) を (K,1) に整形し test (K,1) に揃える。
        ref_scalar=sample(tex_scalar, old_pts)[:, np.newaxis],
        test_scalar=sample(new_tex_scalar, new_pts),
        # O3 shift/vflip(aperiodic)と swap(gradient)は基準・正解が別テクスチャ。
        ref_ap=sample(tex_ap, old_pts),
        correct_ap=sample(new_tex_correct_ap, new_pts),
        test_shift=sample(new_tex_shift, new_pts),
        test_vflip=sample(new_tex_vflip, new_pts),
        ref_grad=sample(tex_grad, old_pts),
        correct_grad=sample(new_tex_correct_grad, new_pts),
        test_swap=sample(new_tex_swap, new_pts),
        # O4
        ref_rt=sample(tex_ms, old_pts),
        test_rt=sample(old_recon, old_pts),
    )


# 文脈は重い(再展開+複数焼き+ラスタライズ)ので mesh 名で1回だけ構築しキャッシュする。
# fixtures は決定的なので、関数スコープの mesh を再構築しても文脈は同一。
_CONTEXT_CACHE: dict[str, SimpleNamespace] = {}


@pytest.fixture
def oracle_context(
    request,
    make_texture,
    bilinear_sample,
    rasterize_coverage,
    erode8,
    face_barycentric_samples,
):
    def _get(mesh_fixture: str) -> SimpleNamespace:
        if mesh_fixture not in _CONTEXT_CACHE:
            mesh = request.getfixturevalue(mesh_fixture)
            _CONTEXT_CACHE[mesh_fixture] = _build_context(
                mesh,
                make_texture=make_texture,
                bilinear_sample=bilinear_sample,
                rasterize_coverage=rasterize_coverage,
                erode8=erode8,
                face_barycentric_samples=face_barycentric_samples,
            )
        return _CONTEXT_CACHE[mesh_fixture]

    return _get


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_o1_main_gate_interior(mesh_fixture, oracle_context) -> None:
    """O1 主ゲート: 内部点・multisine で PSNR >= 40 dB。非空+比率+点数の下限も確認。"""
    ctx = oracle_context(mesh_fixture)
    assert ctx.interior.any()  # 内部点集合の非空(裁定2)。
    n_interior = int(ctx.interior.sum())
    ratio = n_interior / ctx.n_total
    assert ratio >= _INTERIOR_RATIO_MIN[mesh_fixture]
    assert n_interior >= _INTERIOR_COUNT_MIN
    psnr = _point_psnr(ctx.test_ms, ctx.ref_ms, ctx.interior)
    assert psnr >= _GATE_O1


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_o1_main_gate_interior_scalar(mesh_fixture, oracle_context) -> None:
    """O1 scalar 版(should-fix2): 2D 単チャンネル map を焼き内部点 PSNR >= 40 dB。

    確定値表「basecolor/scalar 系を map 別に 40dB」の scalar 側、および production の
    2D スカラー map 分岐(bake_maps が (H,W) 入力を (H,W,1) へ昇格する経路)を再展開後
    オラクルで保護する。3ch 版 O1 と同じ内部点 membership・同じ 40dB 閾値を課す
    (scalar 専用の閾値は作らない)。実測内部点 PSNR: cube 69.7 / sphere 58.3 /
    torus 55.6 dB(いずれも 40dB を大きく超える — production の scalar 分岐は健全)。
    """
    ctx = oracle_context(mesh_fixture)
    assert ctx.interior.any()
    psnr = _point_psnr(ctx.test_scalar, ctx.ref_scalar, ctx.interior)
    assert psnr >= _GATE_O1


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_o2_seam_band(mesh_fixture, oracle_context) -> None:
    """O2 シーム帯: non-wrap シーム点で PSNR >= 20 dB。wrap 点は分類の正しさのみ検証。

    WHY 分離(should-fix1): 旧 O2 は wrap 点(周期シームを跨ぐ面の点。旧 UV が [0,1] を
    折り返し基準値が物理的に無意味)と non-wrap シーム点を単一 mask に集約し1つの PSNR
    で評価していた。実測 torus では wrap 単独 PSNR が 19.467 dB(20dB 未達)なのに、全
    シーム集約は 25.529 dB になって PASS してしまい、「wrap 面の点でも 20dB」を実際には
    証明できていなかった。物理的に無意味な wrap 点に PSNR 下限を課すのは誤りなので、
    20dB ゲートは *有効な non-wrap シーム点*(境界近傍だが周期シームを跨がない点)にのみ
    課し、wrap 点は「分類が正しいこと」だけを別 assert で検証する(PSNR は課さない)。
    閾値 20dB は不変。実測 non-wrap PSNR: cube 68.6 / sphere 51.5 / torus 40.3 dB。
    """
    ctx = oracle_context(mesh_fixture)
    # 20dB ゲートは有効な non-wrap シーム点にのみ課す(wrap 点は含めない)。
    assert ctx.nonwrap_seam.any()
    psnr = _point_psnr(ctx.test_ms, ctx.ref_ms, ctx.nonwrap_seam)
    assert psnr >= _GATE_O2
    # wrap 点は PSNR ゲートを課さず、分類の正しさのみ検証する。
    if ctx.wrap_face.any():
        # 周期メッシュ(sphere/torus): wrap 点は非空で、旧 UV 三角形の軸別 extent が
        # 周期の半分 0.5 で綺麗に分離する(実測 sphere 正規面 max 0.412 / wrap 面 min
        # 0.824、torus 0.031 / 0.969)。この分離が 0.5 しきい値による wrap 分類の妥当性
        # (境界ぎりぎりの面が無く分類が脆くない)を示す。
        assert ctx.wrap_point.any()
        assert float(ctx.old_extent[~ctx.wrap_face].max()) < _WRAP_EXTENT
        assert float(ctx.old_extent[ctx.wrap_face].min()) > _WRAP_EXTENT
    else:
        # cube: UV パッチが 3x2 グリッドの小インセットで周期シームを持たない。wrap 面・
        # wrap 点は存在しないので、その分類(空)が正しいことを確認する。
        assert not ctx.wrap_point.any()


@pytest.mark.parametrize("mesh_fixture", MESHES)
@pytest.mark.parametrize("control", ("shift", "vflip", "swap"))
def test_o3_negative_controls(mesh_fixture, control, oracle_context) -> None:
    """O3 負の対照: 誤対応/誤テクスチャは PSNR < 25 かつ 正解 - 誤り >= 15(内部点)。

    一方向オラクルは往復で相殺される誤り(UV シフト・V 反転・チャンネル入替)を検出
    できる。正解(非周期・正対応)との差が大きく開くことで識別力を示す。
    """
    ctx = oracle_context(mesh_fixture)
    assert ctx.interior.any()
    # swap は gradient(直交チャンネル)基準、shift/vflip は aperiodic 基準。
    if control == "swap":
        correct_vals, wrong_vals, ref_vals = (
            ctx.correct_grad,
            ctx.test_swap,
            ctx.ref_grad,
        )
    else:
        correct_vals = ctx.correct_ap
        wrong_vals = ctx.test_shift if control == "shift" else ctx.test_vflip
        ref_vals = ctx.ref_ap
    psnr_correct = _point_psnr(correct_vals, ref_vals, ctx.interior)
    psnr_wrong = _point_psnr(wrong_vals, ref_vals, ctx.interior)
    assert psnr_wrong < _NEG_CEILING
    assert psnr_correct - psnr_wrong >= _NEG_MARGIN


@pytest.mark.parametrize("mesh_fixture", MESHES)
def test_o4_roundtrip_auxiliary(mesh_fixture, oracle_context) -> None:
    """O4 往復(補助): 旧 -> 新 -> 旧の内部点 PSNR >= 35 dB(劣化量の回帰検知)。"""
    ctx = oracle_context(mesh_fixture)
    assert ctx.interior.any()
    psnr = _point_psnr(ctx.test_rt, ctx.ref_rt, ctx.interior)
    assert psnr >= _GATE_O4
