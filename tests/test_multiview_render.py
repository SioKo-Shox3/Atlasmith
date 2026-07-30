"""moderngl レンダラのゲート(計画v4 §5 Step 2-3 ゲート3〜10)。

GL コンテキストを要するテストには **`@pytest.mark.gl`** を付ける
(`tests/conftest.py` の能力検出が、GL の無い環境では自動的に skip へ落とす)。
入口検証と依存不在の経路は GL を一切使わないので **マーカーを付けない** —
CI(`-m "not ml and not gl"`)でも実行され、契約が守られていることを保証する。

このファイルの主ゲートは **ゲート3(独立ラスタライザとの面ID一致)**。オラクルは
`tests/conftest.py` の `_rasterize_screen_zbuffer`(screen 空間 z バッファ)で、
production とは別実装・別の勝者決定規則を持つ。
"""

from __future__ import annotations

import logging
import sys
import warnings

import numpy as np
import pytest
import trimesh

from atlasmith.segmentation.multiview import RenderedView
from atlasmith.segmentation.multiview import render as render_module
from atlasmith.segmentation.multiview.cameras import (
    build_camera,
    build_cameras,
    validate_frustum,
)
from atlasmith.segmentation.multiview.faceid import encode_face_codes
from atlasmith.segmentation.multiview.render import _ModernglRenderer
from atlasmith.types import MeshData

# 主ゲートの解像度。面と面の境界を除いた「内部画素」を十分な割合で残すために、
# 円筒(128 面・細い側面三角形)でも成立する大きさを採る。
_GATE_IMAGE_SIZE = 384
# 主ゲートの視線方向。軸に平行だと facet が edge-on になって面が消えるので、
# どの軸にも平行でない斜め方向を 1 本だけ決め打つ(決定的)。
_OBLIQUE_DIRECTION = np.array([0.55, 0.45, 0.70])

# `cube_mesh` の facet i(conftest の `faces_def` 順: +X, -X, +Y, -Y, +Z, -Z)は
# 面 index 2i / 2i+1 の 2 三角形。V 方向ゲートで使う +Y facet は面 4/5。
_CUBE_PLUS_Y_FACES = (4, 5)


def _mesh_with_basecolor(mesh: MeshData, texture: np.ndarray) -> MeshData:
    """`mesh` に basecolor を持たせた新しい `MeshData`(元は書き換えない)。"""
    return MeshData(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uv=mesh.uv,
        maps={"basecolor": texture},
        source_vertex=mesh.source_vertex,
    )


def _independent_face_normal(mesh: MeshData, face_index: int) -> np.ndarray:
    """テスト側で独立に計算した単位面法線(production の実装を import しない)。"""
    corners = mesh.vertices[mesh.faces[face_index]]
    raw = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    return raw / np.linalg.norm(raw)


def _expected_normal_colour(normal: np.ndarray) -> np.ndarray:
    """オブジェクト空間法線の符号化色 `round(255 * (n * 0.5 + 0.5))`。"""
    return np.round(np.clip(normal * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.int64)


def _face_colours(view: RenderedView) -> dict[int, np.ndarray]:
    """視点内で「各面がちょうど 1 色である」ことを確かめつつ面 → 色を返す。

    `flat` 修飾子が効いていれば面内の色は定数になる(法線シェーディング時)。
    複数色なら `AssertionError` で落とす。
    """
    colours: dict[int, np.ndarray] = {}
    for face_id in np.unique(view.face_id[view.coverage]):
        pixels = view.color[view.face_id == face_id]
        unique_rows = np.unique(pixels, axis=0)
        assert unique_rows.shape[0] == 1, (
            f"face {int(face_id)} rendered {unique_rows.shape[0]} distinct colours; "
            "flat shading of in_normal is broken"
        )
        colours[int(face_id)] = unique_rows[0].astype(np.int64)
    return colours


# ---------------------------------------------------------------------------
# 入口検証・依存不在(GL 不要 = CI でも走る)
# ---------------------------------------------------------------------------


def test_missing_moderngl_raises_an_actionable_import_error(monkeypatch) -> None:
    """moderngl 未導入の経路が、導入手順と代替を示す `ImportError` になる。

    導入済みの環境でこの経路を踏むために `sys.modules["moderngl"] = None` を
    差し込む(`import moderngl` は None エントリを見つけると `ImportError`)。
    **この monkeypatch 可能性こそが「module 直下ではなく関数内 import」の理由**
    (計画v4 §2.1 規約2 の WHY)。
    """
    monkeypatch.setitem(sys.modules, "moderngl", None)

    with pytest.raises(ImportError) as excinfo:
        render_module._import_moderngl()
    message = str(excinfo.value)
    assert "uv sync --extra ml" in message
    assert "--segmenter geometric" in message
    assert excinfo.value.__cause__ is not None  # `from e` で原因を連鎖している


def test_renderer_rejects_small_image_size(cube_mesh: MeshData) -> None:
    """`image_size < 8` は `ValueError`(ゲート10)。"""
    with pytest.raises(ValueError, match="image_size must be >= 8"):
        _ModernglRenderer(cube_mesh, image_size=7, shading="normal")
    with pytest.raises(ValueError, match="image_size must be an int"):
        _ModernglRenderer(cube_mesh, image_size=64.0, shading="normal")  # type: ignore[arg-type]


def test_renderer_rejects_unknown_shading(cube_mesh: MeshData) -> None:
    """未知の `shading` は `ValueError`(ゲート10)。"""
    with pytest.raises(ValueError, match="unknown shading"):
        _ModernglRenderer(cube_mesh, image_size=64, shading="phong")


def test_renderer_rejects_a_mesh_without_faces() -> None:
    """面 0 枚のメッシュは、moderngl の内部エラーではなく `ValueError` で止まる。"""
    mesh = MeshData(
        vertices=np.zeros((3, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="0 faces"):
        _ModernglRenderer(mesh, image_size=64, shading="normal")


def test_renderer_rejects_bad_face_codes(cube_mesh: MeshData) -> None:
    """`face_codes` の shape / 値域違反は `ValueError`(internal 引数も検証する)。"""
    with pytest.raises(ValueError, match=r"face_codes must have shape \(12,\)"):
        _ModernglRenderer(
            cube_mesh,
            image_size=64,
            shading="normal",
            face_codes=np.arange(5, dtype=np.int64),
        )
    with pytest.raises(ValueError, match=r"within \[0, 16777214\]"):
        _ModernglRenderer(
            cube_mesh,
            image_size=64,
            shading="normal",
            face_codes=np.full(12, -1, dtype=np.int64),
        )


def test_render_view_before_enter_raises_runtime_error(cube_mesh: MeshData) -> None:
    """未 `__enter__` の `render_view` は `RuntimeError`(裁定2 / §0-A 条件10)。

    GL は一切触らない(状態検査が最初に走る)ので、GPU の無い環境でも実行される。
    `rebake(segmentation=...)` を `with` 無しで呼ぶ経路は現実に踏まれるため、
    「まだ入っていない」を「もう出た」と別のメッセージで報告する。
    """
    renderer = _ModernglRenderer(cube_mesh, image_size=64, shading="normal")
    camera = build_camera(
        cube_mesh.vertices, np.array([0.0, 0.0, 1.0]), projection="orthographic"
    )

    with pytest.raises(RuntimeError, match="before __enter__"):
        renderer.render_view(camera)


def test_shading_falls_back_to_normal_without_texture(
    cube_mesh: MeshData, caplog
) -> None:
    """テクスチャ非搭載メッシュは `"normal"` へ落ち、`logging.info` に残る(ゲート8)。

    どう壊れたら落ちるか: 黙って落ちる実装(ログ無し)や、`shading="texture"` の
    まま真っ黒を返す実装になった瞬間に落ちる。
    """
    caplog.set_level(logging.INFO, logger=render_module.__name__)
    renderer = _ModernglRenderer(cube_mesh, image_size=64, shading="texture_normal")

    assert renderer.shading == "normal"
    assert any(
        "falls back to 'normal'" in record.getMessage() for record in caplog.records
    )


def test_mesh_far_from_origin_warns_about_float32(cube_mesh: MeshData) -> None:
    """原点から極端に離れたメッシュで float32 精度の劣化を警告する(2周目 N1)。

    実測(2周目レビュー): `|AABB 中心| / R` が `2e4` で全画素一致率 0.999179、
    `2e6` で内部画素が 6226 件不一致(= 主ゲートなら FAIL)。頂点は f4 で GPU へ
    送るので、`validate_frustum` が float64 で包含を保証した後に精度が落ちる。
    GL 不要(警告は `__init__` で出る)。
    """
    far = MeshData(
        vertices=cube_mesh.vertices + 1.0e6,
        faces=cube_mesh.faces,
        uv=cube_mesh.uv,
        source_vertex=cube_mesh.source_vertex,
    )
    with pytest.warns(UserWarning, match="away from the origin"):
        _ModernglRenderer(far, image_size=64, shading="normal")

    # 原点付近の同じメッシュでは鳴らない(警告が常時鳴っては意味が無い)。
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ModernglRenderer(cube_mesh, image_size=64, shading="normal")


# ---------------------------------------------------------------------------
# ゲート3: 独立ラスタライザとの面ID一致(主ゲート)
# ---------------------------------------------------------------------------


def _compare_with_oracle(
    mesh: MeshData,
    projection: str,
    project_to_ndc,
    rasterize_screen_zbuffer,
    faceid_interior_mask,
) -> tuple[int, int, int, float]:
    """1 視点をレンダして独立オラクルと比べ、`(cov, interior, mismatch, agree)`。"""
    camera = build_camera(mesh.vertices, _OBLIQUE_DIRECTION, projection=projection)
    # オラクルの前提(全頂点が視錐台内 = near クリッピング無し)をテスト内で明示。
    validate_frustum(mesh.vertices, camera)

    with _ModernglRenderer(
        mesh, image_size=_GATE_IMAGE_SIZE, shading="normal"
    ) as renderer:
        view = renderer.render_view(camera)

    oracle_face_id, _oracle_z = rasterize_screen_zbuffer(
        project_to_ndc(mesh.vertices, camera), mesh.faces, _GATE_IMAGE_SIZE
    )
    interior = faceid_interior_mask(view.face_id)
    return (
        int(view.coverage.sum()),
        int(interior.sum()),
        int((view.face_id[interior] != oracle_face_id[interior]).sum()),
        float((view.face_id == oracle_face_id).mean()),
    )


# 主ゲートの fixture ごとの内部画素比率の下限(2周目レビュー B2/B3 の裁定)。
# **WHY 一律 0.30(計画v4 §5 Step 2-3 ゲート3 の本文)を捨てたか**: この比率は
# 「厳密性を証明できた画素の割合」ではなく **面密度 ÷ 解像度** を測っている。実測
# (384px)は cube 0.945 / cylinder 0.634 / sphere 0.765〜0.793 に対し、torus
# (2,048 面)は 0.268〜0.289 で**一律下限を割る**(mismatch は 0 なのに)。
# 一律の値は「粗いメッシュでは緩すぎ、細かいメッシュでは通らない」ので、fixture ごとに
# 実測より少し低い値を置き、**非空であることは全 fixture に課したまま**にする。
_INTERIOR_RATIO_FLOOR = {
    "cube_mesh": 0.90,  # 実測 0.9445 / 0.9473
    "capped_cylinder_mesh": 0.55,  # 実測 0.6344 / 0.6339
    "sphere_mesh": 0.70,  # 実測 0.7934 / 0.7652
}


@pytest.mark.gl
@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
@pytest.mark.parametrize(
    "mesh_name", ["cube_mesh", "capped_cylinder_mesh", "sphere_mesh"]
)
def test_face_id_matches_independent_rasterizer(
    request: pytest.FixtureRequest,
    mesh_name: str,
    projection: str,
    project_to_ndc,
    rasterize_screen_zbuffer,
    faceid_interior_mask,
) -> None:
    """GL の面IDが、独立な screen 空間 z バッファと一致する(★主ゲート)。

    合否:
      - `_faceid_interior_mask` の画素で一致率 **== 1.0(厳密)**
      - 全画素一致率 **>= 0.99**(境界画素の食い違い上限)
      - 量化域の非空: `interior > 0` かつ `interior / coverage >=`
        **fixture ごとの下限**(`_INTERIOR_RATIO_FLOOR`。計画本文の一律 0.30 からの
        変更は 2周目レビュー B2/B3 のオーケストレーター裁定)

    **このゲートの適用範囲(誤読すると危険なので明記する)**: 「内部画素の厳密一致」は
    **面が粗いメッシュでのみ成立する指標**である。実測(2周目レビュー):
    2,048 面の torus で内部画素比率は 0.27(→ 別テスト
    `test_face_id_matches_oracle_on_a_dense_mesh` へ分離)、14,160 面の UV 球で 0.20、
    89,400 面では **内部画素が literally 0**、144 万面 @128px では全画素一致率も
    0.986 まで落ちる。つまり **Atlasmith の実対象規模(1万〜20万面)では、この主ゲートは
    文字どおり空虚になる**。ここは「小さい fixture で面IDの厳密性そのものを証明する」
    目的に限定されたゲートであり、**実規模での妥当性の検証は Step 2-7 の E2E
    (焼き直し PSNR)に委ねる**。

    どう壊れたら落ちるか: MVP の転置忘れ・行反転の欠落・面IDの補間 — いずれも内部画素の
    厳密一致が最初に落ちる。**深度規約の反転だけはここでは捕まらない**(オラクルが同じ
    `Camera.mvp` を消費するため相殺する)ので、`tests/test_multiview_cameras.py` の
    `test_projection_maps_near_far_to_minus_one_plus_one` が別途固定している。
    """
    mesh: MeshData = request.getfixturevalue(mesh_name)
    n_coverage, n_interior, mismatched, agreement = _compare_with_oracle(
        mesh, projection, project_to_ndc, rasterize_screen_zbuffer, faceid_interior_mask
    )

    assert n_coverage > 0
    assert n_interior > 0
    floor = _INTERIOR_RATIO_FLOOR[mesh_name]
    assert n_interior / n_coverage >= floor, (
        f"interior pixels are only {n_interior}/{n_coverage} = "
        f"{n_interior / n_coverage:.3f} of the coverage (floor {floor} for "
        f"{mesh_name}); the exact-match gate would be weaker than measured"
    )

    assert mismatched == 0, (
        f"{mismatched} interior pixels disagree with the independent rasterizer "
        f"({mesh_name}, {projection})"
    )
    assert agreement >= 0.99, (
        f"overall face-id agreement {agreement:.4f} < 0.99 ({mesh_name}, {projection})"
    )


@pytest.mark.gl
@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_face_id_matches_oracle_on_a_dense_mesh(
    torus_mesh: MeshData,
    projection: str,
    project_to_ndc,
    rasterize_screen_zbuffer,
    faceid_interior_mask,
) -> None:
    """面密度の高いメッシュでも厳密一致は成立する(内部画素比率は要求しない)。

    **WHY 主ゲートから分けるか**(2周目レビュー B2/B3): `torus_mesh` は 2,048 面 /
    384px で内部画素比率が **0.268〜0.289** と主ゲートの下限を構造的に割る — が、
    それは「厳密性が成り立たない」のではなく「面が細かくて境界画素の割合が高い」
    だけで、**mismatch は 0**(実測)。したがって比率の要求を外し、
    **内部画素の厳密一致 + 全画素一致率**だけを課す。この形なら面密度が上がっても
    (内部画素が 1 つでも残る限り)意味のあるゲートとして残る。
    """
    n_coverage, n_interior, mismatched, agreement = _compare_with_oracle(
        torus_mesh,
        projection,
        project_to_ndc,
        rasterize_screen_zbuffer,
        faceid_interior_mask,
    )

    assert n_coverage > 0
    # 比率は課さないが「1 画素も無い」= 空虚は許さない。
    assert n_interior > 0, (
        "no interior pixel survived on the dense mesh, so the exact-match "
        "comparison would be vacuous; lower the face density or raise image_size"
    )
    assert mismatched == 0, (
        f"{mismatched} interior pixels disagree with the independent rasterizer "
        f"(torus_mesh, {projection})"
    )
    assert agreement >= 0.99, (
        f"overall face-id agreement {agreement:.4f} < 0.99 (torus_mesh, {projection})"
    )


# ---------------------------------------------------------------------------
# ゲート4: V 方向(2 本)
# ---------------------------------------------------------------------------


@pytest.mark.gl
def test_v_direction_without_texture(cube_mesh: MeshData) -> None:
    """テクスチャを通さない V 方向ゲート: +Y facet が画面上半分に出る(ゲート4(i))。

    カメラはほぼ `+Z` から(up 分岐の閾値 `|dot(dir, [0,0,1])| > 0.99` を満たすので
    up は `+Y`)。**厳密な軸方向 `+Z` にしない WHY**: +Y facet が視線と平行
    (edge-on)になり、投影面積 0 = 1 画素も描かれず、ゲートが空虚になる。
    up の分岐を保ったまま +Y facet が見える最小限の傾き(約 5.7 度)を使う。

    幾何(手計算): true_up ≈ (0, 0.995, -0.10) なので、+Y facet(y=0.5, z∈[-0.5,0.5])
    の screen y は [0.4475, 0.5475]、前面(+Z facet)は [-0.5475, 0.4475]。
    つまり +Y facet は**必ず**画面上端側に来る。
    """
    direction = np.array([0.0, 0.10, 0.995])
    camera = build_camera(cube_mesh.vertices, direction, projection="orthographic")
    assert np.array_equal(camera.up, np.array([0.0, 1.0, 0.0]))

    with _ModernglRenderer(cube_mesh, image_size=256, shading="normal") as renderer:
        view = renderer.render_view(camera)

    plus_y = np.isin(view.face_id, np.array(_CUBE_PLUS_Y_FACES))
    rows = np.nonzero(plus_y)[0]
    assert rows.size > 0, "the +Y facet drew 0 pixels; the gate would be vacuous"
    half = view.face_id.shape[0] // 2
    assert int(rows.max()) < half, (
        f"the +Y facet reaches row {int(rows.max())} (>= H/2 = {half}); glReadPixels "
        "rows are bottom-up and must be reversed before use"
    )


@pytest.mark.gl
def test_v_direction_through_the_texture_path() -> None:
    """テクスチャ経由の V 方向ゲート: 2x2 の四隅が同配置で出る(ゲート4(ii))。

    テクスチャ(row 0 = 画像上端 = V=0)は 左上=赤 / 右上=緑 / 左下=青 / 右下=白。
    正面から見たクアッドの出力四隅が同じ配置になることを見る。

    **WHY ゲート4 が 2 本必要か**: このテクスチャ経路だけだと「アップロードの V
    規約」と「読み戻しの行反転」が**両方逆でも緑になる**(相殺する)。
    テクスチャを通さない (i) と組で初めて向きが固定される。
    """
    # z = 0 平面のクアッド。UV は「画面上端 = V=0」になるよう割り当てる。
    vertices = np.array(
        [
            [-0.5, -0.5, 0.0],  # 左下
            [0.5, -0.5, 0.0],  # 右下
            [0.5, 0.5, 0.0],  # 右上
            [-0.5, 0.5, 0.0],  # 左上
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv = np.array(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    red = [1.0, 0.0, 0.0]
    green = [0.0, 1.0, 0.0]
    blue = [0.0, 0.0, 1.0]
    white = [1.0, 1.0, 1.0]
    texture = np.array([[red, green], [blue, white]], dtype=np.float32)
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps={"basecolor": texture},
        source_vertex=np.arange(4, dtype=np.int64),
    )
    palette = np.round(np.array([red, green, blue, white]) * 255.0).astype(np.int64)

    camera = build_camera(
        vertices, np.array([0.0, 0.0, 1.0]), projection="orthographic"
    )
    image_size = 128
    with _ModernglRenderer(mesh, image_size=image_size, shading="texture") as renderer:
        view = renderer.render_view(camera)

    # 各テクセル中心 (u, v) ∈ {0.25, 0.75}^2 に対応するワールド点を画素へ写す。
    # ワールド x = -0.5 + u、ワールド y = 0.5 - v(上の UV 割り当ての逆写像)。
    sample_world = np.array(
        [
            [-0.25, 0.25, 0.0],  # 左上テクセル(赤)
            [0.25, 0.25, 0.0],  # 右上テクセル(緑)
            [-0.25, -0.25, 0.0],  # 左下テクセル(青)
            [0.25, -0.25, 0.0],  # 右下テクセル(白)
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate(
        [sample_world, np.ones((4, 1), dtype=np.float64)], axis=1
    )
    clip = homogeneous @ camera.mvp.T
    ndc = clip[:, :3] / clip[:, 3:4]
    cols = np.round((ndc[:, 0] + 1.0) * 0.5 * image_size - 0.5).astype(int)
    rows = np.round((1.0 - ndc[:, 1]) * 0.5 * image_size - 0.5).astype(int)

    for expected_index, (row, col) in enumerate(zip(rows, cols)):
        assert bool(view.coverage[row, col])
        sampled = view.color[row, col].astype(np.int64)
        distances = np.abs(palette - sampled).sum(axis=1)
        assert int(np.argmin(distances)) == expected_index, (
            f"pixel (row={row}, col={col}) is {sampled.tolist()}, closest to "
            f"palette entry {int(np.argmin(distances))} but expected "
            f"{expected_index} ({palette[expected_index].tolist()})"
        )
        # バイリニアで隣のテクセルが少し混ざるが、「灰色一色」で全部通るような
        # 空虚な合格を防ぐため絶対距離にも上限を置く。
        assert int(distances[expected_index]) <= 24


# ---------------------------------------------------------------------------
# ゲート5〜7: 被覆整合・GPU 上の sentinel・`flat` 補間の不在
# ---------------------------------------------------------------------------


@pytest.mark.gl
def test_coverage_matches_face_id_on_gpu(capped_cylinder_mesh: MeshData) -> None:
    """`coverage <=> (face_id >= 0)` の違反が 0 件(ゲート5)。

    production 側は違反を `ValueError` で落とすので、ここは「落ちずに返ってきた
    結果が実際に整合している」ことを外から確認する二重の網。alpha ベースなので
    色値には依存しない。
    """
    camera = build_camera(
        capped_cylinder_mesh.vertices, _OBLIQUE_DIRECTION, projection="perspective"
    )
    with _ModernglRenderer(
        capped_cylinder_mesh, image_size=256, shading="normal"
    ) as renderer:
        view = renderer.render_view(camera)

    violations = int((view.coverage != (view.face_id >= 0)).sum())
    assert violations == 0
    assert int(view.coverage.sum()) > 0


# GPU 上で往復させる sentinel 面コード(§0-A 条件6 が指定する 13 値から 12 個)。
# **選び方**: `cube_mesh` は 12 面なので 1 個落とす必要がある。落としたのは `3`
# — バイト境界を跨ぐ組(254/255 = R->G、65534/65535 = G->B)と 24bit 上限付近
# (16777213/16777214)はすべて残し、低位の連番 0/1/2 で「小さい値」の代表は
# 足りているため、`3` だけが新しいバイトパターンを 1 つも足さない。
_SENTINEL_FACE_CODES = np.array(
    [0, 1, 2, 254, 255, 256, 257, 65534, 65535, 65536, 16777213, 16777214],
    dtype=np.int64,
)


@pytest.mark.gl
def test_sentinel_face_ids_survive_the_gpu(
    cube_mesh: MeshData,
    project_to_ndc,
    rasterize_screen_zbuffer,
    faceid_interior_mask,
) -> None:
    """バイト境界を跨ぐ面IDが GPU 上で厳密に往復する(★ゲート6 / BL-10)。

    numpy だけの往復試験は、GPU の固定小数点書き込み(ディザ・ブレンド・sRGB)を
    一切通っていない。ここでは実際に GPU へ書かせ、**各面領域のデコード値が
    sentinel と厳密一致**することを見る。面と面の対応は独立ラスタライザで取る。

    立方体は 1 視点から高々 6 面しか見えないので、反対向きの 2 視点で 12 面すべてを
    覆い、**各 sentinel について被覆非空**を確かめる(空虚な合格を防ぐ)。
    """
    image_size = 256
    seen: dict[int, int] = {int(code): 0 for code in _SENTINEL_FACE_CODES}
    with _ModernglRenderer(
        cube_mesh,
        image_size=image_size,
        shading="normal",
        face_codes=_SENTINEL_FACE_CODES,
    ) as renderer:
        for direction in (np.array([1.0, 1.0, 1.0]), np.array([-1.0, -1.0, -1.0])):
            camera = build_camera(
                cube_mesh.vertices, direction, projection="orthographic"
            )
            view = renderer.render_view(camera)
            oracle_face_id, _z = rasterize_screen_zbuffer(
                project_to_ndc(cube_mesh.vertices, camera), cube_mesh.faces, image_size
            )
            interior = faceid_interior_mask(view.face_id)
            assert int(interior.sum()) > 0

            expected = _SENTINEL_FACE_CODES[oracle_face_id[interior]]
            actual = view.face_id[interior].astype(np.int64)
            bad = int((expected != actual).sum())
            assert bad == 0, (
                f"{bad} interior pixels decoded to a different face id than the "
                f"sentinel emitted for that face (first mismatch: "
                f"expected {expected[expected != actual][:4].tolist()}, "
                f"got {actual[expected != actual][:4].tolist()})"
            )
            for code in np.unique(actual):
                seen[int(code)] += int((actual == code).sum())

    unseen = sorted(code for code, count in seen.items() if count == 0)
    assert unseen == [], (
        f"these sentinel face ids were never drawn, so their round-trip was never "
        f"actually tested: {unseen}"
    )


@pytest.mark.gl
def test_flat_qualifier_blocks_face_id_interpolation(monkeypatch) -> None:
    """`flat` 修飾子の実効性を provoking vertex で実証する(ゲート7 の作り直し)。

    **計画本文のゲート7 は空虚**(§0-A 条件6): 面ごとにアンロールしていて 3 頂点の
    `in_code` が同値なので、`flat` を外しても平滑補間の結果が同値になり落ちない。
    そこで**テスト専用に 1 つの三角形の 3 頂点へ異なる code を積み**、その面が
    覆う全画素のデコード値が**ちょうど 1 種類**であることを見る。補間が起きていれば
    必ず複数種類になる。どの頂点が採られるか(provoking vertex の規約)には
    依存しない書き方にしてある。

    注入は production の内部フック `_build_code_attribute`(頂点ごとのコード色を
    作る唯一の場所)への monkeypatch で行う — 固定契約である
    `_ModernglRenderer.__init__` の signature は変えない。
    """
    corner_codes = np.array([10, 200_000, 16_000_000], dtype=np.int64)

    def _distinct_corner_codes(face_codes: np.ndarray) -> np.ndarray:
        # 3 頂点へ別々のコードを積む(通常経路は np.repeat で同値になる)。
        return encode_face_codes(corner_codes)

    monkeypatch.setattr(render_module, "_build_code_attribute", _distinct_corner_codes)

    vertices = np.array(
        [[-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0]], dtype=np.float64
    )
    mesh = MeshData(
        vertices=vertices,
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        source_vertex=np.arange(3, dtype=np.int64),
    )
    camera = build_camera(
        vertices, np.array([0.0, 0.0, 1.0]), projection="orthographic"
    )

    with _ModernglRenderer(mesh, image_size=128, shading="normal") as renderer:
        view = renderer.render_view(camera)

    covered = view.face_id[view.coverage].astype(np.int64)
    assert covered.size > 0, "the probe triangle drew 0 pixels; the gate is vacuous"
    distinct = np.unique(covered)
    assert distinct.size == 1, (
        f"the probe triangle decoded to {distinct.size} distinct face ids "
        f"{distinct[:8].tolist()}; the `flat` qualifier on in_code is missing or "
        "ineffective"
    )
    assert int(distinct[0]) in corner_codes.tolist()


# ---------------------------------------------------------------------------
# ゲート8: shading の 3 値
# ---------------------------------------------------------------------------


@pytest.mark.gl
def test_shading_modes_produce_the_expected_colours(cube_mesh: MeshData) -> None:
    """`"texture"` / `"normal"` / `"texture_normal"` の色が期待どおり(ゲート8)。

    定数色テクスチャを使うので、バイリニア補間が入っても albedo は厳密に一定。
    期待値はテスト側で独立に計算した面法線から組む(production の法線関数を
    import しない)。
    """
    albedo = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    texture = np.tile(albedo, (8, 8, 1)).astype(np.float32)
    mesh = _mesh_with_basecolor(cube_mesh, texture)
    camera = build_camera(mesh.vertices, _OBLIQUE_DIRECTION, projection="orthographic")
    albedo_bytes = np.round(albedo.astype(np.float64) * 255.0).astype(np.int64)

    results: dict[str, dict[int, np.ndarray]] = {}
    for shading in ("texture", "normal", "texture_normal"):
        with _ModernglRenderer(mesh, image_size=192, shading=shading) as renderer:
            assert renderer.shading == shading  # テクスチャがあるので落ちない
            results[shading] = _face_colours(renderer.render_view(camera))

    visible = sorted(results["normal"])
    assert visible, "no face was visible; the gate would be vacuous"
    for face_id in visible:
        normal_bytes = _expected_normal_colour(_independent_face_normal(mesh, face_id))
        # 量子化の丸めは実装依存(0.5*255 = 127.5 のような同点)なので atol を置く。
        np.testing.assert_allclose(results["normal"][face_id], normal_bytes, atol=1)
        np.testing.assert_allclose(results["texture"][face_id], albedo_bytes, atol=1)
        expected_mix = np.round(
            (0.5 * albedo_bytes / 255.0 + 0.5 * normal_bytes / 255.0) * 255.0
        )
        np.testing.assert_allclose(
            results["texture_normal"][face_id], expected_mix, atol=2
        )


@pytest.mark.gl
def test_normal_shading_is_view_independent(cube_mesh: MeshData) -> None:
    """同一面の色が全視点で**ビット同一**(オブジェクト空間法線の証明 — ゲート8)。

    ビュー空間法線だと同じ面が視点ごとに別色になり、視点間融合が壊れる。
    `maps={}` の `cube_mesh` を使うので、経路としては「テクスチャ無し →
    `"normal"` へ自動フォールバック」の実挙動も同時に見ている。
    """
    cameras = build_cameras(cube_mesh.vertices, n_views=6, projection="perspective")
    per_view: list[dict[int, np.ndarray]] = []
    with _ModernglRenderer(cube_mesh, image_size=192, shading="texture") as renderer:
        assert renderer.shading == "normal"
        for camera in cameras:
            per_view.append(_face_colours(renderer.render_view(camera)))

    # 立方体の 1 面はどの視点からも見えるわけではない(裏面は隠れる)。比較するのは
    # 「2 視点以上に現れた面」で、その全出現が同一色であること。
    first_seen: dict[int, tuple[int, np.ndarray]] = {}
    repeated: list[int] = []
    for view_index, colours in enumerate(per_view):
        for face_id, colour in colours.items():
            if face_id not in first_seen:
                first_seen[face_id] = (view_index, colour)
                continue
            reference_view, reference = first_seen[face_id]
            repeated.append(face_id)
            assert np.array_equal(colour, reference), (
                f"face {face_id} is {colour.tolist()} in view {view_index} but "
                f"{reference.tolist()} in view {reference_view}; the normal is not "
                "object-space (or the shading depends on the camera)"
            )
    assert len(set(repeated)) >= 2, (
        "fewer than 2 faces were visible from more than one view, so the "
        f"view-independence comparison is nearly vacuous: {sorted(set(repeated))}"
    )


# ---------------------------------------------------------------------------
# ゲート9: 寿命とリソース解放(sabotage)
# ---------------------------------------------------------------------------


@pytest.mark.gl
def test_renderer_lifecycle_is_enforced(cube_mesh: MeshData) -> None:
    """3 状態(未入場 / 入場中 / 退場後)の契約が守られる(ゲート9 / 裁定2)。"""
    camera = build_camera(
        cube_mesh.vertices, np.array([0.0, 0.0, 1.0]), projection="orthographic"
    )
    renderer = _ModernglRenderer(cube_mesh, image_size=64, shading="normal")

    with pytest.raises(RuntimeError, match="before __enter__"):
        renderer.render_view(camera)

    with renderer as entered:
        assert entered is renderer
        with pytest.raises(RuntimeError, match="already entered"):
            renderer.__enter__()
        view = renderer.render_view(camera)
        assert int(view.coverage.sum()) > 0

    with pytest.raises(RuntimeError, match="after __exit__"):
        renderer.render_view(camera)
    with pytest.raises(RuntimeError, match="already been exited"):
        renderer.__enter__()


@pytest.mark.gl
def test_three_consecutive_contexts_do_not_leak(cube_mesh: MeshData) -> None:
    """`create_context` を 3 回連続で行っても失敗しない(リーク検出 — ゲート9)。

    どう壊れたら落ちるか: `__exit__` が ctx を解放しない実装に戻ると、ドライバの
    資源を握ったまま次のコンテキスト生成に進み、この 3 連続のどこかで落ちる。
    """
    camera = build_camera(
        cube_mesh.vertices, np.array([0.0, 0.0, 1.0]), projection="orthographic"
    )
    for _ in range(3):
        with _ModernglRenderer(cube_mesh, image_size=64, shading="normal") as renderer:
            view = renderer.render_view(camera)
            assert view.face_id.dtype == np.int32
            assert view.color.dtype == np.uint8
            assert view.coverage.dtype == np.bool_


@pytest.mark.gl
@pytest.mark.parametrize("projection", ["orthographic", "perspective"])
def test_needle_mesh_warns_about_tiny_screen_coverage(projection: str) -> None:
    """細長いメッシュで画面被覆が極小 / 0 のとき警告する(2周目レビュー B4)。

    カメラは AABB の**外接球**に合わせるので、短い辺が 2 本ある形状では投影面積が
    画面のごく一部になる。実測(384px): `1 x 1e-2 x 1e-2` で被覆 0.95%、
    `1 x 1e-6 x 1e-6` では **0 画素**。以前は例外もログも警告も無く「全面背景」の
    view を返していた(= 上流の融合が黙って無意味な入力を受け取る)。

    扁平(1:1:1e-4)は画面の 29% を占めるので警告しない — 壊れるのは短い辺が 2 本の
    ときだけ、という実測に沿ってある。
    """
    thin = trimesh.creation.box(extents=(1.0, 1.0e-2, 1.0e-2))
    needle = trimesh.creation.box(extents=(1.0, 1.0e-6, 1.0e-6))

    for box, expected in ((thin, "covers only"), (needle, "drew 0 pixels")):
        mesh = MeshData(
            vertices=np.asarray(box.vertices, dtype=np.float64),
            faces=np.asarray(box.faces, dtype=np.int64),
            source_vertex=np.arange(len(box.vertices), dtype=np.int64),
        )
        camera = build_camera(mesh.vertices, _OBLIQUE_DIRECTION, projection=projection)
        with _ModernglRenderer(mesh, image_size=384, shading="normal") as renderer:
            with pytest.warns(UserWarning, match=expected) as caught:
                view = renderer.render_view(camera)
        message = str(caught[0].message)
        assert "bounding sphere" in message  # 原因(外接球フィット)
        assert "image_size" in message  # 行動可能な示唆
        assert float(view.coverage.mean()) < 0.01


@pytest.mark.gl
def test_image_size_beyond_the_gl_limit_is_rejected(cube_mesh: MeshData) -> None:
    """ドライバの上限を超える `image_size` を行動可能な `ValueError` にする(N5)。

    以前は生の `moderngl.Error: the framebuffer is not complete` が出て、原因
    (要求サイズが `GL_MAX_TEXTURE_SIZE` を超えた)が呼び出し側に伝わらなかった。
    上限はドライバ依存(当開発機は 32768)なので、入場時に実測値と突き合わせる。
    """
    renderer = _ModernglRenderer(cube_mesh, image_size=1 << 17, shading="normal")
    with pytest.raises(ValueError) as excinfo:
        renderer.__enter__()
    message = str(excinfo.value)
    assert "GL_MAX_TEXTURE_SIZE" in message
    assert "image_size" in message


@pytest.mark.gl
def test_rendered_view_shapes_follow_the_contract(cube_mesh: MeshData) -> None:
    """`RenderedView` の shape / dtype 契約(計画v4 §2.1)。"""
    image_size = 64
    camera = build_camera(
        cube_mesh.vertices, _OBLIQUE_DIRECTION, projection="orthographic"
    )
    with _ModernglRenderer(
        cube_mesh, image_size=image_size, shading="normal"
    ) as renderer:
        view = renderer.render_view(camera)

    assert isinstance(view, RenderedView)
    assert view.face_id.shape == (image_size, image_size)
    assert view.color.shape == (image_size, image_size, 3)
    assert view.coverage.shape == (image_size, image_size)
    assert view.face_id.dtype == np.int32
    assert view.color.dtype == np.uint8
    assert view.coverage.dtype == np.bool_
