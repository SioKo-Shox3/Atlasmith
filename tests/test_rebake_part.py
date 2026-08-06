"""Step 2-7: `rebake`/CLI の部位経路の結線ゲート(計画v4 §5 Step 2-7)。

測るもの:

- **一方向オラクル**(部位経路・`geometric` 固定): 旧テクスチャを旧 UV で独立に
  サンプルした基準値と、焼き上がり新テクスチャを新 UV で同じ独立サンプラで
  サンプルした被験値を比較する。内部点 PSNR >= 40 dB。
  **WHY `geometric` に固定するか**: ML バックエンドは非決定的(SAM2 のマスク提案)
  なので、数値ゲートの基準にはしない。ML の品質は
  `tests/test_multiview_sam2.py` の領分。
- **E2E 島整合**(非循環設計・v2 BL-4): CLI が書いた GLB を読み戻し、**テスト側の
  独立 face 照合**(`conftest._match_faces_by_position` — 面の 3 頂点 3D 座標の
  厳密一致)で出力面→入力面の対応を復元してから入力側ラベルを伝播し、UV アイランド
  ごとのラベル単一性を検査する。production の `face_map` も
  `_check_island_part_consistency` も**使わない** — 使うと「実装が自分自身に
  一致する」だけの循環ゲートになる。出力の再分割もしない。
- **ガター規約**(計画v2 §2.3a 手順3 / 2026-08-06 裁定1): `rebake` が
  `g = min(padding_px, floor(padding_px·texture_size/D))` を計算して `bake_maps`
  へ渡していること、およびその `g` で異部位のガターが衝突しないこと。
- **naive 後方互換**(v2 BL-6): API 出力テクスチャが手組みパイプラインと bit 一致。
- **CLI**: 衝突判定3規則・`--segmenter sam2` の 2 経路(明示=`ImportError` /
  既定経由=警告つき geometric フォールバック)。

数値はすべて確定値(下の定数)であり、緩めない。実測値は各テストの docstring に
記録する(2026-08-06、xatlas 0.0.11 / 当リポジトリの pin)。
"""

from __future__ import annotations

import logging
import subprocess
import sys
import warnings

import numpy as np
import pytest

# WHY test_cli からの import: 計画v4 §5 Step 2-7 が「既存 `tests/test_cli.py:28-53`
# の実 console script 解決ヘルパを**再利用**する」と指定している。同じ解決ロジックを
# 2 箇所に書くと、entry point の場所が変わったとき片方だけ腐る。pytest の既定
# import mode(`prepend`)は `tests/` を `sys.path` へ入れるので素の module 名で引ける
# (`tests/__init__.py` は無い)。
# subprocess のハング上限も同じ理由で共有する(実測に基づく値と WHY は test_cli 側)。
from test_cli import (
    _CLI_TIMEOUT_SEC,
    _SAM2_CLI_TIMEOUT_SEC,
    _find_atlasmith_executable,
)

from atlasmith import MeshData, rebake
from atlasmith.bake import bake_maps
from atlasmith.cli import _build_parser, _check_flag_conflicts, main
from atlasmith.io import load_mesh, save_mesh
from atlasmith.metrics import masked_psnr
from atlasmith.pack import _naive_unwrap_and_pack, _part_unwrap_and_pack
from atlasmith.segmentation import DihedralSegmenter

# --- 確定値(計画v4 §5 Step 2-7 / test-design 準拠。test_bake_oracle と同方式)---
_NEW_RES = 512  # 新 atlas 解像度。
_OLD_RES = 256  # 旧テクスチャ解像度。
_PADDING = 8  # xatlas パッキング(bake のガターは `g` へ縮む — 部位経路)。
_K_SAMPLES = 8  # 面あたりの一様重心サンプル数(+重心で K+1)。
_SAMPLE_SEED = 12345  # サンプル重心の固定シード(決定的)。
_SEAM_ERODE = 2  # seam_margin = 2 テクセル(8 近傍 2 回 erosion)。
_WRAP_EXTENT = 0.5  # 旧 UV 三角形の軸別 extent がこれを超える面は周期シームを wrap。

_GATE_PSNR = 40.0  # 主ゲート(内部点・平滑テクスチャ)。
_INTERIOR_COUNT_MIN = 100  # 内部点の絶対下限(空虚なゲートにしない)。

# 部位数 P と内部点比率の下限。実測値(2026-08-06)を併記する — 下限は実測から
# わずかに下げた値であって「通る値」を後から書いたものではない。
_ORACLE_CASES = {
    # fixture 名: (期待 P, 内部点比率の下限, 実測比率, 実測 PSNR dB)
    "capped_cylinder_mesh": (3, 0.85, 0.8932, 60.55),
    "two_cubes_mesh": (12, 0.80, 0.8333, 59.61),
    "torus_mesh": (1, 0.70, 0.7368, 53.67),
}

# ガター規約ゲートの構成(実測: atlas 200x159 / D=200 / g=5、素の padding_px=8 なら
# 異部位ガターが 361 テクセル重なる)。
_GUTTER_RES = 128
_GUTTER_PADDING = 8

# ガター消失(`g == 0`)を作る構成。`padding_px=1` は D > texture_size(部位経路の
# 常態)である限り必ず `g = 1*ts // D == 0` になる(実測: capped_cylinder @ ts=128 で
# D=141)。
_ZERO_GUTTER_RES = 128
_ZERO_GUTTER_PADDING = 1

_GEOMETRIC = ["--segmenter", "geometric"]


# ---------------------------------------------------------------------------
# 共通ヘルパ
# ---------------------------------------------------------------------------


def _with_texture(mesh: MeshData, make_texture, *, seed: int = 1) -> MeshData:
    """fixture へ平滑テクスチャ(multisine)を載せた新しい `MeshData` を返す。"""
    return MeshData(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uv=mesh.uv,
        maps={
            "basecolor": make_texture("multisine", (_OLD_RES, _OLD_RES), 3, seed=seed)
        },
        source_vertex=mesh.source_vertex,
    )


def _uv_to_texel(points: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """UV 点を、それを含むテクセルの (row, col) へ落とす(テクセル中心規約)。"""
    col = np.clip(np.floor(points[:, 0] * size).astype(np.int64), 0, size - 1)
    row = np.clip(np.floor(points[:, 1] * size).astype(np.int64), 0, size - 1)
    return row, col


def _part_coverage_masks(
    uv: np.ndarray,
    faces: np.ndarray,
    part_of_face: np.ndarray,
    n_parts: int,
    size: int,
) -> np.ndarray:
    """部位ごとのテクセル被覆 `(P, size, size) bool` を**部位ごと独立に**書く。

    **WHY conftest の `rasterize_coverage` を使わないか**: あちらは「先に走査した
    face_id が勝つ」tie-break を持つため、**重なりが原理的に見えない**。ここは
    「異部位のガターが重なっていないか」を測るのが目的なので、部位ごとに独立の
    マスクへ書いてから重ね合わせる(`tests/test_pack_part.py` の N7 群と同方式)。
    """
    masks = np.zeros((n_parts, size, size), dtype=bool)
    uv_px = np.asarray(uv, dtype=np.float64) * size - 0.5
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
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


def _cross_part_overlap(masks: np.ndarray, dilate8, iterations: int) -> int:
    """各部位を `iterations` 回膨張させたときに 2 部位以上が奪い合うテクセル数。"""
    grown = np.stack([dilate8(masks[part], iterations) for part in range(len(masks))])
    return int((grown.sum(axis=0) > 1).sum())


# ---------------------------------------------------------------------------
# 合否基準1: 一方向オラクル(部位経路・geometric 固定)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(_ORACLE_CASES))
def test_part_path_interior_psnr_meets_gate(
    case,
    request,
    tmp_path,
    make_texture,
    bilinear_sample,
    rasterize_coverage,
    erode8,
    face_barycentric_samples,
    match_faces_by_position,
) -> None:
    """部位経路で焼き直したテクスチャの内部点 PSNR が 40 dB 以上(実測 53.7-60.6 dB)。

    基準値と被験値はどちらも conftest の**独立サンプラ**(production の
    `bake._bilinear_sample` を import しない・裁定9)で取る。新旧の対応は
    production の `face_map` ではなく**座標一致による独立照合**で復元するので、
    往復で相殺される誤り(V 反転・面取り違え・corner 置換ミス)がこの一方向比較で
    表に出る。

    内部点 = 新旧どちらのカバレッジ境界からも 2 テクセル以上内側で、かつ旧 UV の
    周期シームを跨がない面の点(torus の解析 UV は周期的で、跨ぐ面の基準値は
    物理的に無意味 — `test_bake_oracle.py` と同じ扱い)。
    """
    expected_parts, ratio_floor, _measured_ratio, _measured_psnr = _ORACLE_CASES[case]
    mesh = _with_texture(request.getfixturevalue(case), make_texture)
    source = tmp_path / "in.glb"
    baked = tmp_path / "out.glb"
    save_mesh(mesh, source)

    labels = DihedralSegmenter().segment(mesh)
    assert int(labels.max()) + 1 == expected_parts, (
        f"{case}: fixture no longer splits into {expected_parts} parts; the oracle "
        "would be measuring a different layout than the recorded one"
    )

    rebake(
        source,
        baked,
        texture_size=_NEW_RES,
        padding_px=_PADDING,
        granularity="part",
        segmentation=DihedralSegmenter(),
    )
    in_mesh = load_mesh(source)
    out_mesh = load_mesh(baked)
    face_map, corner_perm = match_faces_by_position(out_mesh, in_mesh)

    # 新面 i と、それに対応する旧面の corner を**同じ順**に並べる。以降は同一の
    # 重心座標を両者へ当てれば同じ物理表面点になる。
    bary = np.vstack(
        [face_barycentric_samples(_K_SAMPLES, _SAMPLE_SEED), np.full((1, 3), 1.0 / 3.0)]
    )
    n_per_face = bary.shape[0]
    new_tri_uv = np.asarray(out_mesh.uv, dtype=np.float64)[out_mesh.faces]
    old_faces_aligned = np.take_along_axis(in_mesh.faces[face_map], corner_perm, axis=1)
    old_tri_uv = np.asarray(in_mesh.uv, dtype=np.float64)[old_faces_aligned]
    new_points = np.einsum("sk,mkc->msc", bary, new_tri_uv).reshape(-1, 2)
    old_points = np.einsum("sk,mkc->msc", bary, old_tri_uv).reshape(-1, 2)

    wrap_face = (old_tri_uv.max(axis=1) - old_tri_uv.min(axis=1)).max(axis=1) > (
        _WRAP_EXTENT
    )
    wrap_point = np.repeat(wrap_face, n_per_face)

    new_cov, _fid, _bary = rasterize_coverage(
        out_mesh.faces, out_mesh.uv, (_NEW_RES, _NEW_RES)
    )
    old_cov, _fid2, _bary2 = rasterize_coverage(
        in_mesh.faces, in_mesh.uv, (_OLD_RES, _OLD_RES)
    )
    new_row, new_col = _uv_to_texel(new_points, _NEW_RES)
    old_row, old_col = _uv_to_texel(old_points, _OLD_RES)
    interior = (
        erode8(new_cov, _SEAM_ERODE)[new_row, new_col]
        & erode8(old_cov, _SEAM_ERODE)[old_row, old_col]
        & ~wrap_point
    )

    n_interior = int(interior.sum())
    assert n_interior >= _INTERIOR_COUNT_MIN, (
        f"{case}: only {n_interior} interior sample points (< "
        f"{_INTERIOR_COUNT_MIN}); the PSNR gate would be vacuous"
    )
    ratio = n_interior / interior.size
    assert ratio >= ratio_floor, (
        f"{case}: interior point ratio {ratio:.4f} fell below {ratio_floor}; the "
        "sample set no longer represents the surface"
    )

    reference = bilinear_sample(
        in_mesh.maps["basecolor"], old_points[:, 0], old_points[:, 1]
    )
    measured = bilinear_sample(
        out_mesh.maps["basecolor"], new_points[:, 0], new_points[:, 1]
    )
    psnr = masked_psnr(
        measured[interior][:, np.newaxis, :],
        reference[interior][:, np.newaxis, :],
        np.ones((n_interior, 1), dtype=bool),
    )
    assert psnr >= _GATE_PSNR, (
        f"{case}: interior PSNR {psnr:.2f} dB < {_GATE_PSNR} dB "
        f"({n_interior} interior points, ratio {ratio:.4f})"
    )


# ---------------------------------------------------------------------------
# 合否基準2: E2E 島整合(CLI 出力 GLB / 非循環)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(_ORACLE_CASES))
def test_cli_output_islands_lie_inside_a_single_part(
    case,
    request,
    tmp_path,
    make_texture,
    match_faces_by_position,
    island_label_violations,
) -> None:
    """CLI が書いた GLB の各 UV アイランドがちょうど 1 部位に収まる(違反 0 件)。

    対応の復元は座標一致の独立照合だけで行い、`face_map` の再利用も出力の再分割も
    しない(v2 BL-4 の非循環設計)。

    **識別力の所在**: `torus_mesh` は P=1 なので構造的に違反し得ない(記録のため
    残す)。実際に「アイランドが部位をまたげない」ことを測っているのは
    `capped_cylinder_mesh`(P=3)と `two_cubes_mesh`(P=12)である。
    """
    expected_parts = _ORACLE_CASES[case][0]
    mesh = _with_texture(request.getfixturevalue(case), make_texture)
    source = tmp_path / "in.glb"
    baked = tmp_path / "out.glb"
    save_mesh(mesh, source)

    exit_code = main(
        [
            str(source),
            "-o",
            str(baked),
            "--texture-size",
            str(_NEW_RES),
            "--padding",
            str(_PADDING),
            "--granularity",
            "part",
            *_GEOMETRIC,
        ]
    )
    assert exit_code == 0

    in_mesh = load_mesh(source)
    out_mesh = load_mesh(baked)
    face_map, _corner_perm = match_faces_by_position(out_mesh, in_mesh)
    assert sorted(face_map.tolist()) == list(range(in_mesh.faces.shape[0])), (
        "the independent position match is not a bijection onto the input faces; "
        "the part path must neither drop nor duplicate faces"
    )

    labels = DihedralSegmenter().segment(in_mesh)
    assert int(labels.max()) + 1 == expected_parts
    violations = island_label_violations(out_mesh.faces, labels[face_map])
    assert violations == [], (
        f"{case}: {len(violations)} UV island(s) span more than one part: "
        f"{violations[:4]}"
    )


def test_console_script_output_islands_lie_inside_a_single_part(
    tmp_path,
    capped_cylinder_mesh,
    make_texture,
    match_faces_by_position,
    island_label_violations,
) -> None:
    """同じ検査を実 console script の subprocess 出力に対して行う(合否基準10)。

    `main()` 直呼びでは `[project.scripts]` の entry point マッピングが壊れていても
    気づけない(`test_cli._find_atlasmith_executable` の WHY)。
    """
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    baked = tmp_path / "out.glb"
    save_mesh(mesh, source)

    result = subprocess.run(
        [
            _find_atlasmith_executable(),
            str(source),
            "-o",
            str(baked),
            "--texture-size",
            str(_NEW_RES),
            "--padding",
            str(_PADDING),
            "--granularity",
            "part",
            *_GEOMETRIC,
        ],
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT_SEC,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert baked.exists()

    in_mesh = load_mesh(source)
    out_mesh = load_mesh(baked)
    face_map, _ = match_faces_by_position(out_mesh, in_mesh)
    labels = DihedralSegmenter().segment(in_mesh)
    assert island_label_violations(out_mesh.faces, labels[face_map]) == []


# ---------------------------------------------------------------------------
# 合否基準3: sabotage(検査ヘルパが歯を持つこと)
# ---------------------------------------------------------------------------


def test_island_checker_reports_hand_made_violation(island_label_violations) -> None:
    """手組みの「1 アイランドが 2 ラベルを跨ぐ」データを検査が違反として報告する。

    上のゲートが「違反 0 件」を主張する以上、**検査が違反を検出できること自体**を
    別に示さないとゲートが空虚になる(実データで 0 件なのは、検査が常に空リストを
    返す実装でも同じだから)。面 0 と面 1 は頂点 0 と 2 を共有するので 1 アイランド。
    """
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    assert island_label_violations(faces, np.array([0, 1], dtype=np.int64)) == [
        (0, [0, 1])
    ]
    # 負の対照: 同じ形で単一ラベルなら違反ゼロ(誤検出しない)。
    assert island_label_violations(faces, np.array([0, 0], dtype=np.int64)) == []
    # 頂点を共有しない 2 面は別アイランドなので、ラベルが違っても違反ではない。
    disjoint = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    assert island_label_violations(disjoint, np.array([0, 1], dtype=np.int64)) == []


def test_face_matcher_rejects_ambiguous_position_triples(
    match_faces_by_position,
) -> None:
    """照合ヘルパは「同一座標三つ組の面が 2 枚ある」入力を fail させる(skip しない)。

    E2E ゲートは「対応が一意に決まる」ことに全面的に依存している。曖昧な入力を
    黙って片方に当てると、違反を見逃したまま green になる。
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    # 同じ 3 頂点を巻き順違いで 2 枚(座標三つ組が一意でない)。
    ambiguous = MeshData(
        vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    )
    with pytest.raises(AssertionError, match="same set of"):
        match_faces_by_position(ambiguous, ambiguous)


# ---------------------------------------------------------------------------
# 合否基準4: naive 後方互換
# ---------------------------------------------------------------------------


def test_naive_api_texture_is_bit_identical_to_hand_built_pipeline(
    tmp_path, capped_cylinder_mesh, make_texture
) -> None:
    """`rebake(..., granularity="naive")` の出力テクスチャが手組み経路と bit 一致。

    手組み経路 = `load_mesh` → `_naive_unwrap_and_pack` → `bake_maps` → `save_mesh`
    (Phase 1 の `rebake` そのもの)。部位経路の追加で naive のガター反復数が
    変わっていないこと(素の `padding_px` のまま)もここで固定される。

    **比較対象は「出力テクスチャ配列」であって GLB のバイト列ではない**:
    `io.mesh._material_name_for_stem` がマテリアル名を**出力ファイルの stem から**
    導出するので、内容が同一でもファイル名が違えば GLB のバイト列は必ず異なる
    (実測 2026-08-06)。将来「バイト列が違う」と混乱しないようここに記録する。
    """
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    save_mesh(mesh, source)

    api_path = tmp_path / "naive_api.glb"
    rebake(
        source,
        api_path,
        texture_size=_NEW_RES,
        padding_px=_PADDING,
        granularity="naive",
    )

    loaded = load_mesh(source)
    new_mesh, face_map = _naive_unwrap_and_pack(
        loaded, resolution=_NEW_RES, padding_px=_PADDING
    )
    result = bake_maps(
        new_mesh.faces,
        new_mesh.uv,
        loaded.faces[face_map],
        loaded.uv,
        loaded.maps,
        size=(_NEW_RES, _NEW_RES),
        padding_px=_PADDING,
    )
    manual_path = tmp_path / "naive_manual.glb"
    save_mesh(
        MeshData(
            vertices=new_mesh.vertices,
            faces=new_mesh.faces,
            uv=new_mesh.uv,
            maps=result.maps,
            source_vertex=new_mesh.source_vertex,
        ),
        manual_path,
    )

    api_texture = load_mesh(api_path).maps["basecolor"]
    manual_texture = load_mesh(manual_path).maps["basecolor"]
    assert api_texture.shape == (_NEW_RES, _NEW_RES, 3)
    assert np.array_equal(api_texture, manual_texture), (
        "rebake(granularity='naive') no longer reproduces the Phase 1 pipeline "
        f"bit-for-bit (max abs diff {np.abs(api_texture - manual_texture).max()})"
    )


def test_naive_cli_subprocess_succeeds(
    tmp_path, capped_cylinder_mesh, make_texture
) -> None:
    """`--granularity naive` が console script 経由で完走する。"""
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    baked = tmp_path / "out.glb"
    save_mesh(mesh, source)

    result = subprocess.run(
        [
            _find_atlasmith_executable(),
            str(source),
            "-o",
            str(baked),
            "--texture-size",
            str(_NEW_RES),
            "--padding",
            str(_PADDING),
            "--granularity",
            "naive",
        ],
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT_SEC,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert baked.exists()


def test_unknown_granularity_raises(tmp_path, cube_mesh) -> None:
    """未知の `granularity` は黙って既定へ倒さず `ValueError`。"""
    source = tmp_path / "in.glb"
    save_mesh(cube_mesh, source)
    with pytest.raises(ValueError, match="unknown granularity"):
        rebake(source, tmp_path / "out.glb", granularity="chunky")


def test_naive_with_segmentation_backend_raises(tmp_path, cube_mesh) -> None:
    """`granularity="naive"` + `segmentation` は `ValueError`(黙って無視しない)。"""
    source = tmp_path / "in.glb"
    save_mesh(cube_mesh, source)
    with pytest.raises(ValueError, match="does not segment the mesh"):
        rebake(
            source,
            tmp_path / "out.glb",
            granularity="naive",
            segmentation=DihedralSegmenter(),
        )


# ---------------------------------------------------------------------------
# 合否基準5: CLI 衝突判定(計画v4 §4.3 の 3 規則 + §0-B の読み替え)
#
# **拒否側は `main()` 経由**で測る(`parser.error` は再展開が始まる前に SystemExit を
# 投げるので、パイプラインは走らない)。**許容側は `_check_flag_conflicts` を直接**
# 呼ぶ — `main()` で測ると「許容された組み合わせ」が実際に SAM2 を構築してしまい、
# 判定規約のテストが数分かかる ML テストに化けるため。
# ---------------------------------------------------------------------------


def _parse(argv: list[str]):
    parser = _build_parser()
    return parser, parser.parse_args(argv)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--seg-angle", "45"),
        ("--seg-min-faces", "4"),
        ("--seg-views", "8"),
        ("--seg-model", "facebook/sam2.1-hiera-large"),
    ],
)
def test_naive_rejects_every_seg_star_flag(tmp_path, cube_mesh, flag, value) -> None:
    """規則1: `--granularity naive` + `--seg-*` のいずれかを明示 → `parser.error`。"""
    source = tmp_path / "in.glb"
    save_mesh(cube_mesh, source)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                str(source),
                "-o",
                str(tmp_path / "out.glb"),
                "--granularity",
                "naive",
                flag,
                value,
            ]
        )
    assert excinfo.value.code != 0


@pytest.mark.parametrize("backend", ["geometric", "sam2"])
def test_naive_rejects_explicit_segmenter(tmp_path, cube_mesh, backend) -> None:
    """規則2: `--granularity naive` + `--segmenter` を**明示** → `parser.error`。

    `geometric` を明示した場合も拒否する(素朴経路は分割しないので、どちらの値でも
    指定が黙って無視されることに変わりはない)。
    """
    source = tmp_path / "in.glb"
    save_mesh(cube_mesh, source)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                str(source),
                "-o",
                str(tmp_path / "out.glb"),
                "--granularity",
                "naive",
                "--segmenter",
                backend,
            ]
        )
    assert excinfo.value.code != 0


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--seg-views", "8"), ("--seg-model", "facebook/sam2.1-hiera-large")],
)
def test_explicit_geometric_rejects_sam2_only_flags(
    tmp_path, cube_mesh, flag, value
) -> None:
    """規則3: `--segmenter geometric` を**明示** + sam2 専用フラグ → `parser.error`。"""
    source = tmp_path / "in.glb"
    save_mesh(cube_mesh, source)
    with pytest.raises(SystemExit) as excinfo:
        main([str(source), "-o", str(tmp_path / "out.glb"), *_GEOMETRIC, flag, value])
    assert excinfo.value.code != 0


def test_naive_alone_is_accepted() -> None:
    """規則2 の除外: `--segmenter` **未指定**なら `--granularity naive` は通る。"""
    parser, args = _parse(["in.glb", "-o", "out.glb", "--granularity", "naive"])
    assert args.segmenter is None  # 番兵のまま = 未指定
    _check_flag_conflicts(parser, args)  # 例外が出ないこと自体が assert


def test_default_segmenter_accepts_sam2_only_flags() -> None:
    """規則3 の除外(§0-B): 既定経由の sam2 に `--seg-views` を渡すのは合法。

    計画本文の規則3 は「`--segmenter geometric`(**明示または既定**)+ `--seg-views`
    → error」だったが、裁定 E で既定が `sam2` になったため「既定で geometric」という
    状況自体が存在しない。既定経由をここで弾くと、既定バックエンドの視点数を
    指定できなくなる。
    """
    parser, args = _parse(["in.glb", "-o", "out.glb", "--seg-views", "8"])
    assert args.segmenter is None
    _check_flag_conflicts(parser, args)


def test_explicit_geometric_accepts_shared_seg_flags() -> None:
    """`--seg-angle` / `--seg-min-faces` は両バックエンドで有効(規則3 の対象外)。"""
    parser, args = _parse(
        [
            "in.glb",
            "-o",
            "out.glb",
            *_GEOMETRIC,
            "--seg-angle",
            "45",
            "--seg-min-faces",
            "3",
        ]
    )
    _check_flag_conflicts(parser, args)


# ---------------------------------------------------------------------------
# 合否基準6: `--segmenter sam2` の 2 経路
# ---------------------------------------------------------------------------

# `[ml]` 未導入を模すスクリプト(新規インタプリタで実行する)。
#
# **WHY `_import_sam2` を関数ごと差し替えないか**(実測 2026-08-06 で最初こうして
# 失敗した): `_import_sam2` / `_import_torch` の**中身が**「素の `ImportError` を
# 3 経路を提示する行動可能なメッセージへ翻訳する」責務そのものなので、関数を
# 丸ごとスタブに置き換えると**翻訳が一度も走らない**。出てくるのはスタブが投げた
# 文字列だけで、「行動可能なメッセージ」を何も検証できない。
#
# よって潰すのは**その下の `import torch` / `import sam2`** にする。`sys.meta_path`
# の先頭に「`torch` / `sam2` を名指しで拒否する finder」を差し込むと、production の
# `try: import torch except ImportError:` が実際に発火して本物のメッセージが出る。
# torch/sam2 が入った開発機でも、入っていない CI でも同じ経路を通る。
#
# 両方を塞ぐ理由: `build_sam2_segmenter` は `_resolve_device`(torch)を
# `_import_sam2` より先に呼ぶので、本当に `[ml]` の無い機械では torch 側が先に
# 落ちる。どちらのメッセージも `uv sync --extra ml` /
# `pip install "atlasmith[ml]"` / `--segmenter geometric` の 3 経路を提示する契約
# (`sam2_masks.py`)なので、どちらが先に落ちてもこのテストの assert は同じ。
_NO_ML_PREAMBLE = """
import sys


class _BlockMlExtras:
    BLOCKED = {"torch", "sam2"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"simulated: {fullname} is not installed")
        return None


sys.meta_path.insert(0, _BlockMlExtras())

from atlasmith.cli import main

sys.exit(main(sys.argv[1:]))
"""


def test_explicit_sam2_without_ml_exits_nonzero_with_actionable_message(
    tmp_path, cube_mesh, make_texture
) -> None:
    """`--segmenter sam2` を**明示**+`[ml]` 未導入 → 非ゼロ終了 + 行動可能なメッセージ。

    明示指定は「黙ってフォールバックしない」経路(計画v4 §2.6)。新規インタプリタで
    走らせるので、この検査は本セッションの `sys.modules` を汚さない。
    """
    mesh = _with_texture(cube_mesh, make_texture)
    source = tmp_path / "in.glb"
    save_mesh(mesh, source)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _NO_ML_PREAMBLE,
            str(source),
            "-o",
            str(tmp_path / "out.glb"),
            "--texture-size",
            str(_ZERO_GUTTER_RES),
            "--segmenter",
            "sam2",
        ],
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT_SEC,
    )

    assert result.returncode != 0, f"stdout={result.stdout!r}"
    # 行動可能: 3 経路すべてが示されること。
    assert "uv sync --extra ml" in result.stderr
    assert 'pip install "atlasmith[ml]"' in result.stderr
    assert "--segmenter geometric" in result.stderr
    assert not (tmp_path / "out.glb").exists(), (
        "the explicit sam2 path must not write an output file when it cannot build "
        "the backend"
    )


def test_default_sam2_without_ml_warns_and_falls_back_to_geometric(
    tmp_path, cube_mesh, make_texture, monkeypatch
) -> None:
    """**既定経由**で `[ml]` 未導入 → 警告を出して `geometric` で続行(§0-B 裁定 E)。

    CI(`uv sync --locked` = extras 無し)と既定 CLI パスを green に保つための裁定。
    「黙って」ではないこと(= 警告が出ること)がこのテストの主題。

    ここは in-process の monkeypatch で足りる: 測るのは「`ImportError` が出たら
    警告して geometric で続行する」という CLI の分岐であって、`ImportError` の
    **中身**ではない(行動可能なメッセージの検証は上の subprocess テストが
    production の翻訳経路ごと担当している)。ただし理由が握り潰されていない
    ことは見る — 警告文に元の例外文が埋まっていること。
    """
    from atlasmith.segmentation.multiview import sam2_masks

    reason = "simulated: the [ml] extra is not installed"

    def _missing(*_args, **_kwargs):
        raise ImportError(reason)

    # `_resolve_device`(torch)が `_import_sam2` より先に走るので両方を潰す。
    monkeypatch.setattr(sam2_masks, "_import_torch", _missing)
    monkeypatch.setattr(sam2_masks, "_import_sam2", _missing)

    mesh = _with_texture(cube_mesh, make_texture)
    source = tmp_path / "in.glb"
    baked = tmp_path / "out.glb"
    save_mesh(mesh, source)

    with pytest.warns(
        UserWarning, match="falling back to --segmenter geometric"
    ) as rec:
        exit_code = main(
            [
                str(source),
                "-o",
                str(baked),
                "--texture-size",
                str(_NEW_RES),
                "--padding",
                str(_PADDING),
            ]
        )

    assert exit_code == 0
    assert baked.exists()
    assert any(reason in str(w.message) for w in rec), (
        "the fallback warning must quote why the [ml] backend could not be built"
    )


@pytest.mark.gl
@pytest.mark.ml
def test_explicit_sam2_cli_subprocess_succeeds(
    tmp_path, cube_mesh, make_texture
) -> None:
    """`--segmenter sam2` の明示指定が実機で完走する(小 fixture・`--seg-views 6`)。

    品質は測らない(ML の数値ゲートは `tests/test_multiview_sam2.py`)。ここは
    「CLI → `make_sam2_segmenter` → `with` による寿命管理 → `rebake`」の結線だけ。
    """
    mesh = _with_texture(cube_mesh, make_texture)
    source = tmp_path / "in.glb"
    baked = tmp_path / "out.glb"
    save_mesh(mesh, source)

    result = subprocess.run(
        [
            _find_atlasmith_executable(),
            str(source),
            "-o",
            str(baked),
            "--texture-size",
            str(_ZERO_GUTTER_RES),
            "--padding",
            str(_PADDING),
            "--granularity",
            "part",
            "--segmenter",
            "sam2",
            "--seg-views",
            "6",
        ],
        capture_output=True,
        text=True,
        timeout=_SAM2_CLI_TIMEOUT_SEC,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert baked.exists()


# ---------------------------------------------------------------------------
# 合否基準7: アトラス寸法の通知(2026-08-06 裁定で「比較」から「帰結」へ変更)
#
# 計画 §2.5 は `D > texture_size` を警告にしていたが、実測でそれは部位経路の**常態**
# (4 fixture x angle 2 種 x resolution 6 種 = 48 構成すべてで成立)だと判明したため、
# 警告は「ガターが 0 に落ちた」ときだけに絞り、密度低下自体は `logging.info` に落と
# した(`src/atlasmith/__init__.py` の `rebake` docstring に実測を記録)。
# ---------------------------------------------------------------------------


def test_rebake_warns_when_gutter_collapses_to_zero(
    tmp_path, capped_cylinder_mesh, make_texture
) -> None:
    """`g == 0` で `rebake` が警告する(実測: ts=128 / pad=1 → D=141 / g=0)。"""
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    save_mesh(mesh, source)

    with pytest.warns(UserWarning, match="bake gutter collapsed to 0 texels"):
        rebake(
            source,
            tmp_path / "out.glb",
            texture_size=_ZERO_GUTTER_RES,
            padding_px=_ZERO_GUTTER_PADDING,
            granularity="part",
        )


def test_rebake_is_silent_when_the_gutter_survives(
    tmp_path, capped_cylinder_mesh, make_texture
) -> None:
    """負の対照: `g >= 1` なら `D > texture_size` でも警告は出ない。

    この構成の実測は atlas 200x159 / D=200 / g=5 — `D > texture_size` は成立して
    いるので、「常態では黙る」ことをここで固定する(裁定の核心)。
    """
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    save_mesh(mesh, source)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rebake(
            source,
            tmp_path / "out.glb",
            texture_size=_GUTTER_RES,
            padding_px=_GUTTER_PADDING,
            granularity="part",
        )


def test_rebake_is_silent_when_the_user_asked_for_no_padding(
    tmp_path, capped_cylinder_mesh, make_texture
) -> None:
    """負の対照: `padding_px=0` は利用者がガター無しを選んだ状態なので警告しない。

    `g == 0` という**結果**は同じでも意味が違う。ここで警告すると「要求どおりに
    動いた」ことに文句を言うことになり、本当にガターが消えた事故(上のテスト)の
    信号が埋もれる。
    """
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    save_mesh(mesh, source)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rebake(
            source,
            tmp_path / "out.glb",
            texture_size=_GUTTER_RES,
            padding_px=0,
            granularity="part",
        )

    assert [w for w in caught if "gutter collapsed" in str(w.message)] == []


def test_rebake_logs_atlas_density_at_info(
    tmp_path, capped_cylinder_mesh, make_texture, caplog
) -> None:
    """密度低下は `logging.info` に数値つきで残る(常態なので出続けることを固定)。"""
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    save_mesh(mesh, source)

    with caplog.at_level(logging.INFO, logger="atlasmith"):
        rebake(
            source,
            tmp_path / "out.glb",
            texture_size=_GUTTER_RES,
            padding_px=_GUTTER_PADDING,
            granularity="part",
        )

    records = [r for r in caplog.records if "part atlas is" in r.getMessage()]
    assert len(records) == 1, f"expected exactly one density record, got {records}"
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert "density hint, not a cap" in message
    assert f"texture_size={_GUTTER_RES}" in message


def test_cli_does_not_duplicate_the_gutter_warning(
    tmp_path, capped_cylinder_mesh, make_texture
) -> None:
    """裁定2: ガター消失警告は `rebake` の責務。CLI は同じ警告を二重に出さない。"""
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    save_mesh(mesh, source)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exit_code = main(
            [
                str(source),
                "-o",
                str(tmp_path / "out.glb"),
                "--texture-size",
                str(_ZERO_GUTTER_RES),
                "--padding",
                str(_ZERO_GUTTER_PADDING),
                *_GEOMETRIC,
            ]
        )

    assert exit_code == 0
    gutter_warnings = [
        w for w in caught if "bake gutter collapsed to 0 texels" in str(w.message)
    ]
    assert len(gutter_warnings) == 1, (
        f"expected exactly one gutter warning, got {len(gutter_warnings)}: "
        f"{[str(w.message)[:60] for w in gutter_warnings]}"
    )


# ---------------------------------------------------------------------------
# 合否基準8: ガター規約 `g` が効いていること(P >= 3 の fixture)
# ---------------------------------------------------------------------------


def test_part_path_gutter_uses_the_scaled_iteration_count(
    tmp_path, capped_cylinder_mesh, make_texture, dilate8, match_faces_by_position
) -> None:
    """`rebake` が素の `padding_px` ではなく `g` でガターを膨らませ、部位が衝突しない。

    測り方(production の値を**出力から復元**する):
      1. 焼き上がりテクスチャの非背景領域 `filled` を取る。`bake._grow_gutter` は
         被覆から 8 近傍で 1 リングずつ広げるので、`filled` は必ず
         `dilate8(cov_all, k)` の形をしている。
      2. `k` を 0..padding_px+2 で走査し、`filled` と一致する `k` を求める。実測では
         **ちょうど 1 つ**に定まる(= production が実際に使った反復数)。
      3. それが `g = min(padding_px, floor(padding_px·texture_size/D))` と一致する
         こと、かつその `g` で異部位のガターが 1 テクセルも重ならないことを見る。

    **識別力(ミューテーションで落ちること)**: 実測 2026-08-06 で
    `capped_cylinder` @ `texture_size=128` / `padding_px=8` は atlas 200x159
    (D=200)→ `g=5`。`g` を素の `padding_px=8` へ戻すと (a) 復元される反復数が 8 に
    なって期待値 5 と食い違い、(b) 異部位ガターが **361 テクセル**重なる。最後の
    assert はその「素の padding_px なら衝突する」ことを**このテスト内で**確認する
    ので、ゲートが空虚でないことが in-suite で示される。
    """
    mesh = _with_texture(capped_cylinder_mesh, make_texture)
    source = tmp_path / "in.glb"
    baked = tmp_path / "out.glb"
    save_mesh(mesh, source)

    in_mesh = load_mesh(source)
    labels = DihedralSegmenter().segment(in_mesh)
    n_parts = int(labels.max()) + 1
    assert n_parts >= 3, "the gutter gate needs P >= 3 to observe cross-part collisions"

    # 期待値の導出はテスト側で独立に行う(production の `_gutter_iterations` を
    # import しない — それでは式が自分自身に一致するだけになる)。
    _new_mesh, _face_map, dims = _part_unwrap_and_pack(
        in_mesh, labels, resolution=_GUTTER_RES, padding_px=_GUTTER_PADDING
    )
    atlas_edge = max(dims.width, dims.height)
    expected_g = min(_GUTTER_PADDING, _GUTTER_PADDING * _GUTTER_RES // atlas_edge)
    assert 1 <= expected_g < _GUTTER_PADDING, (
        f"this fixture/parameter pair no longer discriminates: g={expected_g} vs "
        f"padding_px={_GUTTER_PADDING} (atlas {dims.width}x{dims.height})"
    )

    rebake(
        source,
        baked,
        texture_size=_GUTTER_RES,
        padding_px=_GUTTER_PADDING,
        granularity="part",
    )
    out_mesh = load_mesh(baked)
    out_face_map, _ = match_faces_by_position(out_mesh, in_mesh)
    masks = _part_coverage_masks(
        out_mesh.uv, out_mesh.faces, labels[out_face_map], n_parts, _GUTTER_RES
    )
    coverage = masks.any(axis=0)
    # 背景は 0 で、テクスチャは非零なので「非背景」がガター込みの充填領域になる。
    filled = out_mesh.maps["basecolor"].max(axis=2) > 1.0 / 512.0

    matching = [
        k
        for k in range(_GUTTER_PADDING + 3)
        if np.array_equal(dilate8(coverage, k), filled)
    ]
    assert matching == [expected_g], (
        f"the baked fill region matches dilate8(coverage, k) for k in {matching}, "
        f"expected exactly [{expected_g}] (atlas {dims.width}x{dims.height}, "
        f"D={atlas_edge}, padding_px={_GUTTER_PADDING})"
    )
    assert _cross_part_overlap(masks, dilate8, expected_g) == 0, (
        "different parts' gutters overlap even at the scaled iteration count"
    )
    # 空虚でないことの in-suite 証明: 素の padding_px なら必ず衝突する。
    raw_overlap = _cross_part_overlap(masks, dilate8, _GUTTER_PADDING)
    assert raw_overlap > 0, (
        "using the raw padding_px would not collide here, so this gate cannot "
        "detect the mutation it exists to catch"
    )


def test_gutter_iteration_formula_boundaries() -> None:
    """`_gutter_iterations` の境界(単位): `D <= ts` で不変、`pad*ts < D` で 0。"""
    from atlasmith import _gutter_iterations

    assert _gutter_iterations(8, 1024, 512) == 8  # D < ts: Phase 1 と同じ関係
    assert _gutter_iterations(8, 1024, 1024) == 8  # D == ts: 境界
    assert _gutter_iterations(8, 128, 200) == 5  # 実測ケース
    assert _gutter_iterations(1, 128, 141) == 0  # 実測ケース(ガター消失)
    # `g == 0` は 1 に床上げしない(裁定1)。
    assert _gutter_iterations(2, 64, 1000) == 0


# ---------------------------------------------------------------------------
# 合否基準9: テクスチャ無しメッシュの部位経路回帰
# ---------------------------------------------------------------------------


def test_part_path_without_textures_writes_geometry_and_uv_only(
    tmp_path, capped_cylinder_mesh
) -> None:
    """`maps` 空入力は幾何 + 新 UV だけを書き出す(Phase 1 と同挙動)。"""
    source = tmp_path / "in.glb"
    baked = tmp_path / "out.glb"
    save_mesh(capped_cylinder_mesh, source)

    rebake(
        source,
        baked,
        texture_size=_GUTTER_RES,
        padding_px=_GUTTER_PADDING,
        granularity="part",
    )

    out_mesh = load_mesh(baked)
    assert out_mesh.maps == {}
    assert out_mesh.uv is not None
    assert out_mesh.faces.shape == capped_cylinder_mesh.faces.shape
