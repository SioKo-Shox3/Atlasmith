"""io/bake 共通の合成 fixture(決定的・シード固定)。

メッシュ factory(cube/sphere/torus)と、テクスチャ factory(gradient/multisine/
aperiodic/face_id)を提供する。Phase 1(bake)のテストからも共用する前提
(test-design-step0-4.md 第1部)。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable

import numpy as np
import pytest
import trimesh

from atlasmith.segmentation.multiview import Camera, RenderedView
from atlasmith.types import MeshData

# Phase 1 (bake, quantize8=False) 用の許容誤差定数。Step 0-4 の io テストは
# quantize8=True の厳密比較(max|Δ|<=1e-6)しか使わないため未使用だが、
# テスト設計(第2部「許容誤差」)が conftest への記録を指示しているため残す。
NON_QUANTIZED_PIXEL_TOLERANCE = 0.5 / 255 + 1e-6  # ~= 2.0e-3


# ---------------------------------------------------------------------------
# メッシュ factory
# ---------------------------------------------------------------------------


def _build_cube_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """6 面 x 4 頂点の独立コーナーを持つ立方体(24頂点/12面)。

    `trimesh.creation.box()` は UV を持たないため不採用 — 面別に UV パッチを
    完全分離できることが V方向検証(V1/X1)に必須。UV は 3列x2行グリッドへ
    面ごとに割り当て、パッチ内側 10% インセット(バイリニア滲み防止)。
    """
    faces_def = [
        (np.array([0.5, -0.5, -0.5]), np.array([0, 1, 0]), np.array([0, 0, 1])),  # +X
        (np.array([-0.5, -0.5, 0.5]), np.array([0, 1, 0]), np.array([0, 0, -1])),  # -X
        (np.array([-0.5, 0.5, -0.5]), np.array([1, 0, 0]), np.array([0, 0, 1])),  # +Y
        (np.array([-0.5, -0.5, 0.5]), np.array([1, 0, 0]), np.array([0, 0, -1])),  # -Y
        (np.array([-0.5, -0.5, 0.5]), np.array([1, 0, 0]), np.array([0, 1, 0])),  # +Z
        (np.array([-0.5, 0.5, -0.5]), np.array([1, 0, 0]), np.array([0, -1, 0])),  # -Z
    ]
    verts: list[np.ndarray] = []
    faces: list[list[int]] = []
    uv: list[list[float]] = []
    for fi, (origin, edge_u, edge_v) in enumerate(faces_def):
        base = len(verts)
        verts.extend(
            [origin, origin + edge_u, origin + edge_u + edge_v, origin + edge_v]
        )
        row, col = fi // 3, fi % 3
        u0, u1 = col / 3 + 0.1 / 3, (col + 1) / 3 - 0.1 / 3
        v0, v1 = row / 2 + 0.1 / 2, (row + 1) / 2 - 0.1 / 2
        # V反転で行(row)が入れ替わる非対称レイアウト: row0とrow1は別の面集合。
        uv.extend([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
    return (
        np.array(verts, dtype=np.float64),
        np.array(faces, dtype=np.int64),
        np.array(uv, dtype=np.float32),
    )


@pytest.fixture
def cube_mesh() -> MeshData:
    vertices, faces, uv = _build_cube_geometry()
    return MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps={},
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )


def _analytic_sphere_uv(vertices: np.ndarray) -> np.ndarray:
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    r = np.linalg.norm(vertices, axis=1)
    u = 0.5 + np.arctan2(y, x) / (2.0 * np.pi)
    v = 0.5 - np.arcsin(np.clip(z / r, -1.0, 1.0)) / np.pi
    return np.stack([u, v], axis=1).astype(np.float32)


@pytest.fixture
def sphere_mesh() -> MeshData:
    # u シームの頂点複製はしない(格納テスト用途。Phase 1 オラクルではシーム帯
    # 分類で除外される前提 — test-design-step0-4.md 第1部)。
    ico = trimesh.creation.icosphere(subdivisions=2)
    vertices = np.asarray(ico.vertices, dtype=np.float64)
    faces = np.asarray(ico.faces, dtype=np.int64)
    uv = _analytic_sphere_uv(vertices)
    return MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps={},
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )


def _analytic_torus_uv(vertices: np.ndarray, major_radius: float) -> np.ndarray:
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    theta = np.arctan2(y, x)  # 主角度(トーラス中心軸まわり)
    rho = np.sqrt(x**2 + y**2) - major_radius
    phi = np.arctan2(z, rho)  # 副角度(チューブ断面まわり)
    u = (theta / (2.0 * np.pi)) % 1.0
    v = (phi / (2.0 * np.pi)) % 1.0
    return np.stack([u, v], axis=1).astype(np.float32)


@pytest.fixture
def torus_mesh() -> MeshData:
    major_radius = 1.0
    torus = trimesh.creation.torus(major_radius=major_radius, minor_radius=0.3)
    vertices = np.asarray(torus.vertices, dtype=np.float64)
    faces = np.asarray(torus.faces, dtype=np.int64)
    uv = _analytic_torus_uv(vertices, major_radius)
    return MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps={},
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# テクスチャ factory
# ---------------------------------------------------------------------------


def _gradient_texture(height: int, width: int, channels: int) -> np.ndarray:
    cols = (np.arange(width) + 0.5) / width
    rows = (np.arange(height) + 0.5) / height
    img = np.zeros((height, width, channels), dtype=np.float64)
    if channels >= 1:
        img[:, :, 0] = cols[np.newaxis, :]
    if channels >= 2:
        img[:, :, 1] = rows[:, np.newaxis]
    if channels >= 3:
        img[:, :, 2] = 0.25
    for ch in range(3, channels):
        # 例: alpha。1.0固定にすると「破棄して既定値で埋める」バグを検出できない
        # ため、判別可能な値にする。
        img[:, :, ch] = 0.6
    return img


def _sine_sum_texture(
    height: int, width: int, channels: int, seed: int, freqs: tuple[float, ...]
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.linspace(0.0, 1.0, height, endpoint=False)
    cols = np.linspace(0.0, 1.0, width, endpoint=False)
    v_grid, u_grid = np.meshgrid(rows, cols, indexing="ij")
    img = np.zeros((height, width, channels), dtype=np.float64)
    for ch in range(channels):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal = np.zeros((height, width), dtype=np.float64)
        for freq in freqs:
            signal += np.sin(2.0 * np.pi * freq * u_grid + phase)
            signal += np.sin(2.0 * np.pi * freq * v_grid + phase)
        c_min, c_max = signal.min(), signal.max()
        img[:, :, ch] = (signal - c_min) / (c_max - c_min) if c_max > c_min else 0.5
    return img


def _quantize8(arr: np.ndarray) -> np.ndarray:
    return np.round(arr * 255.0) / 255.0


def _build_texture(
    kind: str,
    size: tuple[int, int] = (64, 64),
    channels: int = 3,
    *,
    seed: int = 0,
    quantize8: bool = False,
) -> np.ndarray:
    """`float32 (H, W, C) [0, 1]` の合成テクスチャを返す。"""
    height, width = size
    if kind == "gradient":
        # V反転・チャンネル入替・転置を1枚で検出可能な低周波勾配。
        img = _gradient_texture(height, width, channels)
    elif kind == "multisine":
        # Phase 1 主ゲート用の平滑テクスチャ(整数サイクルのサイン和)。
        img = _sine_sum_texture(height, width, channels, seed, freqs=(3.0, 7.0, 13.0))
    elif kind == "aperiodic":
        # 無理数比周波数 — シフトが偶然一致(エイリアス)しない負の対照用。
        img = _sine_sum_texture(
            height, width, channels, seed, freqs=(1.0, np.sqrt(2.0), np.sqrt(5.0))
        )
    else:
        raise ValueError(f"Unknown texture kind: {kind!r}")
    if quantize8:
        img = _quantize8(img)
    return img.astype(np.float32)


@pytest.fixture
def make_texture() -> Callable[..., np.ndarray]:
    return _build_texture


def _barycentric_grid(
    px: np.ndarray,
    py: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0 = uv0
    x1, y1 = uv1
    x2, y2 = uv2
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def _build_face_id_texture(
    mesh: MeshData, size: tuple[int, int] = (64, 64)
) -> np.ndarray:
    """各 UV 三角形を面色 (ch0=(i+0.5)/M, ch1=1-ch0, ch2=0.5) で塗る素朴ラスタライザ。

    Phase 1 の独立カバレッジ生成(B3)の前身。テクセル中心 UV は横断規約の
    変換式 `((c+0.5)/W, (r+0.5)/H)` に従う。ピクセル格子はまとめて numpy 化し、
    面ループのみ回す(素朴=非高速化アルゴリズムの意で、実装は許容範囲でベクトル化)。
    """
    height, width = size
    if mesh.uv is None:
        raise ValueError("make_face_id_texture requires a mesh with uv coordinates")
    cols = (np.arange(width) + 0.5) / width
    rows = (np.arange(height) + 0.5) / height
    v_grid, u_grid = np.meshgrid(rows, cols, indexing="ij")
    img = np.zeros((height, width, 3), dtype=np.float32)
    uv = np.asarray(mesh.uv, dtype=np.float64)
    n_faces = len(mesh.faces)
    eps = -1e-9
    for i, face in enumerate(mesh.faces):
        w0, w1, w2 = _barycentric_grid(
            u_grid, v_grid, uv[face[0]], uv[face[1]], uv[face[2]]
        )
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        ch0 = (i + 0.5) / n_faces
        img[inside] = np.array([ch0, 1.0 - ch0, 0.5], dtype=np.float32)
    return img


@pytest.fixture
def make_face_id_texture() -> Callable[..., np.ndarray]:
    return _build_face_id_texture


# ---------------------------------------------------------------------------
# Phase 1 独立実装(裁定9 + Step 1-1 二重レビュー追補)
#
# production(src/atlasmith/bake/transfer.py)とは *独立に設計された* オラクル/評価道具。
# レビュー指摘(Opus: 逐語コピーで独立性が形式的 / Codex: CW 面で corner 1/2 を戻し
# 忘れる実バグ)を受け、production の「meshgrid ベクトル化 + edge 関数 + top-left
# tie-break + 巻き順スワップ」という構造をあえて共有しない別実装にしている:
#   - 内外判定: 3 つの部分三角形の符号付き 2 倍面積(sub-triangle area)方式。
#     production の edge 関数とは変数の持ち方が異なる。
#   - 重心座標: 常に入力 faces_uv の *元の corner 順* (0,1,2) で算出する。頂点の内部
#     並べ替え(巻き順スワップ)を一切しないので、CW 面でも corner 1/2 が入れ替わらない
#     (旧テスト実装が持っていた欠陥を「戻し処理」ではなく構造で排除している)。
#   - tie-break: 共有辺は「先に走査した = face_id が小さい面が所有」する規則(production
#     の top-left とは別規則だが決定的)。face_id/bary は最小 face_id の面のものになる。
#   - 走査: 面ごとの bbox を素朴に per-texel 走査(遅くてよい / test-design 準拠)。
#   - 定数も独自に選定(production の 1e-9 / 1e-12 を流用しない)。
# ---------------------------------------------------------------------------

# 正規化済み重心座標での辺の許容(production の _BARY_EPS=1e-9 とは別値を独自選定)。
_INSIDE_EPS = 1e-7
# 縮退三角形(テクセル空間の符号付き 2 倍面積がほぼ 0)の除外しきい値(独自選定)。
_MIN_DOUBLE_AREA = 1e-10


def _double_area(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    """三角形 (a, b, c) の符号付き 2 倍面積(2D 外積の z 成分)。CCW で正・CW で負。"""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _rasterize_coverage(
    faces_uv: np.ndarray, uv: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """独立ラスタライザ(sub-triangle area・元 corner 順・最小 face_id tie-break)。

    戻り値 (cov (H,W) bool, face_id (H,W) int64 無被覆-1, bary (H,W,3) float64)。
    テクセル中心 UV = ((c+0.5)/W, (r+0.5)/H)。texel 空間で中心は (x=c, y=r)。
    bary[r,c] は face_id の面の入力 corner 順 (faces_uv[i,0..2]) に対する重心。
    """
    height, width = size
    cov = np.zeros((height, width), dtype=bool)
    face_id = np.full((height, width), -1, dtype=np.int64)
    bary = np.zeros((height, width, 3), dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    for i, face in enumerate(faces_uv):
        # 入力 corner 順のまま texel 空間へ置く。巻き順スワップをしないので、あとで
        # corner を「元に戻す」処理が原理的に不要 — これが CW corner バグの構造的排除。
        ax = float(uv[face[0], 0]) * width - 0.5
        ay = float(uv[face[0], 1]) * height - 0.5
        bx = float(uv[face[1], 0]) * width - 0.5
        by = float(uv[face[1], 1]) * height - 0.5
        cx = float(uv[face[2], 0]) * width - 0.5
        cy = float(uv[face[2], 1]) * height - 0.5
        area2 = _double_area(ax, ay, bx, by, cx, cy)  # CCW で正・CW で負
        if abs(area2) < _MIN_DOUBLE_AREA:
            continue  # 縮退面は転写に寄与しない。
        # bbox を [0,W-1]x[0,H-1] に clamp(画像外・反対端への回り込みを遮断; C9)。
        cmin = max(0, int(np.floor(min(ax, bx, cx))))
        cmax = min(width - 1, int(np.ceil(max(ax, bx, cx))))
        rmin = max(0, int(np.floor(min(ay, by, cy))))
        rmax = min(height - 1, int(np.ceil(max(ay, by, cy))))
        for r in range(rmin, rmax + 1):
            py = float(r)  # テクセル中心の texel 空間 y。
            for c in range(cmin, cmax + 1):
                if cov[r, c]:
                    continue  # tie-break: 既に小さい face_id が所有 → 上書きしない。
                px = float(c)  # テクセル中心の texel 空間 x。
                # 部分三角形の符号付き面積 / 全体面積 = 正規化重心(元 corner 順)。
                # w0=対 corner0・w1=対 corner1・w2=対 corner2。巻き順に依らず corner に
                # 正しく対応する(CW でも入れ替わらない)。
                w0 = _double_area(px, py, bx, by, cx, cy) / area2
                w1 = _double_area(ax, ay, px, py, cx, cy) / area2
                w2 = _double_area(ax, ay, bx, by, px, py) / area2
                if w0 >= -_INSIDE_EPS and w1 >= -_INSIDE_EPS and w2 >= -_INSIDE_EPS:
                    cov[r, c] = True
                    face_id[r, c] = i
                    bary[r, c, 0] = w0
                    bary[r, c, 1] = w1
                    bary[r, c, 2] = w2
    return cov, face_id, bary


def _shift_bool(mask: np.ndarray, dr: int, dc: int, fill: bool) -> np.ndarray:
    """近傍 `mask[r+dr, c+dc]` を (r, c) へ集める非循環シフト(np.roll 不使用)。"""
    out = np.full_like(mask, fill)
    height, width = mask.shape[:2]
    r_src0, r_src1 = max(0, dr), min(height, height + dr)
    c_src0, c_src1 = max(0, dc), min(width, width + dc)
    r_dst0, r_dst1 = max(0, -dr), min(height, height - dr)
    c_dst0, c_dst1 = max(0, -dc), min(width, width - dc)
    if r_src0 < r_src1 and c_src0 < c_src1:
        out[r_dst0:r_dst1, c_dst0:c_dst1] = mask[r_src0:r_src1, c_src0:c_src1]
    return out


def _dilate8(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """8 近傍・非循環・border=False の二値膨張(np.roll 不使用)。"""
    out = mask.copy()
    for _ in range(iters):
        acc = out.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                acc |= _shift_bool(out, dr, dc, False)
        out = acc
    return out


def _erode8(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """8 近傍・非循環・border=False の二値収縮(境界外は False 扱いで削れる)。"""
    out = mask.copy()
    for _ in range(iters):
        acc = out.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                acc &= _shift_bool(out, dr, dc, False)
        out = acc
    return out


@pytest.fixture
def rasterize_coverage() -> Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return _rasterize_coverage


@pytest.fixture
def erode8() -> Callable[..., np.ndarray]:
    return _erode8


@pytest.fixture
def dilate8() -> Callable[..., np.ndarray]:
    return _dilate8


# ---------------------------------------------------------------------------
# Phase 1 Step 1-2 独立オラクル道具(裁定9、production 非 import)
#
# 一方向オラクル(test_bake_oracle.py)が使う「独立サンプラ」と「一様重心サンプル」。
# サンプラは production の bake._bilinear_sample と *構造を変えて* 実装する(Step 1-1
# レビューで「逐語コピーは独立性が形式的」と指摘されたため):
#   - production は top/bottom の入れ子 lerp(先に水平、次に垂直)で合成する。
#   - こちらは 4 隅の重み (1-tx)(1-ty) 等を明示計算し、4 隅画素を1本に束ねた重み和
#     (einsum)で合成する。中間変数の持ち方(gx/gy/ix/iy/tx/ty)も別に取る。
#   - 座標規約 x=u*W-0.5, y=v*H-0.5 と clamp 境界は品質契約なので共有する(実装構造
#     のみ独立にし、数学的定義は同一 — でなければオラクルとして無意味)。
# ---------------------------------------------------------------------------


def _sample_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """独立バイリニアサンプラ(4 隅重み和・clamp 境界)。

    `image` は `(H, W, C)` か `(H, W)`。`u, v` は同 shape の1次元配列 `(K,)`。
    戻り値は `(K, C)` または(2D 入力時)`(K,)`。オラクルの基準値・被験値の両方に
    使う(裁定9 — production の sampler を import しない)。
    """
    img = np.asarray(image, dtype=np.float64)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    gx = np.asarray(u, dtype=np.float64) * width - 0.5
    gy = np.asarray(v, dtype=np.float64) * height - 0.5
    ix = np.floor(gx)
    iy = np.floor(gy)
    tx = gx - ix  # [0, 1) 横方向の小数部
    ty = gy - iy  # [0, 1) 縦方向の小数部
    col0 = np.clip(ix.astype(np.int64), 0, width - 1)
    col1 = np.clip(ix.astype(np.int64) + 1, 0, width - 1)
    row0 = np.clip(iy.astype(np.int64), 0, height - 1)
    row1 = np.clip(iy.astype(np.int64) + 1, 0, height - 1)
    # 4 隅の重み(和 = 1)を明示的に組む: (行0,列0)/(行0,列1)/(行1,列0)/(行1,列1)。
    weights = np.stack(
        [
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        ],
        axis=-1,
    )  # (K, 4)
    corners = np.stack(
        [img[row0, col0], img[row0, col1], img[row1, col0], img[row1, col1]],
        axis=1,
    )  # (K, 4, C)
    out = np.einsum("kn,knc->kc", weights, corners)
    return out[:, 0] if squeeze else out


def _uniform_barycentric(k: int, seed: int) -> np.ndarray:
    """三角形一様サンプルの重心座標を sqrt 法で K 点生成する(seed 固定・決定的)。

    r1, r2 ~ U(0,1) に対し s = sqrt(r1) とおくと (1-s, s(1-r2), s r2) が三角形上の
    面積測度で一様な重心座標になる。全面で同じ K 点集合を使い回す前提(seed 固定)。
    戻り値 `(K, 3) float64`(各行が (w0, w1, w2)、和 = 1)。
    """
    rng = np.random.default_rng(seed)
    r1 = rng.random(k)
    r2 = rng.random(k)
    s = np.sqrt(r1)
    return np.stack([1.0 - s, s * (1.0 - r2), s * r2], axis=1)


@pytest.fixture
def bilinear_sample() -> Callable[..., np.ndarray]:
    return _sample_bilinear


@pytest.fixture
def face_barycentric_samples() -> Callable[..., np.ndarray]:
    return _uniform_barycentric


# ---------------------------------------------------------------------------
# Phase 2 Step 2-2 部位分割 fixture
#
# 既存の fixture / ヘルパ / 定数は 1 行も変更していない(追加のみ)。
#
# `cube_mesh` の weld 隣接構造(テストが明示的に assert する数値の導出根拠):
#   - 位置weld 後の相異なる頂点は **8**(24 頂点は 6 面 x 4 corner の複製で、
#     立方体の頂点1個につき 3 面ぶんのコピーがある: 8 x 3 = 24)。
#   - 辺数 **E == 18**: 閉じた三角形メッシュのオイラー標数 V - E + F = 2 に
#     V=8, F=12 を代入すると E = 18。内訳は立方体の稜線 12 本 + 各 facet を
#     2 三角形に割る対角線 6 本。
#   - `two_cubes_mesh` の 2 つの立方体は x 方向に 3.0 離れており位置を共有しない
#     ため、**立方体間の辺は 0 本**(全 36 本 = 18 + 18 がそれぞれの立方体に閉じる)。
# ---------------------------------------------------------------------------


@pytest.fixture
def two_cubes_mesh() -> MeshData:
    """互いに離れた 2 個の立方体(48 頂点 / 24 面)を 1 メッシュに連結したもの。

    2 個目は `(3.0, 0.0, 0.0)` 平行移動。UV は 1 個目を u 方向 [0, 0.5]、2 個目を
    [0.5, 1.0] へ収めた解析 UV(`_build_cube_geometry` の面別パッチを u だけ
    半分に縮めて左右へ振り分ける)。

    **用途は「連結成分の分離」ゲート専用**(計画v4 §5 Step 2-2 / BL-1)。非連結
    なので多視点融合ゲートには使わない。
    """
    vertices, faces, uv = _build_cube_geometry()
    n_vertices = len(vertices)
    all_vertices = np.concatenate(
        [vertices, vertices + np.array([3.0, 0.0, 0.0])], axis=0
    )
    all_faces = np.concatenate([faces, faces + n_vertices], axis=0)
    uv_left = uv.copy()
    uv_left[:, 0] = uv[:, 0] * 0.5
    uv_right = uv.copy()
    uv_right[:, 0] = 0.5 + uv[:, 0] * 0.5
    all_uv = np.concatenate([uv_left, uv_right], axis=0).astype(np.float32)
    return MeshData(
        vertices=all_vertices,
        faces=all_faces,
        uv=all_uv,
        maps={},
        source_vertex=np.arange(len(all_vertices), dtype=np.int64),
    )


# UV パッチの内側マージン(パッチ寸法に対する比)。既存 `_build_cube_geometry`
# (:34)と同じ 10%。領域どうしが辺を共有していると、Step 2-6/2-7 の焼き直し
# オラクルでその直線上だけバイリニア滲みが起き、原因の切り分けが難しい形で PSNR が
# 落ちる。fixture 側で最初から領域を離しておく。
_UV_PATCH_INSET = 0.1


def _place_in_patch(
    s: np.ndarray, t: np.ndarray, rect: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """`[0,1]^2` の (s, t) を矩形 `rect=(u0, u1, v0, v1)` の内側 10% へ写す。"""
    u0, u1, v0, v1 = rect
    margin_u = (u1 - u0) * _UV_PATCH_INSET
    margin_v = (v1 - v0) * _UV_PATCH_INSET
    return (
        u0 + margin_u + s * (u1 - u0 - 2.0 * margin_u),
        v0 + margin_v + t * (v1 - v0 - 2.0 * margin_v),
    )


# 円筒 fixture の UV 領域(インセット前の外枠)。3 領域は辺を共有しない。
_CYLINDER_UV_RECTS = {
    "side": (0.0, 1.0, 0.0, 0.5),
    "top": (0.0, 0.5, 0.5, 1.0),
    "bottom": (0.5, 1.0, 0.5, 1.0),
}


def _build_capped_cylinder_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`trimesh.creation.cylinder(radius=0.5, height=2.0, sections=32)` + 解析 UV。

    UV レイアウト(3 領域。各領域は `_UV_PATCH_INSET` ぶん内側へ寄せてあるので、
    領域どうしは辺も点も共有しない):
      側面        外枠 u [0, 1]   / v [0, 0.5]  — u = 展開角 / 2pi、v = 上端 → 下端
      +Z キャップ 外枠 u [0, 0.5] / v [0.5, 1]  — 円盤の平面 UV
      -Z キャップ 外枠 u [0.5, 1] / v [0.5, 1]  — 同上(x を反転 = 下から見た向き)

    **面ごとに頂点をアンロールする(3M 頂点)WHY**: 側面の円筒展開とキャップの
    円盤 UV は rim 頂点に別々の UV を要求し、頂点あたり 1 組の UV では両立
    しない(円筒展開の theta=0/2pi シームも同様)。glTF が UV シームで頂点を
    複製するのと同じ事情なので、fixture 側でも複製で表現する。位置は trimesh の
    出力行をそのままコピーするだけなので、位置weld は元の 66 頂点へビット厳密に
    畳み戻る — weld を前提とした隣接構築(`segmentation/adjacency.py`)の実地の
    対照でもある。
    """
    radius, height, sections = 0.5, 2.0, 32
    cylinder = trimesh.creation.cylinder(
        radius=radius, height=height, sections=sections
    )
    # trimesh 側の三角形化が変わると 66 / 384 / 128 / 192 を assert する 4 本の
    # テストが同時に意味不明な数値で落ちるので、ここで位相を1行に固定して切り分ける
    # (`pyproject.toml` の `trimesh>=4.12.2` に上限が無いため)。
    if len(cylinder.vertices) != 66 or len(cylinder.faces) != 128:
        raise ValueError(
            "capped cylinder fixture assumes trimesh emits 66 vertices / 128 faces "
            f"for sections=32, got {len(cylinder.vertices)} / {len(cylinder.faces)}"
        )
    base_vertices = np.asarray(cylinder.vertices, dtype=np.float64)
    base_faces = np.asarray(cylinder.faces, dtype=np.int64)
    corners = base_vertices[base_faces]  # (M, 3, 3)

    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    axis_dot = normals[:, 2]
    is_top = axis_dot >= 0.5
    is_bottom = axis_dot <= -0.5
    is_side = ~(is_top | is_bottom)
    if int(is_top.sum()) == 0 or int(is_bottom.sum()) == 0 or int(is_side.sum()) == 0:
        raise ValueError(
            "capped cylinder fixture expects top/bottom/side faces, got "
            f"{int(is_top.sum())}/{int(is_bottom.sum())}/{int(is_side.sum())}"
        )

    uv = np.zeros((len(base_faces), 3, 2), dtype=np.float64)
    angle = np.arctan2(corners[..., 1], corners[..., 0]) / (2.0 * np.pi) % 1.0
    # シーム(theta=0/2pi)を跨ぐ面は、小さい側に 1 周ぶん足して u を単調にする。
    # 側面の 1 三角形が張る角度は 1/32 周なので、0.5 周のギャップはシームでしか
    # 生じない。結果として展開角の正規化値は [0, 1] に収まる。
    seam_shift = np.where(angle < angle.max(axis=1, keepdims=True) - 0.5, 1.0, 0.0)
    # 各領域とも [0,1]^2 の正規化座標を作ってから `_place_in_patch` で外枠へ写す。
    side_s = (angle + seam_shift)[is_side]
    side_t = 0.5 - corners[is_side][..., 2] / height  # z=+1 -> 0(上端)/ z=-1 -> 1
    uv[is_side, :, 0], uv[is_side, :, 1] = _place_in_patch(
        side_s, side_t, _CYLINDER_UV_RECTS["side"]
    )
    # 半径 0.5 の円盤 → [0,1]^2 の正規化座標(中心が 0.5, 0.5)。
    uv[is_top, :, 0], uv[is_top, :, 1] = _place_in_patch(
        corners[is_top][..., 0] + 0.5,
        corners[is_top][..., 1] + 0.5,
        _CYLINDER_UV_RECTS["top"],
    )
    uv[is_bottom, :, 0], uv[is_bottom, :, 1] = _place_in_patch(
        0.5 - corners[is_bottom][..., 0],
        corners[is_bottom][..., 1] + 0.5,
        _CYLINDER_UV_RECTS["bottom"],
    )

    return (
        corners.reshape(-1, 3),
        np.arange(len(base_faces) * 3, dtype=np.int64).reshape(-1, 3),
        uv.reshape(-1, 2).astype(np.float32),
    )


@pytest.fixture
def capped_cylinder_mesh() -> MeshData:
    """両端に平らなキャップを持つ円筒(128 面 / 位置weld 後 66 頂点)。

    側面 64 面・+Z キャップ 32 面・-Z キャップ 32 面。側面とキャップの二面角は
    90 度なので、`angle_deg=60` では 3 部位に割れることが期待値
    (計画v4 §5 Step 2-2)。
    """
    vertices, faces, uv = _build_capped_cylinder_geometry()
    return MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps={},
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )


@pytest.fixture
def nonmanifold_mesh() -> MeshData:
    """1 本の辺を **3 面** が共有する非多様体メッシュ(負の対照)。

    辺 (v0, v1) を面 0/1/2 が共有する。面 0 と面 1 は同一平面・同一法線なので、
    もし隣接と見なされれば `angle_deg=180` では必ず結合する — 「非多様体辺は
    必ずカット」という規約が効いていなければ落ちるテストになる。他の辺はどれも
    1 面しか持たないので、隣接は 1 組も生じないのが期待値。
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0: 共有辺の端点
            [1.0, 0.0, 0.0],  # 1: 共有辺の端点
            [0.0, 1.0, 0.0],  # 2: 面 0 の第 3 頂点(z=0 平面)
            [0.0, -1.0, 0.0],  # 3: 面 1 の第 3 頂点(z=0 平面・面 0 と同一法線)
            [0.0, 0.0, 1.0],  # 4: 面 2 の第 3 頂点(z=0 平面から立ち上がる)
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    return MeshData(
        vertices=vertices,
        faces=faces,
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )


@pytest.fixture
def zero_area_mesh() -> MeshData:
    """零面積(3 点が共線)の面を 1 枚含むメッシュ(負の対照)。

    面 0 (v0,v1,v2) と面 1 (v0,v2,v3) は z=0 平面の正方形を 2 分割したもので、
    辺 (v0,v2) を共有し二面角 0 度。面 2 (v0,v1,v4) は v0/v1/v4 が x 軸上で共線
    なので零面積で、辺 (v0,v1) を面 0 と共有する(= 多様体隣接としては成立する)。
    零面積面のカット規約が無いと法線 0 ベクトルの内積 0 が 90 度と解釈され、
    `angle_deg=180` では面 2 まで結合してしまう — それを弾くための対照。
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [2.0, 0.0, 0.0],  # 4: v0, v1 と共線
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]], dtype=np.int64)
    return MeshData(
        vertices=vertices,
        faces=faces,
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Phase 2 Step 2-1: `gl` / `ml` マーカーの能力検出 skip(計画v4 §2.7)
#
# 既存の fixture / ヘルパ / 定数は 1 行も変更していない(以下はすべて追加)。
#
# WHY 「import 可能性」ではなく能力検出か: extras を入れた CI ランナーでは moderngl が
# import **できてしまう**が GL コンテキストは無い。`pytest.importorskip("moderngl")` は
# この状況を skip にできず、テストは `create_context` で fail する。torch も同様に、
# import できても CUDA GPU が見えなければ ML ゲートは意味を成さない。よって
# 「実際に作れるか / 実際に CUDA が見えるか」を試して判定する。
#
# WHY 検出を遅延させるか: 検出は moderngl / torch / sam2 を **実際に import する**ので、
# 呼んだ時点でセッションの `sys.modules` にそれらが常駐する。該当マーカーの item が
# 1 つも収集されていなければ 1 度も import しない(`-m "not gl and not ml"` = CI 相当の
# 実行では走らない)。**この副作用があるため import 隔離ゲート
# (`tests/test_import_isolation.py`)は同一プロセスではなく subprocess で検査する**
# — 計画 §0-A 条件1。
# ---------------------------------------------------------------------------

# 能力検出の対象マーカー。`pyproject.toml` の `markers` と同期させること
# (`--strict-markers` 下では未登録マーカーは collection エラーになる)。
_CAPABILITY_MARKERS = ("gl", "ml")


def _gl_available() -> bool:
    """standalone な OpenGL コンテキストを実際に生成できるかを試す。

    成功したら即 `release()` して True を返す。moderngl 未導入・ドライバ不在・
    EGL/GLX の初期化失敗などはすべて False。例外型はプラットフォームとドライバで
    多様(`ImportError` / `moderngl.Error` / `OSError` 等)なので `Exception` で受ける。
    """
    try:
        import moderngl

        ctx = moderngl.create_context(standalone=True)
    except Exception:
        return False
    try:
        ctx.release()
    except Exception:
        # 解放に失敗しても「コンテキストを作れた」という判定自体は変えない。
        pass
    return True


def _ml_available() -> bool:
    """torch と sam2 が import でき、かつ CUDA GPU が見えるかを判定する。

    CPU 版 torch(Windows の既定 wheel)では `torch.cuda.is_available()` が False に
    なるため、ML ゲートは skip される — これは想定内(計画v4 §3)。
    """
    try:
        import sam2  # noqa: F401
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


_CAPABILITY_DETECTORS: dict[str, Callable[[], bool]] = {
    "gl": _gl_available,
    "ml": _ml_available,
}

_CAPABILITY_SKIP_REASONS = {
    "gl": (
        "no usable OpenGL context: moderngl.create_context(standalone=True) failed "
        "(needs `uv sync --extra ml` and a machine with a working GPU/GL driver)"
    ),
    "ml": (
        "torch/sam2 not importable or no CUDA GPU visible "
        "(needs `uv sync --extra ml` plus a CUDA build of torch)"
    ),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """`gl` / `ml` マーカー付きの item を、能力が無い環境では skip に落とす。

    検出は **該当マーカーの item が実際に収集されているときだけ**、マーカーごとに
    高々 1 回だけ走る(結果は `detected` にキャッシュ)。
    """
    detected: dict[str, bool] = {}
    for item in items:
        for marker_name in _CAPABILITY_MARKERS:
            if item.get_closest_marker(marker_name) is None:
                continue
            if marker_name not in detected:
                detected[marker_name] = _CAPABILITY_DETECTORS[marker_name]()
            if not detected[marker_name]:
                item.add_marker(
                    pytest.mark.skip(reason=_CAPABILITY_SKIP_REASONS[marker_name])
                )
                # gl/ml 両方が付いた item でも skip マーカーは 1 個で足りる。
                break


# ---------------------------------------------------------------------------
# Phase 2 Step 2-3: screen 空間 z バッファの独立オラクル(計画v4 §5 Step 2-3 / BL-3)
#
# 既存の fixture / ヘルパ / 定数は 1 行も変更していない(以下はすべて追加)。
#
# **WHY 既存 `_rasterize_coverage`(:289)を使わず新ヘルパを足すのか**: 既存実装の
# 勝者決定は **深度ではなく最小 face_id**(`if cov[r, c]: continue`、:324)。UV 空間では
# チャートが重ならないので Phase 1 ではそれで正しかったが、**screen 空間かつ背面
# カリング無効** では表面と裏面がほぼ全画素で競合し、GL は手前の面を・既存実装は最小
# index の面を選ぶ(cube を -X から見ると GL は面 2/3、既存実装は面 0/1)。したがって
# 面ID主ゲートのオラクルには使えない。**既存ヘルパは改変せず、別物として追加する。**
# ---------------------------------------------------------------------------

# screen 空間の重心座標での内外判定の許容(独自選定)。既存 `_INSIDE_EPS`(=1e-7)とは
# 別値 — 共有辺の 1 画素幅を両方の面で拾いすぎないよう小さくとる。
_SCREEN_INSIDE_EPS = 1e-9
# screen 空間の符号付き 2 倍面積がこれ未満の面は縮退として捨てる(独自選定)。
_SCREEN_MIN_DOUBLE_AREA = 1e-12
# 近傍シフトで画像外を埋める番兵。実在値(-1 = 背景、0.. = 面)と衝突しない。
_NO_FACE_SENTINEL = -2


def _project_to_ndc(vertices: np.ndarray, camera: object) -> np.ndarray:
    """頂点 `(N, 3)` を `camera` の NDC `(N, 3) float64` へ写す(clip → `/w`)。

    独立オラクル `_rasterize_screen_zbuffer` の入力を作るためのテスト側ヘルパ。
    production の `Camera.mvp` を使うのは意図的 — オラクルが独立しているべきなのは
    **ラスタライズの勝者決定**であって、「比較対象と同じカメラを見ている」ことは
    比較の前提そのものだから。

    Args:
        vertices: 頂点 `(N, 3)`。
        camera: `atlasmith.segmentation.multiview.Camera`(`mvp` を持つもの)。
    """
    verts = np.asarray(vertices, dtype=np.float64)
    homogeneous = np.concatenate(
        [verts, np.ones((verts.shape[0], 1), dtype=np.float64)], axis=1
    )
    clip = homogeneous @ camera.mvp.T  # type: ignore[attr-defined]
    return clip[:, :3] / clip[:, 3:4]


def _rasterize_screen_zbuffer(
    verts_ndc: np.ndarray, faces: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """screen 空間 z バッファの独立ラスタライザ(GL の面ID出力のオラクル)。

    Args:
        verts_ndc: NDC まで持っていった頂点 `(N, 3)`(clip → `/w`)。
            **前提: 全頂点が視錐台内**(near クリッピングは扱わない —
            `cameras.validate_frustum` が production 側で保証する)。
        faces: 三角形 `(M, 3)`。
        size: 正方形画像の一辺(画素)。レンダラの `image_size` と同じ。

    Returns:
        `(face_id (H, W) int64, ndc_z (H, W) float64)`。背景の `face_id` は -1 で、
        そのとき `ndc_z` は `+inf`。**row 0 = 画面上端**(GL の読み戻しを行反転した
        後の規約に合わせる)。

    深度は **screen 空間で線形補間**する(GL の深度補間と同じ)。属性の遠近補正は
    要らない — 比べるのは深度だけで、`ndc_z` は screen 空間で線形なので
    **透視投影でも正しい**。同深度の tie は最小 face index が勝つ(GL の
    `depth_func='<'` と面 index 順の描画に一致する)。
    """
    ndc = np.asarray(verts_ndc, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    n = int(size)
    face_id = np.full((n, n), -1, dtype=np.int64)
    depth = np.full((n, n), np.inf, dtype=np.float64)
    # 画素中心が (col + 0.5, row + 0.5) になる screen 座標。row は上端が 0 なので、
    # y は NDC を反転して作る(GL の viewport は下端原点 / こちらは読み戻し後の規約)。
    sx = (ndc[:, 0] + 1.0) * 0.5 * n
    sy = (1.0 - ndc[:, 1]) * 0.5 * n
    sz = ndc[:, 2]
    for i, face in enumerate(face_array):
        ia, ib, ic = int(face[0]), int(face[1]), int(face[2])
        ax, ay = sx[ia], sy[ia]
        bx, by = sx[ib], sy[ib]
        cx, cy = sx[ic], sy[ic]
        area2 = _double_area(ax, ay, bx, by, cx, cy)
        if abs(area2) < _SCREEN_MIN_DOUBLE_AREA:
            continue  # 縮退三角形は 1 画素も塗らない。
        col_lo = max(0, int(np.floor(min(ax, bx, cx) - 0.5)))
        col_hi = min(n - 1, int(np.ceil(max(ax, bx, cx) + 0.5)))
        row_lo = max(0, int(np.floor(min(ay, by, cy) - 0.5)))
        row_hi = min(n - 1, int(np.ceil(max(ay, by, cy) + 0.5)))
        if col_lo > col_hi or row_lo > row_hi:
            continue
        cols = np.arange(col_lo, col_hi + 1, dtype=np.float64) + 0.5
        rows = np.arange(row_lo, row_hi + 1, dtype=np.float64) + 0.5
        px, py = np.meshgrid(cols, rows)
        # 部分三角形の符号付き面積比 = 正規化重心座標(入力 corner 順のまま)。
        w0 = _double_area(px, py, bx, by, cx, cy) / area2
        w1 = _double_area(ax, ay, px, py, cx, cy) / area2
        w2 = _double_area(ax, ay, bx, by, px, py) / area2
        inside = (
            (w0 >= -_SCREEN_INSIDE_EPS)
            & (w1 >= -_SCREEN_INSIDE_EPS)
            & (w2 >= -_SCREEN_INSIDE_EPS)
        )
        if not inside.any():
            continue
        z = w0 * sz[ia] + w1 * sz[ib] + w2 * sz[ic]
        window = (slice(row_lo, row_hi + 1), slice(col_lo, col_hi + 1))
        # 厳密な `<` なので、同深度なら先に走査した(= index の小さい)面が残る。
        wins = inside & (z < depth[window])
        depth[window] = np.where(wins, z, depth[window])
        face_id[window] = np.where(wins, i, face_id[window])
    return face_id, depth


def _shift_face_id(arr: np.ndarray, dr: int, dc: int, fill: int) -> np.ndarray:
    """`arr[r+dr, c+dc]` を (r, c) へ集める非循環シフト(整数配列版)。

    既存の `_shift_bool`(:341)と同じシフト規約だが、あちらは名前も docstring も
    bool 前提なので **改変せずに値版を足す**(既存ヘルパ不可侵の制約)。
    """
    out = np.full_like(arr, fill)
    height, width = arr.shape[:2]
    r_src0, r_src1 = max(0, dr), min(height, height + dr)
    c_src0, c_src1 = max(0, dc), min(width, width + dc)
    r_dst0, r_dst1 = max(0, -dr), min(height, height - dr)
    c_dst0, c_dst1 = max(0, -dc), min(width, width - dc)
    if r_src0 < r_src1 and c_src0 < c_src1:
        out[r_dst0:r_dst1, c_dst0:c_dst1] = arr[r_src0:r_src1, c_src0:c_src1]
    return out


def _faceid_interior_mask(face_id: np.ndarray) -> np.ndarray:
    """面ID画像の「内部画素」= 前景かつ **8 近傍すべてが同じ face_id** の画素。

    画像境界の外は「別の面」扱いなので、外周 1 画素は必ず False になる。

    **WHY bool 収縮(`_erode8`)ではないのか**(BL-3(b)): 被覆 bool を収縮しても
    削れるのは **シルエット境界だけ** で、面と面の境界画素(前景どうしなので bool
    では区別が付かない)が残る。そこは GL の fill rule とオラクルの内外判定が正当に
    食い違いうる場所なので、厳密一致を課す領域から外さなければならない。
    """
    fid = np.asarray(face_id)
    interior = fid >= 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            interior &= _shift_face_id(fid, dr, dc, _NO_FACE_SENTINEL) == fid
    return interior


@pytest.fixture
def project_to_ndc() -> Callable[..., np.ndarray]:
    return _project_to_ndc


@pytest.fixture
def rasterize_screen_zbuffer() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    return _rasterize_screen_zbuffer


@pytest.fixture
def faceid_interior_mask() -> Callable[..., np.ndarray]:
    return _faceid_interior_mask


# ---------------------------------------------------------------------------
# Phase 2 Step 2-4: 融合ゲートの fixture とスタブ(計画v4 §5 Step 2-4 / §2.4.4)
#
# 既存の fixture / ヘルパ / 定数は 1 行も変更していない(以下はすべて追加)。
#
# ここに置くもの:
#   - `peanut_mesh` — 鋭い折れ目を持たない滑らかな 2 膨らみ(**numpy のみ**で生成)。
#     Step 2-4 では「幾何プライア単独では P==1 に退化する」前提の実証に使い、
#     Step 2-5 の主 ML ゲートでもそのまま使う(計画v4 §5 Step 2-4 の ★ 節)。
#   - `_StaticRenderer` / `_StaticMaskProposer` — 合成 `RenderedView` と固定マスクを
#     返すスタブ(§2.4.4)。これで `MultiViewSegmenter.segment()` 全体を GPU 無しで
#     厳密ゲートにかけられる。
#   - 帯状の合成面IDバッファ生成(`_build_block_view` / `_masks_from_face_groups`)。
# ---------------------------------------------------------------------------

# `peanut_mesh` の極座標グリッド解像度と半径のうねり(計画v4 §5 Step 2-4)。
_PEANUT_N_THETA = 48
_PEANUT_N_PHI = 64
_PEANUT_BULGE = 0.35


def _build_peanut_geometry(
    n_theta: int = _PEANUT_N_THETA, n_phi: int = _PEANUT_N_PHI
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`r(theta) = 1 + 0.35*cos(2*theta)` の回転面を極座標グリッドで張る。

    theta in [0, pi] / phi in [0, 2pi]。UV は `u = phi/2pi`, `v = theta/pi` なので、
    **phi = 2pi の列と theta = 0/pi の極は頂点を複製する**(u/v が別値になるため
    1 頂点では表せない — glTF が UV シームで頂点を複製するのと同じ事情)。

    **位置を「ビット厳密に」一致させている WHY**: 複製頂点は
    `segmentation.adjacency.weld_vertices` が**厳密一致**でしか畳まない
    (許容誤差 weld を持たないのが設計)。素朴に `np.sin(theta)` を使うと
    `sin(pi) = 1.22e-16 != 0` なので南極の 65 頂点が微妙に別位置になり、
    `np.sin(2*pi) = -2.45e-16 != 0` なのでシーム列も畳まれない。実測(2026-07-29):
    weld 後の頂点が 3010 ではなく **3121**、辺が 9024 ではなく **8913** になり、
    閉多様体でなくなる。そこで極の `sin` と `cos`、シーム列の `sin`/`cos` を
    **解析値で上書き**してから座標を組む。

    極の三角形は**扇状に 1 枚**だけ張る(退化四角形を 2 分割しない)。weld 後に
    2 corner が同一頂点へ潰れる面は `build_face_adjacency` が
    「位置的縮退面」として 3 辺すべてを隣接から外すため、素朴に 2 枚張ると
    極の周囲が全カットされ、`DihedralSegmenter` が P==1 に退化しなくなる。

    巻き順は全面**外向き**(`(dP/dtheta) x (dP/dphi)` が外向き法線)。
    `dihedral_angles` は符号なし角なので、巻き順が混ざると同一平面でも 180 度と
    評価され過分割される(`adjacency.py` の既知の限界)。

    Returns:
        `(vertices (N, 3) float64, faces (M, 3) int64, uv (N, 2) float32)`。
    """
    theta = np.linspace(0.0, np.pi, n_theta + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    # 極: sin = 0 / cos = +-1 を解析値で固定する(上の WHY 参照)。
    sin_theta[0], sin_theta[-1] = 0.0, 0.0
    cos_theta[0], cos_theta[-1] = 1.0, -1.0
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    # シーム: phi = 2pi は phi = 0 と同一点。
    cos_phi[-1], sin_phi[-1] = cos_phi[0], sin_phi[0]
    # cos(2t) = 2cos^2(t) - 1 で組むと、極(cos t = +-1)の半径が両端で厳密に一致する。
    radius = 1.0 + _PEANUT_BULGE * (2.0 * cos_theta * cos_theta - 1.0)
    x = (radius * sin_theta)[:, np.newaxis] * cos_phi[np.newaxis, :]
    y = (radius * sin_theta)[:, np.newaxis] * sin_phi[np.newaxis, :]
    z = np.broadcast_to((radius * cos_theta)[:, np.newaxis], x.shape)
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    uv = np.stack(
        [
            np.broadcast_to((phi / (2.0 * np.pi))[np.newaxis, :], x.shape).ravel(),
            np.broadcast_to((theta / np.pi)[:, np.newaxis], x.shape).ravel(),
        ],
        axis=1,
    ).astype(np.float32)

    stride = n_phi + 1
    faces: list[list[int]] = []
    for row in range(n_theta):
        for col in range(n_phi):
            corner_a = row * stride + col  # (theta_i,   phi_j)
            corner_b = row * stride + col + 1  # (theta_i,   phi_j+1)
            corner_c = (row + 1) * stride + col + 1  # (theta_i+1, phi_j+1)
            corner_d = (row + 1) * stride + col  # (theta_i+1, phi_j)
            if row < n_theta - 1:
                # 南極の帯では c と d が同一点なので張らない。
                faces.append([corner_a, corner_d, corner_c])
            if row > 0:
                # 北極の帯では a と b が同一点なので張らない。
                faces.append([corner_a, corner_c, corner_b])
    return vertices, np.array(faces, dtype=np.int64), uv


def _peanut_truth_labels(mesh: MeshData) -> np.ndarray:
    """`peanut_mesh` の真値ラベル = 面重心の z の符号(計画v4 §5 Step 2-4)。

    北の膨らみ = 0 / 南の膨らみ = 1。くびれ(z = 0)を跨ぐ面が無いことは
    fixture の前提ゲートで確認する(重心 z の最小絶対値は約 0.0142)。
    """
    centroids = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].mean(axis=1)
    return (centroids[:, 2] < 0.0).astype(np.int64)


@pytest.fixture
def peanut_mesh() -> MeshData:
    """くびれた 2 つの膨らみを持つ滑らかな閉曲面(6016 面 / 位置weld 後 3010 頂点)。

    **設計意図**(計画v4 §5 Step 2-4): 接合部に鋭い折れ目が無いので
    **幾何プライア単独では P==1 に退化する**。多視点 ML が幾何に勝てるかを見る
    ための土俵であり、Step 2-4 ではその前提(P==1)自体をゲートで実証する。
    """
    vertices, faces, uv = _build_peanut_geometry()
    return MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps={},
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )


@pytest.fixture
def peanut_truth_labels() -> Callable[[MeshData], np.ndarray]:
    return _peanut_truth_labels


# 合成ビューで 1 面に割り当てる正方ブロックの一辺(画素)。4 なら 1 面 16 画素で、
# `assign_ratio` の既定 0.5 に対して十分な粒度がある。
_BLOCK_SIZE = 4


def _build_block_view(n_faces: int, *, block: int = _BLOCK_SIZE) -> RenderedView:
    """面 f を幅 `block` の縦帯へ写す合成 `RenderedView` を組む。

    画像は `(block + 2, n_faces * block + 2)`。**外周 1 画素は背景**にしてあるので、
    `coverage` が全 True の退化した入力にならない(段階A の背景除外が空虚な検査に
    ならないようにするため)。`color` は融合経路が読まないので 0 で埋める
    (色を読むのは `MaskProposer` だけ)。

    `coverage <=> (face_id >= 0)` は `RenderedView` の契約であり、`assign_view_faces`
    が `validate_coverage_consistency` で実際に検査する。

    Args:
        n_faces: メッシュの面数 M(全面が 1 画素以上写る)。
        block: 1 面あたりのブロックの一辺。

    Returns:
        `RenderedView(face_id (H, W) int32, color (H, W, 3) uint8, coverage bool)`。
    """
    height = block + 2
    width = n_faces * block + 2
    face_id = np.full((height, width), -1, dtype=np.int32)
    for face in range(n_faces):
        face_id[1 : 1 + block, 1 + face * block : 1 + (face + 1) * block] = face
    coverage = face_id >= 0
    color = np.zeros((height, width, 3), dtype=np.uint8)
    return RenderedView(face_id=face_id, color=color, coverage=coverage)


def _masks_from_face_groups(
    face_id: np.ndarray, groups: Sequence[Iterable[int]]
) -> np.ndarray:
    """面 index の集合ごとに 1 枚のマスクを作る `(K, H, W) bool`。

    どのグループにも入らない面の画素はどのマスクにも入らない = 段階A で
    `UNASSIGNED` になる。これで「その視点では観測されなかった面」を作れる。
    """
    ids = np.asarray(face_id)
    masks = np.zeros((len(groups), *ids.shape), dtype=bool)
    for index, group in enumerate(groups):
        wanted = np.array(sorted(set(int(face) for face in group)), dtype=np.int64)
        masks[index] = np.isin(ids, wanted)
    return masks


class _StaticRenderer:
    """`MeshRenderer` 契約を満たす合成レンダラ(視点ごとの `RenderedView` を返す)。

    **未入場 / 退場後の `render_view` を `RuntimeError` にしてある WHY**:
    `_ModernglRenderer` と同じ 3 状態(未入場・入場中・退場後)を再現することで、
    「`segment()` が本当に `with` で囲んでいるか」が検査できる。囲んでいなければ
    このスタブが即座に落ちる(2026-07-29 裁定1: `__enter__`/`__exit__` を呼ぶのは
    `segment()` 側)。`enter_count` / `exit_count` で回数も観測できる。
    """

    def __init__(self, views: Sequence[RenderedView]) -> None:
        self._views = list(views)
        self._state = "new"
        self.enter_count = 0
        self.exit_count = 0
        self.render_count = 0

    def __enter__(self) -> _StaticRenderer:
        if self._state != "new":
            raise RuntimeError(f"_StaticRenderer cannot be re-entered ({self._state})")
        self._state = "open"
        self.enter_count += 1
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._state = "closed"
        self.exit_count += 1

    def render_view(self, camera: Camera) -> RenderedView:
        if self._state != "open":
            raise RuntimeError(
                f"_StaticRenderer.render_view called while {self._state}; the caller "
                "must wrap it in `with renderer_factory(mesh) as renderer:`"
            )
        self.render_count += 1
        return self._views[camera.index]


class _StaticMaskProposer:
    """`MaskProposer` 契約を満たす固定マスク提案器(視点順に 1 組ずつ返す)。

    `propose` は呼ばれた順に `masks_per_view[i]` を返す — `RenderedView` には視点
    index が入っていないので、**呼び出し順が視点順であること**を前提にする
    (`segment()` はカメラを `index` 昇順に回す)。`K = 0`(空 `(0, H, W)`)も渡せる。
    """

    def __init__(self, masks_per_view: Sequence[np.ndarray]) -> None:
        self._masks_per_view = list(masks_per_view)
        self.propose_count = 0
        self.close_count = 0

    def propose(self, view: RenderedView) -> np.ndarray:
        if self.close_count:
            raise RuntimeError("_StaticMaskProposer.propose called after close()")
        masks = self._masks_per_view[self.propose_count]
        self.propose_count += 1
        return masks

    def close(self) -> None:
        self.close_count += 1


@pytest.fixture
def build_block_view() -> Callable[..., RenderedView]:
    return _build_block_view


@pytest.fixture
def masks_from_face_groups() -> Callable[..., np.ndarray]:
    return _masks_from_face_groups


@pytest.fixture
def static_renderer() -> Callable[..., _StaticRenderer]:
    return _StaticRenderer


@pytest.fixture
def static_mask_proposer() -> Callable[..., _StaticMaskProposer]:
    return _StaticMaskProposer
