"""io/bake 共通の合成 fixture(決定的・シード固定)。

メッシュ factory(cube/sphere/torus)と、テクスチャ factory(gradient/multisine/
aperiodic/face_id)を提供する。Phase 1(bake)のテストからも共用する前提
(test-design-step0-4.md 第1部)。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import trimesh

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
