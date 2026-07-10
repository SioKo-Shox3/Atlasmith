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
