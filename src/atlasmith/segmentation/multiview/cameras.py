"""多視点レンダリングの決定的カメラ配置(計画v4 §2.4.1)。

責務は3つだけ:

  1. Fibonacci 球による視点方向の決定的生成(**RNG 不使用**)。
  2. AABB 中心と外接球半径から view / projection 行列を組む。
  3. 「全頂点が視錐台に収まる」ことの検証(収まらなければ `ValueError`。
     カメラ距離の自動調整はしない — 計画v4 §2.4.1)。

行列は**行優先(数学記法)**の 4x4 で、列ベクトルに左から掛ける規約
(`clip = mvp @ [x, y, z, 1]`)。GL へ渡すときだけ転置する(GL は列優先)。

依存方向(計画v4 §2.1): numpy のみ。GL も trimesh も知らない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_FOV_DEG",
    "DEFAULT_N_VIEWS",
    "MAX_PERSPECTIVE_FOV_DEG",
    "PROJECTIONS",
    "Camera",
    "build_camera",
    "build_cameras",
    "fibonacci_directions",
    "validate_frustum",
]

# 既定の視点数(計画v4 §2.4.1)。
DEFAULT_N_VIEWS = 24
# 既定の垂直画角(度)。**透視投影専用**(下の `_camera_distance` の WHY を参照)。
DEFAULT_FOV_DEG = 30.0
# 受け付ける投影方式。未知の値は `ValueError`(計画v4 §5 Step 2-3 ゲート10)。
PROJECTIONS = ("perspective", "orthographic")

# 外接球を画面いっぱいに詰め込まないための余裕(半径に対する比)。1.0 だと球が
# 画面境界に接し、丸め次第で頂点が視錐台外へ出る。
_EXTENT_MARGIN = 1.1
# near/far 側の余裕。`near = d - R*1.2` / `far = d + R*1.2`(計画v4 §2.4.1)。
_DEPTH_MARGIN = 1.2
# 正射影時のカメラ距離 `d = R * 2.2`(オーケストレーター裁定1、2026-07-29)。
# WHY 2.2: near/far 規約が `near = d - R*1.2 > 0` を要求するので `d > 1.2R` が
# 必要条件。2.2R なら `near = R` の余裕が残る。
_ORTHOGRAPHIC_DISTANCE_FACTOR = 2.2

# **透視投影で実際に使える画角の上限**(度、これ未満)。
# 導出: `d = R * _EXTENT_MARGIN / sin(fov/2)` と `near = d - R * _DEPTH_MARGIN > 0` から
# `sin(fov/2) < _EXTENT_MARGIN / _DEPTH_MARGIN`(= 1.1/1.2)。つまり
# `fov < 2 * asin(1.1/1.2) = 132.887...` 度。**画角を広げると距離が縮み、near 平面が
# メッシュに追いつく**という距離規約の帰結であり、`0 < fov < 180` という一般的な
# 範囲より厳しい。定数から計算しているので、余裕係数を変えても式とずれない。
MAX_PERSPECTIVE_FOV_DEG = 2.0 * math.degrees(math.asin(_EXTENT_MARGIN / _DEPTH_MARGIN))

# up ベクトルの決定的分岐(計画v4 §2.4.1)。視線方向と up がほぼ平行だと
# `cross(forward, up)` が縮退するので、そのときだけ第2候補へ倒す。
_UP_PRIMARY = np.array([0.0, 0.0, 1.0])
_UP_FALLBACK = np.array([0.0, 1.0, 0.0])
_UP_PARALLEL_TOL = 0.99

# 視錐台判定の相対許容。行列積の丸めで、境界にちょうど乗る頂点(外接球の接点など)が
# 数 ulp だけ外へ出ることがある。カメラ距離を自動調整しない設計なので、ここを 0 に
# すると「設計上収まっているはずの入力」が丸めだけで落ちる。
_FRUSTUM_RTOL = 1e-6


@dataclass(frozen=True, eq=False)
class Camera:
    """1視点ぶんのカメラ(位置と2つの行列)。

    フィールド:
        index: 視点 index(エラーメッセージとログの識別用)。
        eye: カメラ位置 `(3,) float64`。
        target: 注視点 `(3,) float64`(= メッシュの AABB 中心)。
        up: up ベクトル `(3,) float64`。
        view: ワールド → ビューの 4x4 `float64`(行優先)。
        proj: ビュー → クリップの 4x4 `float64`(行優先)。

    **`eq=False` の WHY**: フィールドが ndarray なので、dataclass の生成する
    構造的 `__eq__` は要素ごとの比較配列を返し `bool()` で
    `ValueError: truth value of an array ...` になる。等価性は同一性で定義し、
    値の比較は呼び出し側が `np.array_equal` で明示的に行う(決定性ゲートも
    そうしている)。
    """

    index: int
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    view: np.ndarray
    proj: np.ndarray

    @property
    def mvp(self) -> np.ndarray:
        """`proj @ view`(モデル行列は恒等 — メッシュはワールド座標のまま)。"""
        return self.proj @ self.view


def fibonacci_directions(n_views: int) -> np.ndarray:
    """Fibonacci 球による単位視点方向 `(V, 3) float64` を返す。

    `z_i = 1 - (2i+1)/N`、`theta_i = pi(1+sqrt(5))·i`(計画v4 §2.4.1)。
    **RNG を使わないので同一入力に対してビット決定的**。

    Args:
        n_views: 視点数。1 以上の整数。

    Returns:
        `(n_views, 3) float64` の単位ベクトル。

    Raises:
        ValueError: `n_views` が整数でない、または 1 未満のとき。
    """
    if isinstance(n_views, bool) or not isinstance(n_views, (int, np.integer)):
        raise ValueError(f"n_views must be an int, got {type(n_views).__name__}")
    if int(n_views) < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    count = int(n_views)
    i = np.arange(count, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / float(count)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * i
    # `maximum(0, ...)` は 1 - z^2 が丸めで負になる端点対策(sqrt の NaN 化を防ぐ)。
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    dirs = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def build_camera(
    vertices: np.ndarray,
    direction: np.ndarray,
    *,
    projection: str,
    fov_deg: float = DEFAULT_FOV_DEG,
    index: int = 0,
) -> Camera:
    """`direction` の方向からメッシュ全体を見る 1 台のカメラを組む。

    Args:
        vertices: 対象メッシュの頂点 `(N, 3)`。読むだけで書き換えない。
        direction: カメラが置かれる向き(中心からカメラへ向かう単位でない
            ベクトルでもよい。内部で正規化する)。
        projection: `"perspective"` か `"orthographic"`。
        fov_deg: 垂直画角(度)。**透視投影専用**であり、正射影では距離にも
            画面範囲にも影響しない(裁定1)。範囲は
            **透視: `0 < fov_deg < MAX_PERSPECTIVE_FOV_DEG`(約 132.887 度)**、
            正射影: `0 < fov_deg < 180`(値は使われないが検証はする — 入口検証を
            投影方式で分岐させると「正射影なら不正な画角が黙って通る」穴になる)。
            透視の上限が 180 より厳しいのは距離規約の帰結
            (`MAX_PERSPECTIVE_FOV_DEG` の導出コメントを参照)。
        index: 視点 index(エラーメッセージ用)。

    Returns:
        `Camera`。

    Raises:
        ValueError: 引数が範囲外・投影方式が未知・外接球半径が非正/非有限、
            または**全頂点が視錐台に収まらない**とき。
    """
    center, radius = _bounding_sphere(vertices)
    return _build_camera(
        vertices,
        direction,
        center=center,
        radius=radius,
        projection=projection,
        fov_deg=fov_deg,
        index=index,
    )


def build_cameras(
    vertices: np.ndarray,
    *,
    n_views: int = DEFAULT_N_VIEWS,
    projection: str,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> list[Camera]:
    """メッシュを取り囲む `n_views` 台のカメラを決定的に並べる。

    中心 = `vertices` の AABB 中心、`R` = 中心からの最大距離(外接球半径)。
    方向は `fibonacci_directions`。**RNG 不使用なので 2 回呼べば同じ行列が出る。**

    `projection` に既定値を置かない **WHY**: 計画v4 §2.4.1 は既定を
    「Step 2-1.5 spike の実測で確定」としているが、spike(2026-07-29)は
    工学的検証のみで**既定を決めていない**
    (`Docs/agent-guide/technique-ml-part-segmentation.md` 末尾)。ここで暗黙の
    既定を作ると、未決の設計判断がライブラリの挙動として固定されてしまう。
    バックエンド(Step 2-4 の `MultiViewSegmenter`)側で明示的に決める。

    Args:
        vertices: 対象メッシュの頂点 `(N, 3)`。読むだけで書き換えない。
        n_views: 視点数(既定 24)。
        projection: `"perspective"` か `"orthographic"`。
        fov_deg: 垂直画角(度)。**透視投影専用**(裁定1)。範囲は `build_camera`
            と同じ(透視では約 132.887 度未満)。

    Returns:
        `index` が 0..n_views-1 の `Camera` のリスト。

    Raises:
        ValueError: `build_camera` と同じ条件。
    """
    directions = fibonacci_directions(n_views)
    center, radius = _bounding_sphere(vertices)
    return [
        _build_camera(
            vertices,
            direction,
            center=center,
            radius=radius,
            projection=projection,
            fov_deg=fov_deg,
            index=index,
        )
        for index, direction in enumerate(directions)
    ]


def validate_frustum(vertices: np.ndarray, camera: Camera) -> None:
    """全頂点が `camera` の視錐台内にあることを確認する。

    近平面クリッピングは扱わない(計画v4 §2.4.1)。クリップされた三角形は
    独立オラクル(テスト側の screen 空間 z バッファ)と GL で挙動が食い違い、
    主ゲートが「どちらが正しいか分からない」形で落ちるため、**そもそも
    起こさない**という設計にしてある。この関数はその前提の番人。

    Args:
        vertices: 検査する頂点 `(N, 3)`。
        camera: 検査対象のカメラ。

    Raises:
        ValueError: 視錐台外の頂点が 1 つでもあるとき(件数を報告する)。
    """
    verts = _as_vertices(vertices)
    homogeneous = np.concatenate(
        [verts, np.ones((verts.shape[0], 1), dtype=np.float64)], axis=1
    )
    clip = homogeneous @ camera.mvp.T
    w = clip[:, 3]
    limit = np.abs(w) * (1.0 + _FRUSTUM_RTOL)
    inside = (
        (w > 0.0)
        & (np.abs(clip[:, 0]) <= limit)
        & (np.abs(clip[:, 1]) <= limit)
        & (clip[:, 2] >= -limit)
        & (clip[:, 2] <= limit)
    )
    n_outside = int((~inside).sum())
    if n_outside:
        raise ValueError(
            f"camera {camera.index}: {n_outside} of {verts.shape[0]} vertices fall "
            "outside the view frustum. Camera distance is not auto-adjusted "
            "(plan v4 2.4.1); build the cameras from the same vertices you render."
        )


def _build_camera(
    vertices: np.ndarray,
    direction: np.ndarray,
    *,
    center: np.ndarray,
    radius: float,
    projection: str,
    fov_deg: float,
    index: int,
) -> Camera:
    """外接球を計算済みの内部ビルダ(`build_cameras` が V 回呼ぶ)。"""
    fov = _validate_fov(fov_deg)
    if projection not in PROJECTIONS:
        raise ValueError(
            f"unknown projection {projection!r}, expected one of {list(PROJECTIONS)}"
        )
    if projection == "perspective" and fov >= MAX_PERSPECTIVE_FOV_DEG:
        # 実際に使える上限は 180 度ではない。ここで弾かないと下の `near <= 0` に
        # 落ちて「距離規約が壊れた」と原因を誤報告する(2周目レビュー B5)。
        raise ValueError(
            f"fov_deg={fov} is too wide for the perspective camera: the distance "
            f"rule d = R*{_EXTENT_MARGIN}/sin(fov/2) then puts the near plane "
            f"(d - R*{_DEPTH_MARGIN}) at or below 0. The upper bound is "
            f"fov < 2*asin({_EXTENT_MARGIN}/{_DEPTH_MARGIN}) = "
            f"{MAX_PERSPECTIVE_FOV_DEG:.3f} degrees; use a narrower fov (the "
            f"default is {DEFAULT_FOV_DEG}) or switch to projection="
            "'orthographic', where fov_deg is unused."
        )
    unit = np.asarray(direction, dtype=np.float64).reshape(-1)
    if unit.shape != (3,):
        raise ValueError(f"direction must have shape (3,), got {np.shape(direction)}")
    norm = float(np.linalg.norm(unit))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"direction must be a non-zero finite vector, got {direction}")
    unit = unit / norm

    distance = _camera_distance(radius, projection=projection, fov_deg=fov)
    near = distance - radius * _DEPTH_MARGIN
    far = distance + radius * _DEPTH_MARGIN
    if near <= 0.0:
        # 上の画角ガードと正射影の `d = 2.2R`(near = R)で、実用範囲では到達しない。
        # 残るのは**画角が上限ぎりぎり**で near が数 ulp だけ負に丸まる場合なので、
        # 「規約が壊れた」と断定せず両方の可能性を示す(黙ってクランプはしない)。
        raise ValueError(
            f"near plane must be positive, got {near} "
            f"(distance={distance}, radius={radius}, projection={projection!r}, "
            f"fov_deg={fov}). If fov_deg is just under "
            f"{MAX_PERSPECTIVE_FOV_DEG:.3f}, use a narrower one; otherwise the "
            "camera-distance rule itself is inconsistent."
        )
    if projection == "perspective":
        proj = _perspective(fov, near, far)
    else:
        proj = _orthographic(radius * _EXTENT_MARGIN, near, far)

    up = (
        _UP_PRIMARY
        if abs(float(unit @ _UP_PRIMARY)) <= _UP_PARALLEL_TOL
        else _UP_FALLBACK
    )
    eye = center + unit * distance
    camera = Camera(
        index=int(index),
        eye=eye,
        target=center.copy(),
        up=up.copy(),
        view=_look_at(eye, center, up),
        proj=proj,
    )
    # production の fail-loud: 距離規約が正しければ必ず収まる。収まらないなら
    # 前提(外接球・余裕係数)が壊れているので、レンダする前にここで止める。
    validate_frustum(vertices, camera)
    return camera


def _bounding_sphere(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """AABB 中心と、そこからの最大距離(外接球半径)を返す。

    **WHY AABB 中心**: 最小外接球(Welzl 等)は実装が重く、決定性の議論も増える。
    多視点レンダに必要なのは「全頂点を確実に含む」ことだけで、半径を
    中心からの実測最大距離にすれば包含は厳密に保証される。
    """
    verts = _as_vertices(vertices)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(verts - center, axis=1).max())
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"bounding sphere radius must be finite and positive, got {radius} "
            "(a degenerate mesh whose vertices are all identical, or with "
            "non-finite coordinates, cannot be framed by a camera)"
        )
    return center, radius


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    """`(N, 3) float64` として検証した配列を返す(コピーはしない)。"""
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape}")
    if verts.shape[0] < 1:
        raise ValueError("vertices must have at least one row (N >= 1)")
    return verts


def _validate_fov(fov_deg: float) -> float:
    """`0 < fov_deg < 180` の有限値であることを確認して float で返す。"""
    fov = float(fov_deg)
    if not math.isfinite(fov) or not 0.0 < fov < 180.0:
        raise ValueError(f"fov_deg must be finite and in (0, 180), got {fov_deg}")
    return fov


def _camera_distance(radius: float, *, projection: str, fov_deg: float) -> float:
    """中心からカメラまでの距離。

    - 透視: `R / sin(fov/2) · 1.1`(半径 R の球が画角に収まる距離に余裕を掛ける)。
      画角が広いほど距離が縮むので、`near = d - 1.2R > 0` が画角の上限
      (`MAX_PERSPECTIVE_FOV_DEG`)を決める。呼び出し側で先に検証済み。
    - 正射影: `R · 2.2`(裁定1)。**画角は距離に効かない** — 正射影では画面範囲を
      half-extent が決めるので、画角で距離を決めるのは概念的に不自然。距離は
      near/far を正にできる値であれば足りる。
    """
    if projection == "perspective":
        return radius / math.sin(math.radians(fov_deg) * 0.5) * _EXTENT_MARGIN
    return radius * _ORTHOGRAPHIC_DISTANCE_FACTOR


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """右手系 lookAt(カメラは -Z を向く)。`(4, 4) float64` 行優先。"""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    side = np.cross(forward, up)
    side = side / np.linalg.norm(side)
    true_up = np.cross(side, forward)
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = side
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -float(side @ eye)
    view[1, 3] = -float(true_up @ eye)
    view[2, 3] = float(forward @ eye)
    return view


def _perspective(fov_deg: float, near: float, far: float) -> np.ndarray:
    """アスペクト比 1(正方形画像)の透視投影 `(4, 4) float64`。"""
    focal = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = focal
    proj[1, 1] = focal
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2.0 * far * near / (near - far)
    proj[3, 2] = -1.0
    return proj


def _orthographic(half_extent: float, near: float, far: float) -> np.ndarray:
    """アスペクト比 1(正方形画像)の正射影 `(4, 4) float64`。"""
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = 1.0 / half_extent
    proj[1, 1] = 1.0 / half_extent
    proj[2, 2] = -2.0 / (far - near)
    proj[2, 3] = -(far + near) / (far - near)
    proj[3, 3] = 1.0
    return proj
