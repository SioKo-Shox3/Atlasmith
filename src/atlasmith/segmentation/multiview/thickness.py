"""面ごとの厚み(Shape Diameter)— numpy 純正 BVH による内向きレイキャスト。

SAM2 の追加入力チャンネル「SDF 画像」の土台(計画v4 §5 Step 2-5)。実測
(2026-08-01 probe)で、単色 peanut は法線シェーディング単独だと SAM2 が
全滅(accuracy 0.5000)し、**厚みチャンネル単独で複合と同値の 0.9574** に
達した — つまりこのモジュールが ML バックエンドの成立条件である。

**このファイルが `sam2_masks.py`(隔離モジュール)と別である WHY**
(2026-08-01 オーケストレーター裁定 — 計画v4 §2.1 の module 一覧への追加):
厚みは決定的な幾何量であり、torch も GPU も無しに厳密テストできる。隔離
ファイルへ混ぜると「ML が無いと検証できないコード」に見えてしまい、
numpy 部分のテスト性が落ちる。このモジュールは **numpy のみ** に依存する。

**総当たりではなく BVH である WHY**(裁定C): レイ R 本 × 三角形 M 枚の
総当たりは O(R*M) で、想定上限の 20 万面(R = M)では 4*10^10 交差判定に
なりスケールしない。median-split の BVH + 反復(非再帰)トラバーサルなら
O(M log M) 級で済む。**依存は追加しない**(rtree を要求する `trimesh.ray` や
embree 系バインディングは使わない)— numpy だけで自前実装する。

決定性: RNG 不使用。構築は安定ソート(`np.argsort(kind="stable")`)、交差の
最小値更新は順序非依存(`np.minimum`)なので、同一入力に対しビット決定的。
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from atlasmith.segmentation.adjacency import face_normals

__all__ = [
    "compute_face_thickness01",
    "raycast_first_hit",
    "thickness_to_image",
]

LOG = logging.getLogger(__name__)

# --- スケール相対の数値許容(probe は絶対値 1e-4/1e-6/1e-12 だった) -----------
#
# **WHY 相対値へ変えたか**: 主用途の AI 生成メッシュは単位系が任意(m/cm/無次元)
# で届く。絶対オフセット 1e-4 は、bbox 対角 1e-3 の極小メッシュでは寸法の 10% を
# 突き抜け、対角 1e4 の巨大メッシュでは丸め誤差に埋もれる。すべて bbox 対角
# (長さの次元)を基準にすることでスケール不変にする。probe と同スケールの
# メッシュ(対角 ~4.7 の peanut)では probe とほぼ同じ実効値になる。

# レイ始点を面重心から内側へ引っ込める距離(対角比)。自面ヒットの回避。
_ORIGIN_OFFSET_RATIO = 1e-4
# これ以下の t は「自面近傍の数値ノイズ」として棄却する(対角比)。
_MIN_HIT_RATIO = 1e-6
# Möller–Trumbore の行列式がこの絶対値未満なら「レイと三角形が平行」として棄却。
# det = e1 . (D x e2) は長さの 2 乗の次元を持つので対角の 2 乗を掛ける。
_DET_EPS_RATIO = 1e-12

# 葉ノードに残す三角形数の上限。小さいほど交差判定が減りノード数が増える。
# 8 は「葉の総当たりが (R_chunk, 8) の小さな broadcast で済む」経験的な中庸。
_LEAF_SIZE = 8

# 正規化のクリップ位置(パーセンタイル)。probe と同じ 5/95 — 外れ値(貫通など)
# に [0,1] の大半を食われないようにするため。
_CLIP_PERCENTILES = (5.0, 95.0)
# クリップ幅がこれ未満なら「全面ほぼ同厚」とみなし一様値 0.5 を返す。
_DEGENERATE_RANGE = 1e-12


def _validate_mesh_arrays(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(N, 3)` 頂点と `(M, 3)` 整数面の contract を検証して float64/int64 で返す。"""
    verts = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape}")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {face_array.shape}")
    if face_array.size and not np.issubdtype(face_array.dtype, np.integer):
        raise ValueError(
            f"faces must be an integer array, got dtype {face_array.dtype}"
        )
    return verts, face_array.astype(np.int64, copy=False)


def _bbox_diagonal(vertices: np.ndarray) -> float:
    """スケール相対許容の基準になる bbox 対角長。頂点 0 個なら 0。"""
    if vertices.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))


def _build_bvh(
    tri_lo: np.ndarray, tri_hi: np.ndarray, tri_centroids: np.ndarray
) -> tuple[np.ndarray, ...]:
    """median-split BVH を**反復的に**構築する(裁定C — 再帰しない)。

    分割規約(すべて決定的):
      - 分割軸 = 三角形重心の広がりが最大の軸(`np.argmax` は tie で最小軸)。
      - 重心座標の**安定ソート**で並べ、中央(`count // 2`)で 2 分する。
      - `count <= _LEAF_SIZE`、または全重心が同一点(広がり 0 — これ以上
        分けられない)なら葉にする。後者が無いと同一重心の三角形群で無限分割に陥る。

    Returns:
        `(node_lo (n,3), node_hi (n,3), node_left (n,), node_right (n,),
        node_start (n,), node_count (n,), order (M,))`。`node_left < 0` が葉で、
        葉は `order[start:start+count]` の三角形 index を持つ。
    """
    n_tris = int(tri_lo.shape[0])
    order = np.arange(n_tris, dtype=np.int64)
    node_lo: list[np.ndarray] = []
    node_hi: list[np.ndarray] = []
    node_left: list[int] = []
    node_right: list[int] = []
    node_start: list[int] = []
    node_count: list[int] = []
    # (start, count, parent, side)。parent < 0 はルート。LIFO で処理するが、
    # ノードの内容は (start, count) だけで決まるので処理順は結果に影響しない。
    tasks: list[tuple[int, int, int, int]] = [(0, n_tris, -1, 0)]
    while tasks:
        start, count, parent, side = tasks.pop()
        index = len(node_lo)
        if parent >= 0:
            if side == 0:
                node_left[parent] = index
            else:
                node_right[parent] = index
        segment = order[start : start + count]
        node_lo.append(tri_lo[segment].min(axis=0))
        node_hi.append(tri_hi[segment].max(axis=0))
        node_left.append(-1)
        node_right.append(-1)
        centroids = tri_centroids[segment]
        extent = centroids.max(axis=0) - centroids.min(axis=0)
        if count <= _LEAF_SIZE or float(extent.max()) <= 0.0:
            node_start.append(start)
            node_count.append(count)
            continue
        node_start.append(-1)
        node_count.append(0)
        axis = int(np.argmax(extent))
        local = np.argsort(centroids[:, axis], kind="stable")
        order[start : start + count] = segment[local]
        half = count // 2
        tasks.append((start + half, count - half, index, 1))
        tasks.append((start, half, index, 0))
    return (
        np.asarray(node_lo, dtype=np.float64),
        np.asarray(node_hi, dtype=np.float64),
        np.asarray(node_left, dtype=np.int64),
        np.asarray(node_right, dtype=np.int64),
        np.asarray(node_start, dtype=np.int64),
        np.asarray(node_count, dtype=np.int64),
        order,
    )


def _first_hit_among(
    origins: np.ndarray,
    directions: np.ndarray,
    tri_v0: np.ndarray,
    tri_e1: np.ndarray,
    tri_e2: np.ndarray,
    min_hit: float,
    det_eps: float,
) -> np.ndarray:
    """葉の三角形集合に対する Möller–Trumbore(両面ヒット)の最小 t を返す。

    レイ R 本 × 三角形 T 枚を全対 broadcast する — T は葉サイズ
    (`<= _LEAF_SIZE`)なので中間配列は小さい。`t > min_hit` を満たす最小 t、
    無ければ `inf`。det の符号は問わない(裏面からのヒットも数える —
    厚みレイはメッシュ内部を通るため、裏面が「反対側の壁」である)。
    """
    pvec = np.cross(directions[:, None, :], tri_e2[None, :, :])
    det = np.einsum("tk,rtk->rt", tri_e1, pvec)
    parallel = np.abs(det) < det_eps
    inv_det = 1.0 / np.where(parallel, 1.0, det)
    tvec = origins[:, None, :] - tri_v0[None, :, :]
    u = np.einsum("rtk,rtk->rt", tvec, pvec) * inv_det
    qvec = np.cross(tvec, tri_e1[None, :, :])
    v = np.einsum("rk,rtk->rt", directions, qvec) * inv_det
    t = np.einsum("tk,rtk->rt", tri_e2, qvec) * inv_det
    valid = ~parallel & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > min_hit)
    return np.where(valid, t, np.inf).min(axis=1)


def raycast_first_hit(
    vertices: np.ndarray,
    faces: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """各レイの最初のヒット距離を返す(ヒット無しは `inf`)。

    交差は両面(表裏を区別しない)。`t` は `directions` の長さを単位とするので、
    距離が欲しければ単位ベクトルを渡すこと(`compute_face_thickness01` は単位
    法線を渡す)。自明に小さい `t`(自面近傍)と平行レイの棄却は、メッシュの
    bbox 対角に相対的なしきい値で行う(モジュール docstring の WHY 参照)。

    Args:
        vertices: `(N, 3)`。書き換えない。
        faces: `(M, 3)` 整数。書き換えない。
        origins: レイ始点 `(R, 3)`。
        directions: レイ方向 `(R, 3)`(単位ベクトル推奨)。

    Returns:
        `(R,) float64`。ヒット無し(および `M == 0`)は `inf`。

    Raises:
        ValueError: いずれかの配列 shape が契約に合わないとき。
    """
    verts, face_array = _validate_mesh_arrays(vertices, faces)
    origin_array = np.asarray(origins, dtype=np.float64)
    direction_array = np.asarray(directions, dtype=np.float64)
    if origin_array.ndim != 2 or origin_array.shape[1] != 3:
        raise ValueError(f"origins must have shape (R, 3), got {origin_array.shape}")
    if direction_array.shape != origin_array.shape:
        raise ValueError(
            f"directions must have the same shape as origins "
            f"{origin_array.shape}, got {direction_array.shape}"
        )
    n_rays = int(origin_array.shape[0])
    if face_array.shape[0] == 0 or n_rays == 0:
        return np.full(n_rays, np.inf, dtype=np.float64)

    diagonal = _bbox_diagonal(verts)
    min_hit = _MIN_HIT_RATIO * diagonal
    det_eps = _DET_EPS_RATIO * diagonal * diagonal

    triangles = verts[face_array]
    tri_v0 = triangles[:, 0, :]
    tri_e1 = triangles[:, 1, :] - tri_v0
    tri_e2 = triangles[:, 2, :] - tri_v0
    bvh = _build_bvh(
        triangles.min(axis=1), triangles.max(axis=1), triangles.mean(axis=1)
    )
    node_lo, node_hi, node_left, node_right, node_start, node_count, order = bvh

    best = np.full(n_rays, np.inf, dtype=np.float64)
    # 方向成分が厳密に 0 の軸は**除算せず明示分岐する**(2026-08-03 反証レビュー
    # B-4)。旧実装は 0 を 1e-300 で置換していたが、`origin[k]` がノード AABB の
    # `hi[k]` と厳密一致すると `(hi - origin) * 1e300 = 0.0 * 1e300 = 0.0` で far が
    # 0 に張り付き、前方(near > 0)のノードが軒並み `far >= near` で棄却された
    # (実測: AABB 上面をかすめる軸平行レイが全ヒット取りこぼし)。レイがその軸で
    # 動かない以上、スラブ判定は「origin がスラブ内か」だけであり、
    # 内なら制約なし `[-inf, +inf]`、外なら交差なし `[+inf, -inf]` が正しい。
    zero_dir = direction_array == 0.0
    inv_dir = 1.0 / np.where(zero_dir, 1.0, direction_array)
    # フロンティア方式のトラバーサル: (ノード, そのノードの AABB を試すレイ集合)
    # をスタックで回す。レイごとの再帰やノードごとの python ループ入れ子を避け、
    # 各ステップを numpy のベクトル演算に保つ。処理順は best の途中経過(枝刈り)
    # にしか影響せず、最終値は `np.minimum` の可換性により順序非依存。
    stack: list[tuple[int, np.ndarray]] = [(0, np.arange(n_rays, dtype=np.int64))]
    with np.errstate(over="ignore"):
        while stack:
            node, rays = stack.pop()
            origin = origin_array[rays]
            inv = inv_dir[rays]
            zero = zero_dir[rays]
            t1 = (node_lo[node] - origin) * inv
            t2 = (node_hi[node] - origin) * inv
            slab_near = np.minimum(t1, t2)
            slab_far = np.maximum(t1, t2)
            inside = (origin >= node_lo[node]) & (origin <= node_hi[node])
            slab_near = np.where(zero, np.where(inside, -np.inf, np.inf), slab_near)
            slab_far = np.where(zero, np.where(inside, np.inf, -np.inf), slab_far)
            near = np.maximum(slab_near.max(axis=1), 0.0)
            far = slab_far.min(axis=1)
            alive = (far >= near) & (near < best[rays])
            rays = rays[alive]
            if rays.size == 0:
                continue
            if node_left[node] < 0:
                segment = order[node_start[node] : node_start[node] + node_count[node]]
                t = _first_hit_among(
                    origin_array[rays],
                    direction_array[rays],
                    tri_v0[segment],
                    tri_e1[segment],
                    tri_e2[segment],
                    min_hit,
                    det_eps,
                )
                best[rays] = np.minimum(best[rays], t)
            else:
                stack.append((int(node_left[node]), rays))
                stack.append((int(node_right[node]), rays))
    return best


def compute_face_thickness01(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """面ごとの厚み(Shape Diameter)を [0, 1] に正規化して返す。

    各面の重心から内向き(`-normal`)へレイを飛ばし、最初のヒット(= 反対側の
    壁)までの距離を厚みとする。probe(2026-08-01)と同じ定義:

      - 始点は重心から内側へ `bbox 対角 * 1e-4` オフセット(自面ヒット回避)。
      - 零面積面(法線が定義できない)とヒット無しの面は、有効な厚みの
        **中央値**で埋める(画像上で「異常に明るい/暗い穴」を作らないため)。
      - 5/95 パーセンタイルでクリップして [0, 1] へ線形写像。全面ほぼ同厚
        (クリップ幅がほぼ 0)なら一様に 0.5 + `UserWarning`(下記)。

    **巻き順が反転したメッシュの救済(2026-08-03 反証レビュー B-2)**: 法線が
    一貫して内向きのメッシュ(AI 生成・インポート由来で頻出 = 本ツールの主対象
    入力)では `-normal` のレイが全て外へ逃げる。そこで**全レイ逃走時に限り
    `+normal` で 1 回だけ再計算**し、成功したら `LOG.warning` で告知して続行する
    (メッシュ自体は閉じているのに「watertight でない」と誤診断して落ちるのを
    避けるため)。**既知の限界: 巻き順が「混在」しているメッシュは救えない** —
    一部の面だけが逃げるので全レイ逃走の条件に当たらず、その面はヒット無しとして
    中央値で埋められる(厚みとしては嘘だが、画像に穴は空かない)。混在巻き順の
    正規化はこの関数の責務ではない。

    Args:
        vertices: `(N, 3)`。書き換えない。
        faces: `(M, 3)` 整数。書き換えない。

    Returns:
        `(M,) float64`、値域 [0, 1]。`M == 0` なら `(0,)`。

    Warns:
        UserWarning: 5/95 パーセンタイル幅が退化して全面 0.5 になったとき
            (= SDF 画像が一様グレーになり SAM2 への信号が消える)。黙って
            無信号の画像を渡さない(N-1)。

    Raises:
        ValueError: shape 契約違反。
        RuntimeError: 全面が零面積でレイを 1 本も張れないとき、または**両方向**
            とも 1 本もヒットしないとき(開いた曲面、あるいは巻き順不整合 —
            どちらでも厚みが定義できない)。SAM2 側はこの例外を握り潰さない
            (黙って無意味な SDF 画像を作らない)。
    """
    verts, face_array = _validate_mesh_arrays(vertices, faces)
    n_faces = int(face_array.shape[0])
    if n_faces == 0:
        return np.zeros(0, dtype=np.float64)

    normals, zero_area = face_normals(verts, face_array)
    valid = ~np.asarray(zero_area, dtype=bool)
    if not valid.any():
        raise RuntimeError(
            "all faces are zero-area; cannot cast shape-diameter rays "
            "(the mesh has no usable geometry for the SDF channel)"
        )

    diagonal = _bbox_diagonal(verts)
    offset = _ORIGIN_OFFSET_RATIO * diagonal
    centroids = verts[face_array].mean(axis=1)
    directions = -normals[valid]
    origins = centroids[valid] + directions * offset
    distances_valid = raycast_first_hit(verts, face_array, origins, directions)
    if not np.isfinite(distances_valid).any():
        # 全レイ逃走 → 巻き順が一貫して内向きの可能性(docstring の WHY)。
        # 逆方向で 1 回だけ試す。再計算は「0 ヒット」のときだけなので、
        # 正常なメッシュに追加コストは掛からない。
        directions = -directions
        retried = raycast_first_hit(
            verts, face_array, centroids[valid] + directions * offset, directions
        )
        if np.isfinite(retried).any():
            LOG.warning(
                "thickness: every -normal ray escaped but +normal rays hit the "
                "mesh, so the face winding looks consistently inverted (normals "
                "point inward). Computed the shape diameter along +normal "
                "instead; the mesh itself is closed, only its winding convention "
                "is reversed."
            )
            distances_valid = retried
        else:
            raise RuntimeError(
                "no shape-diameter ray hit the mesh in either direction (rays "
                "along -normal and along +normal both escaped from every face). "
                "Either the surface is open (not watertight - a lone triangle or "
                "a shell with boundaries), or its winding is inconsistent so that "
                "no single direction points inward. The SDF channel needs a "
                "closed, consistently wound surface; use `--segmenter geometric` "
                "for such meshes."
            )

    distances = np.full(n_faces, np.inf, dtype=np.float64)
    distances[valid] = distances_valid
    hit = np.isfinite(distances)
    n_filled = int((~hit).sum())
    if n_filled:
        LOG.info(
            "thickness: %d/%d faces had no inward hit (or zero area); "
            "filled with the median of the %d measured faces",
            n_filled,
            n_faces,
            int(hit.sum()),
        )
    distances[~hit] = float(np.median(distances[hit]))

    low, high = (float(v) for v in np.percentile(distances, _CLIP_PERCENTILES))
    if high - low < _DEGENERATE_RANGE:
        # N-1(2026-08-03 反証レビュー): ここへ落ちると SDF 画像は一様グレーで、
        # SAM2 へ渡る信号が**ゼロ**になる。ML を指定したのに幾何プライア相当へ
        # 崩れる経路なので、LOG ではなく警告で知らせる(黙って続けない)。
        warnings.warn(
            f"the shape-diameter channel is degenerate: the 5/95 percentile "
            f"spread over {n_faces} faces is {high - low:.3g}, so every face "
            "collapses to a uniform 0.5 and the SDF image becomes flat gray "
            f"with no signal for SAM2 ({n_filled} face(s) had no hit and were "
            "filled with the median). Either the mesh really has a uniform "
            "diameter (a cube, a shell), or too few rays hit. Consider "
            '`channels=("shading",)` for a textured mesh, or '
            "`--segmenter geometric`.",
            UserWarning,
            stacklevel=2,
        )
        return np.full(n_faces, 0.5, dtype=np.float64)
    return np.clip((distances - low) / (high - low), 0.0, 1.0)


def thickness_to_image(face_id: np.ndarray, thickness01: np.ndarray) -> np.ndarray:
    """面ID バッファから SDF グレースケール 3ch 画像を合成する(新規レンダ不要)。

    `face_id` は既にレンダ済みなので、面ごとのスカラを画素へ**引くだけ**で
    SDF 画像が得られる(probe 実証済みの方式 — 視点ごとの追加レイキャスト無し)。
    背景(`face_id < 0`)は 0(黒)。SAM2 は RGB を要求するので同値 3ch にする。

    Args:
        face_id: `(H, W)` 整数、背景は負値。
        thickness01: `compute_face_thickness01` の戻り値 `(M,) float64`、[0, 1]。

    Returns:
        `(H, W, 3) uint8`(C 連続)。

    Raises:
        ValueError: shape/dtype/値域の契約違反、または `face_id` が
            `thickness01` の範囲外の面を参照しているとき(メッシュ跨ぎの
            取り違え — stale な厚みを黙って使わない)。
    """
    ids = np.asarray(face_id)
    values = np.asarray(thickness01, dtype=np.float64)
    if ids.ndim != 2:
        raise ValueError(f"face_id must have shape (H, W), got {ids.shape}")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"face_id must be an integer array, got dtype {ids.dtype}")
    if values.ndim != 1:
        raise ValueError(f"thickness01 must have shape (M,), got {values.shape}")
    if values.size and (
        not np.isfinite(values).all()
        or float(values.min()) < 0.0
        or float(values.max()) > 1.0
    ):
        raise ValueError("thickness01 values must be finite and within [0, 1]")
    max_id = int(ids.max()) if ids.size else -1
    if max_id >= values.size:
        raise ValueError(
            f"face_id references face {max_id} but thickness01 has only "
            f"{values.size} entries; the thickness was computed for a different "
            "mesh (set_face_thickness must run for the mesh being segmented)"
        )
    if values.size == 0:
        gray01 = np.zeros(ids.shape, dtype=np.float64)
    else:
        gray01 = np.where(ids >= 0, values[np.maximum(ids, 0)], 0.0)
    gray = np.round(gray01 * 255.0).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
