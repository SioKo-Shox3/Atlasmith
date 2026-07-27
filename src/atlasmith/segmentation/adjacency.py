"""位置weld・面隣接・二面角 — 部位分割の幾何前段(numpy のみ)。

責務は3つ: (1) 位置が一致する頂点を束ねる weld 写像、(2) weld 代表 index で
正規化した辺を「ちょうど2面が共有する」ものだけ隣接とみなす面隣接、(3) 隣接
ペアの二面角。

**`geometric.py` から分離してある WHY**: Step 2-4 の `multiview/fusion.py` が
**同じ weld 隣接**を再利用する(計画v4 §2.3 末尾)。多視点融合の段階B/C は
「観測面どうしを結ぶ辺」を面隣接そのものの上で数えるため、幾何バックエンドの
実装詳細に相乗りさせるとバックエンド間に不要な依存が生まれる。

依存方向(計画v4 §2.1): `segmentation` 配下は **numpy のみ**。trimesh / xatlas /
PIL / torch / moderngl / sam2 と `atlasmith.io` / `atlasmith.pack` /
`atlasmith.bake` は import しない。
"""

from __future__ import annotations

import numpy as np


def weld_vertices(vertices: np.ndarray) -> np.ndarray:
    """位置が一致する頂点を束ね、各頂点 → 代表頂点 index の写像を返す。

    `np.lexsort` で行を辞書順に並べ、隣接行が完全一致するかぎり同一グループとする。
    グループの代表は **グループ内で最小の元頂点 index**(決定的)。

    **epsilon を使わない WHY**(計画v2 §2.1 / v4 §2.3): `load_mesh` は
    `trimesh.load(path, process=False)` で weld しない(`src/atlasmith/io/mesh.py`
    :107-118)ため、glTF が UV シームで複製した頂点は**同一値のコピー**として残る。
    複製の除去には厳密一致で足り、許容誤差 weld は「閾値をいくつにするか」という
    新たな技術判断を生む。`source_vertex` は io では恒等写像(同 :141-142)なので
    weld の代用にならない。

    IEEE 比較の隅(実装として明示しておく):
      - `-0.0` と `0.0` は等しいので同一グループになる(幾何的にも同一点)。
      - `NaN` はどの値とも等しくないので、`NaN` を含む頂点は必ず単独グループになる。

    Args:
        vertices: 頂点座標 `(N, 3)`。値は読むだけで書き換えない。

    Returns:
        `(N,) int64`。`weld_map[i]` は頂点 `i` の代表頂点 index。

    Raises:
        ValueError: `vertices` の shape が `(N, 3)` でないとき。
    """
    verts = np.asarray(vertices)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape}")
    n_vertices = verts.shape[0]
    weld_map = np.empty(n_vertices, dtype=np.int64)
    if n_vertices == 0:
        return weld_map

    # lexsort は「最後のキーが第一ソートキー」なので列を x, y, z の順に並べる。
    order = np.lexsort((verts[:, 2], verts[:, 1], verts[:, 0]))
    ordered = verts[order]
    starts_group = np.ones(n_vertices, dtype=bool)
    if n_vertices > 1:
        starts_group[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    group = np.cumsum(starts_group) - 1

    n_groups = int(group[-1]) + 1
    representative = np.full(n_groups, np.iinfo(np.int64).max, dtype=np.int64)
    # ソート順に依らずグループ内最小の *元* index を代表にする。
    np.minimum.at(representative, group, order)
    weld_map[order] = representative[group]
    return weld_map


def build_face_adjacency(faces: np.ndarray, weld_map: np.ndarray) -> np.ndarray:
    """weld 代表 index で正規化した辺を共有する **ちょうど2面** のペアを返す。

    **3面以上が共有する辺(非多様体)は隣接に数えない = 必ずカットする**
    (計画v2 §2.1 手順1)。決定的かつ保守的な扱いで、どの2面を選ぶかという
    恣意的な判断を持ち込まない。

    **位置的縮退面の扱い**: weld で2つ以上の corner が同一代表に潰れる面は、
    その3辺すべてを隣接候補から除外する(= 全隣接カット、計画v2 §2.1 / §2.6)。
    除外は共有数の数え上げ**より前**に効くので、結果は2方向に分かれる:

      - 相手が1面だけだった辺は共有数が 2 → **1** に落ち、両側ともカットされる。
      - 生の入力では3面が共有していた辺は 3 → **2** に落ち、**残り2面が新たに
        隣接になる**(縮退面を「そこに無いもの」として扱った帰結)。AI 生成
        メッシュに頻出する「内部辺に貼り付いたゼロ幅スライバ三角形」でこの経路を
        踏む。縮退面を先に落とす選択は決定的で幾何的にも擁護できるため意図どおり
        だが、「非多様体辺は必ずカット」だけを読むと予測できない挙動なので明記する。

    同じ2面が複数の辺を共有する場合(例: 2枚が張り合わさった退化形状)、その
    ペアは**共有辺の本数だけ行が現れる**。小部位マージ(`labels.merge_small_parts`)
    が「共有境界辺数」を数えるのにこの重複をそのまま使う。

    Args:
        faces: 三角形の頂点 index `(M, 3)`。読むだけで書き換えない。
        weld_map: `weld_vertices` の戻り値 `(N,)`。

    Returns:
        `(E, 2) int64`。各行は隣接する2面の index で `row[0] < row[1]`。行は
        第0列 → 第1列の辞書順に整列済み(決定的)。

    Raises:
        ValueError: `faces` の shape が `(M, 3)` でない、`weld_map` が1次元でない、
            または `faces` が `weld_map` の範囲外を指すとき。
    """
    face_array = np.asarray(faces)
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {face_array.shape}")
    weld = np.asarray(weld_map)
    if weld.ndim != 1:
        raise ValueError(f"weld_map must have shape (N,), got {weld.shape}")

    empty = np.empty((0, 2), dtype=np.int64)
    n_faces = face_array.shape[0]
    if n_faces == 0:
        return empty
    if face_array.size and (face_array.min() < 0 or face_array.max() >= weld.shape[0]):
        raise ValueError(
            f"faces reference vertex indices outside weld_map of length "
            f"{weld.shape[0]} (min={face_array.min()}, max={face_array.max()})"
        )

    welded_corners = weld[face_array]  # (M, 3)
    collapsed = (
        (welded_corners[:, 0] == welded_corners[:, 1])
        | (welded_corners[:, 1] == welded_corners[:, 2])
        | (welded_corners[:, 2] == welded_corners[:, 0])
    )
    usable_faces = np.flatnonzero(~collapsed)
    if usable_faces.size == 0:
        return empty

    corners = welded_corners[usable_faces]
    edges = np.concatenate(
        [corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)  # 辺の向きを昇順に正規化
    owner = np.tile(usable_faces, 3)

    # 第一キー = edges[:, 0]、第二キー = edges[:, 1]、第三キー = owner。
    # 第三キーまで指定するのは、同一辺グループ内の2面が必ず昇順に並ぶようにするため。
    order = np.lexsort((owner, edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owner = owner[order]

    starts_group = np.ones(edges.shape[0], dtype=bool)
    if edges.shape[0] > 1:
        starts_group[1:] = np.any(edges[1:] != edges[:-1], axis=1)
    group = np.cumsum(starts_group) - 1
    counts = np.bincount(group)
    group_start = np.flatnonzero(starts_group)

    manifold = np.flatnonzero(counts == 2)
    if manifold.size == 0:
        return empty
    first = group_start[manifold]
    pairs = np.stack([owner[first], owner[first + 1]], axis=1).astype(np.int64)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def face_normals(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """単位面法線と零面積面マスクを返す。

    法線は `cross(v1 - v0, v2 - v0)` を正規化したもの。**零面積面(外積の長さが
    厳密に 0、または NaN)には法線が定義できない**ので `[0, 0, 0]` を入れ、
    マスクで呼び出し側へ知らせる。閾値による「ほぼ零面積」の判定はしない —
    しきい値の選定は独立した技術判断であり、計画が求めていない。

    Args:
        vertices: 頂点座標 `(N, 3)`。読むだけで書き換えない。
        faces: 三角形の頂点 index `(M, 3)`。読むだけで書き換えない。

    Returns:
        `(normals (M, 3) float64, zero_area (M,) bool)`。
    """
    verts = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces)
    corners = verts[face_array]
    raw = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    # `~(lengths > 0)` と書くのは NaN を零面積側へ寄せるため(`lengths == 0` だと
    # NaN が「有効な法線」として通り抜ける)。
    zero_area = ~(lengths > 0.0)
    normals = np.zeros_like(raw)
    np.divide(raw, lengths[:, np.newaxis], out=normals, where=~zero_area[:, np.newaxis])
    return normals, zero_area


def dihedral_angles(normals: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """隣接ペアの二面角(度、`[0, 180]`)を返す。

    凸/凹は区別しない(minima-rule は Phase 3 候補 — 計画v2 §2.1 手順2)。

    **零面積面を含むペアの値は無意味**: `face_normals` が零ベクトルを返すため
    内積 0 = 90 度になる。呼び出し側は `face_normals` の `zero_area` マスクで
    そのペアを除外すること(90 度という値に依存しない) — 両方をまとめて正しく
    適用したいなら `smooth_edge_mask` を使う。

    **既知の限界: 一貫した巻き方向を前提とする。** 式は符号なし角
    `arccos(n_i . n_j)` なので(計画v2 §2.1 手順2 の指定どおり)、隣接2面の
    巻き順が互いに逆だと法線が反転し、**同一平面上の面ペアでも 180 度**と
    評価される。結果としてどの `angle_deg` を指定しても必ずカットされ、巻き順が
    不整合なメッシュは過分割される(実測: 平面グリッド 8 面のうち 1 面を反転
    すると `min_faces=1` で P=1 ではなく P=3)。既定 `min_faces` では小部位
    マージが結果的に埋め合わせることが多いが、それは設計ではなく副作用である。
    CLAUDE.md が主用途と定める AI 生成メッシュは巻き順が不整合なことがあるため、
    現実に踏みうる限界として記録する。符号付き二面角(凹凸の区別)は Phase 3 候補。

    Args:
        normals: `face_normals` の単位法線 `(M, 3)`。
        adjacency: `build_face_adjacency` の戻り値 `(E, 2)`。

    Returns:
        `(E,) float64`。
    """
    pairs = np.asarray(adjacency)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"adjacency must have shape (E, 2), got {pairs.shape}")
    if pairs.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    dots = np.einsum("ij,ij->i", normal_array[pairs[:, 0]], normal_array[pairs[:, 1]])
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def validate_angle_deg(angle_deg: float) -> float:
    """二面角しきい値の契約(有限かつ `0 < angle_deg <= 180`)を検証して float で返す。

    `DihedralSegmenter.__init__` と `smooth_edge_mask` が**同じ検証**を使うために
    切り出してある(計画v2 §2.1 のパラメータ検証)。判定を2箇所に書くと、片方だけが
    `nan` を通してしきい値比較が全 False になり「全辺カット」が無言で起きる。

    Args:
        angle_deg: 検証するしきい値(度)。

    Returns:
        `float` へ正規化した `angle_deg`。

    Raises:
        ValueError: 非有限、または `(0, 180]` の範囲外のとき。
    """
    angle = float(angle_deg)
    if not np.isfinite(angle):
        raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")
    if not 0.0 < angle <= 180.0:
        raise ValueError(f"angle_deg must be in (0, 180], got {angle}")
    return angle


def smooth_edge_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    adjacency: np.ndarray,
    *,
    angle_deg: float,
) -> np.ndarray:
    """「二面角がしきい値以下 **かつ** 両側とも零面積でない」辺の bool マスクを返す。

    幾何的に「この辺の両側は同じ滑らかな面の一部とみなしてよい」という判定を
    **1箇所に閉じ込める**ための関数。二面角の判定(`dihedral_angles`)と零面積面の
    カット(`face_normals` の `zero_area`)は**必ず組で適用しなければ正しくない** —
    零面積面の法線は零ベクトルで、内積 0 = 90 度と評価されるため、`angle_deg >= 90`
    の設定では零面積面が滑らかな辺として繋がってしまう。

    **共有部品にしてある WHY**: `DihedralSegmenter`(`geometric.py` 手順2〜3)と、
    Step 2-4 の `multiview/fusion.py` 段階B の**幾何プライア** `prior(a, b)` が
    まったく同じ判定を必要とする。呼び出し側で2行に分けて書くと、片方だけを
    写し忘れた実装が「§6 の幾何プライアへの完全劣化ゲート(全マスク未割当の入力で
    出力が `DihedralSegmenter` と一致すること)」を特定のメッシュでだけ破る、という
    再現しにくい壊れ方をする。計画v4 §2.3 末尾の「`fusion.py` は `adjacency.py` /
    `labels.py` を再利用する(重複実装しない)」に対応する分離。

    Args:
        vertices: 頂点座標 `(N, 3)`。読むだけで書き換えない。
        faces: 三角形の頂点 index `(M, 3)`。読むだけで書き換えない。
        adjacency: `build_face_adjacency` の戻り値 `(E, 2)`。
        angle_deg: この角度**以下**を「滑らか」とみなすしきい値(度)。境界は
            **閉**(`角度 <= angle_deg`)。有限かつ `0 < angle_deg <= 180`。

    Returns:
        `(E,) bool`。`True` = 滑らかな辺(部位を跨がない)。

    Raises:
        ValueError: `angle_deg` が非有限・範囲外のとき、または `adjacency` の
            shape が `(E, 2)` でないとき。
    """
    # 入口で検証する WHY: `angle_deg=nan` だと比較が全 False になり「全辺カット」が
    # 無言で起きる。`DihedralSegmenter` は `__init__` で弾くが、本関数は公開関数で
    # `fusion.py` が直接呼ぶ予定なので、そちらの経路にも同じ番人を置く。
    angle = validate_angle_deg(angle_deg)
    pairs = np.asarray(adjacency)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"adjacency must have shape (E, 2), got {pairs.shape}")
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    normals, zero_area = face_normals(vertices, faces)
    usable = ~(zero_area[pairs[:, 0]] | zero_area[pairs[:, 1]])
    return usable & (dihedral_angles(normals, pairs) <= angle)
