"""部位ごと xatlas 展開+単一アトラス一括パック(internal・Phase 2)。

`_part_unwrap_and_pack` は面ラベル `labels` で分けた部位ごとに独立したサブメッシュを
1 つの `xatlas.Atlas` へ `add_mesh` し、`generate` を **1 回だけ**呼ぶ。こうすると
全部位が同一アトラス空間へ正しい相対テクセル密度でパックされる。

**「各 UV アイランド ⊆ 単一部位」がどの機構で保証されるか**(2026-08-05 の反証レビュー
指摘 B1 を受けて明示する): 保証しているのは xatlas の振る舞いではなく**連結時の頂点
index レンジ設計**である。部位 p の新頂点は `atlas[p]` の `vmapping` から作られ、連結の
際に「それまでの部位の累積新頂点数」`vertex_offset` を足してから `faces` に書く。よって
部位 p の新頂点 index はちょうど `[offset_p, offset_p + N_p)` に閉じ、**異なる部位の面が
新頂点 index を共有することは構成上あり得ない**(実測: cube 2 分割 `{0:[0,11],
1:[12,23]}` / capped_cylinder `{0:[0,191], 1:[192,287], 2:[288,383]}` — 全て disjoint)。
「頂点共有の連結成分」で定義したアイランドは、したがって部位をまたげない。
`_check_island_part_consistency` はこの**構成上の性質に対する自己検査**であり、xatlas に
対する防波堤ではない(詳細は同関数の docstring)。

戻り値の `faces` / `face_map` の契約は `_naive_unwrap_and_pack` と**同一**であり、
`rebake` の既存整列コード `mesh.faces[face_map]` がそのまま適用できる。第3要素の
`AtlasDims` だけが追加分で、xatlas が実際に確保したアトラス寸法(テクセル)を返す。

依存方向(横断規約 `Docs/agent-guide/architecture.md:38`): `pack → types (+xatlas)`。
trimesh/PIL/bake/io/**segmentation** は import しない。ラベル契約の検疫を
`segmentation.labels.validate_labels` に委ねず本モジュール内で行っているのはこのため
(検疫のメッセージも pack 固有の対処 — `--seg-angle` / `--granularity naive` — を
案内する必要があり、Protocol 側の汎用メッセージでは文脈が合わない)。

xatlas 実測(2026-07-27 / xatlas 0.0.11 / このリポジトリの pin。3 部位・面数 2/8/18・
`resolution=64` / `padding=2`):

- `atlas.width, atlas.height = 78, 60` — **アトラスは正方形とは限らず、`resolution` を
  超えることもある**。`PackOptions` に正方形を強制する選択肢は無い(属性は bilinear /
  blockAlign / bruteForce / create_image / max_chart_size / padding / resolution /
  rotate_charts / rotate_charts_to_axis / texels_per_unit のみ)。
- UV は **per-axis 正規化**: `u = pixel_x / width`, `v = pixel_y / height`。証拠として
  「UV 辺長 / 3D 辺長」の比は生 UV では軸によって 0.231〜0.300 とばらつく(spread
  2.3e-01)のに対し、`uv * (width, height)` では全部位・全辺で 18.0 テクセル/単位に
  揃う(spread 2.7e-06)。
- `atlas[p]` は `add_mesh` の**追加順**に対応する(面数 2/8/18 がその順で返る)。
- `atlas.atlas_count == 1`(= 単一アトラスに収まった)を直接取得できる。

アトラス寸法規約(計画v2 §2.3a・承認事項 D の案 (b) = 等方スケール+正方形出力):
`D = max(width, height)` として UV を `u' = u·width/D`, `v' = v·height/D` に写す。
per-axis 正規化を一度アトラス画素空間へ戻してから一律 `1/D` で割るので、チャートは
正方形 `[0,1]²` の `[0, width/D]×[0, height/D]` 部分に**等方密度で**収まる。出力
テクスチャは従来どおり正方形のままでよく、公開契約が変わらない。
`D > texture_size` のときのガター反復数の調整と警告は `rebake` の責務(計画v4 §2.5)。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import xatlas

from atlasmith.pack.xatlas_naive import _build_face_map
from atlasmith.types import MeshData

# 部位数 P の上限(`Docs/agent-guide/technique-part-segmentation.md` の既定値)。
# 部位ごとに `add_mesh` するので P はそのまま xatlas のメッシュ数になる。過分割は
# パック効率とチャート数を無意味に悪化させるため、黙って続行せず対処を案内して止める。
_MAX_PARTS = 1024

# UV 値域検査の許容(float32 丸めとアトラス画素→正規化の往復ぶん)。
_UV_BOUND_TOL = 1e-5


class AtlasDims(NamedTuple):
    """xatlas が実際に確保したアトラスの寸法(テクセル)。

    `width != height` があり得る(上のモジュール docstring の実測参照)。焼き先
    テクスチャの一辺 `texture_size` とは別物で、両者の比がガター反復数の調整に効く。
    """

    width: int
    height: int


def _validate_part_labels(labels: np.ndarray, n_faces: int) -> int:
    """`_part_unwrap_and_pack` 入口の labels 検疫を行い、部位数 P を返す。

    `SegmentationBackend` の契約(shape/dtype/`0..P-1` 連番)に加えて、**pack 側の
    入口要件**である `M >= 1` / `P >= 1` / `P <= _MAX_PARTS` を要求する。自前
    バックエンドの出力もここが防波堤になる(計画v2 §2.3 手順0)。

    Args:
        labels: 検疫するラベル配列。
        n_faces: メッシュの面数 M。

    Returns:
        部位数 P。

    Raises:
        ValueError: 契約違反・入口要件違反(実際の shape/dtype/値をメッセージに含む)。
    """
    if not isinstance(labels, np.ndarray):
        raise ValueError(
            "_part_unwrap_and_pack: labels must be a numpy ndarray, got "
            f"{type(labels).__name__}"
        )
    if labels.ndim != 1:
        raise ValueError(
            f"_part_unwrap_and_pack: labels must have shape (M,), got {labels.shape}"
        )
    if labels.shape[0] != n_faces:
        raise ValueError(
            "_part_unwrap_and_pack: labels must have one entry per face "
            f"({n_faces}), got {labels.shape[0]}"
        )
    if labels.dtype != np.int64:
        raise ValueError(
            f"_part_unwrap_and_pack: labels must have dtype int64, got {labels.dtype}"
        )
    unique = np.unique(labels)
    n_parts = int(unique.shape[0])
    if not np.array_equal(unique, np.arange(n_parts, dtype=np.int64)):
        raise ValueError(
            "_part_unwrap_and_pack: labels must be consecutive 0..P-1, got "
            f"{n_parts} distinct values with min={unique.min()} max={unique.max()}"
        )
    # M == 0 と P == 0 は契約の下では同値(空配列の値集合は空)だが、どちらが破れても
    # 同じ「展開する面が無い」状態なので 1 つのガードで両方を名指しして落とす。
    if n_faces < 1 or n_parts < 1:
        raise ValueError(
            "_part_unwrap_and_pack: the part path requires at least one face and "
            f"one part, got M={n_faces} P={n_parts}"
        )
    if n_parts > _MAX_PARTS:
        raise ValueError(
            f"_part_unwrap_and_pack: too many parts (P={n_parts} > {_MAX_PARTS}). "
            "Raise --seg-angle or --seg-min-faces to merge parts, or fall back to "
            "--granularity naive. (Not falling back silently: an over-segmented "
            "atlas wastes texels and charts.)"
        )
    return n_parts


def _face_islands(faces: np.ndarray) -> np.ndarray:
    """面を「頂点 index の共有」で連結成分(= UV アイランド)へ分ける。

    新メッシュの面配列に対して使う。xatlas はシームで頂点を分割するので、頂点を共有
    する面の連結成分がそのまま 1 つのチャート(UV アイランド)になる。

    Args:
        faces: 面 `(M, K)`(K は corner 数。三角形なら 3)。

    Returns:
        `(M,) int64`。値は**その成分に属する最小の面 index**(連番ではない)。

    Raises:
        ValueError: `faces` が 2 次元でないとき。
    """
    face_array = np.asarray(faces, dtype=np.int64)
    if face_array.ndim != 2:
        raise ValueError(
            f"_face_islands: faces must be 2-D (M, K), got shape {face_array.shape}"
        )
    n_faces = int(face_array.shape[0])
    parent = list(range(n_faces))

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:  # 経路圧縮
            parent[item], item = root, parent[item]
        return root

    if n_faces > 0 and face_array.shape[1] > 0:
        # 頂点で corner をグループ化してから、同一頂点を共有する面を順に union する。
        # 「常に小さい根を親にする」ので結果は union の順序に依らず一意になる
        # (`segmentation.labels.UnionFind` と同じ決定性の理由)。
        corner_face = np.repeat(np.arange(n_faces, dtype=np.int64), face_array.shape[1])
        corner_vertex = face_array.ravel()
        order = np.argsort(corner_vertex, kind="stable")
        sorted_vertex = corner_vertex[order]
        sorted_face = corner_face[order].tolist()
        boundaries = np.flatnonzero(sorted_vertex[1:] != sorted_vertex[:-1]) + 1
        starts = np.concatenate(([0], boundaries)).tolist()
        ends = np.concatenate((boundaries, [sorted_vertex.shape[0]])).tolist()
        for start, end in zip(starts, ends):
            anchor = sorted_face[start]
            for index in range(start + 1, end):
                root_a, root_b = find(anchor), find(sorted_face[index])
                if root_a != root_b:
                    low, high = (
                        (root_a, root_b) if root_a < root_b else (root_b, root_a)
                    )
                    parent[high] = low
    return np.array([find(index) for index in range(n_faces)], dtype=np.int64)


def _check_island_part_consistency(faces: np.ndarray, face_labels: np.ndarray) -> None:
    """各 UV アイランドが単一部位に属することを検査する(production 不変条件)。

    これが部位経路の存在理由そのもの(「各アイランド ⊆ 単一部位」)なので、テストだけ
    でなく production 経路にも置く。混在アトラスを黙って返さない。

    **この検査が何に対する防波堤で、何に対してはそうでないか**(2026-08-05 反証レビュー
    B1): これは **xatlas に対する防波堤ではない**。`_part_unwrap_and_pack` は部位ごとの
    新頂点 index を `[offset_p, offset_p + N_p)` という互いに素なレンジへ配置してから
    連結する(module docstring 参照)ので、**xatlas が何を返そうと**異なる部位の面が
    新頂点を共有することは構成上あり得ず、本検査は実データでは発火しない。守っている
    のは「部位ごとの index レンジが disjoint である」という**構成上の性質**そのもので、
    発火しうる唯一の経路は連結オフセット算術の破壊である(実証: `vertex_offset` の加算を
    落とすミューテーションで実 fixture が本 `ValueError` を出す)。したがって
    「実行して 0 件だった」ことは xatlas の振る舞いの証拠にはならない。

    Args:
        faces: 新メッシュの面 `(M, K)`(新頂点 index)。
        face_labels: 新面ごとの部位ラベル `(M,)`(= `labels[face_map]`)。

    Raises:
        ValueError: 面数が食い違うとき、または 2 部位以上にまたがるアイランドが
            あるとき(違反アイランドの面 index と部位の集合をメッセージに含む)。
    """
    face_array = np.asarray(faces, dtype=np.int64)
    label_array = np.asarray(face_labels, dtype=np.int64)
    if label_array.ndim != 1 or label_array.shape[0] != face_array.shape[0]:
        raise ValueError(
            "_check_island_part_consistency: face_labels must have shape (M,) with "
            f"M={face_array.shape[0]}, got {label_array.shape}"
        )
    if face_array.shape[0] == 0:
        return

    islands = _face_islands(face_array)
    _roots, inverse = np.unique(islands, return_inverse=True)
    inverse = inverse.reshape(islands.shape)  # numpy 2.0/2.1 の shape 差
    n_islands = int(_roots.shape[0])
    lowest = np.full(n_islands, np.iinfo(np.int64).max, dtype=np.int64)
    highest = np.full(n_islands, np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(lowest, inverse, label_array)
    np.maximum.at(highest, inverse, label_array)
    mixed = np.flatnonzero(lowest != highest)
    if mixed.shape[0] == 0:
        return

    first = int(mixed[0])
    offending = np.flatnonzero(inverse == first)
    parts = np.unique(label_array[offending])
    raise ValueError(
        "_check_island_part_consistency: UV island rooted at new face "
        f"{int(islands[offending[0]])} spans {parts.shape[0]} parts "
        f"{parts.tolist()}; every UV island must lie inside a single part "
        f"({int(mixed.shape[0])} island(s) violate this; offending faces "
        f"{offending[:8].tolist()}{'...' if offending.shape[0] > 8 else ''})"
    )


def _check_part_face_structure(
    faces_part: np.ndarray, face_ids: np.ndarray, part: int
) -> None:
    """部位内の面が `_build_face_map` の前提を満たすかを**大域面 index で**検査する。

    前提は 2 つ: (a) 面の 3 corner が相異なる頂点 index を指す(非退化)、(b) 同一の
    頂点 index 集合を持つ面が部位内に 2 つ無い(重複面)。どちらも `_build_face_map`
    自身が検出できるが、あちらは部位ローカルに reindex された配列しか見ていないため
    `old face 0` のように**ローカル index で誤称**してしまい、利用者が実際の面を特定
    できない(2026-08-05 反証レビュー N1 の実測)。同じ条件をここで先に検査して、
    大域面 index と大域頂点 index で報告する。

    **重複面の検査を部位内に閉じる WHY**: 大域で検査すると、別々の部位に分かれた重複面
    (現状は偶発的に通る)を新たに拒否することになり、挙動が変わる。BL-7 裁定は
    「対応は約束しないが、部位が分かれた場合に通ることはあり得る」なので、判定範囲は
    `_build_face_map` と同じ「部位内」に揃える。

    Args:
        faces_part: 部位の面 `(M_p, 3)`(**大域**頂点 index)。
        face_ids: 部位の面の大域 index `(M_p,)`。
        part: 部位 index(メッセージ用)。

    Raises:
        ValueError: 退化面・重複面があるとき(大域 index を含む)。
    """
    seen: dict[frozenset[int], int] = {}
    for local_index, corners in enumerate(faces_part.tolist()):
        global_face = int(face_ids[local_index])
        key = frozenset(int(vertex) for vertex in corners)
        if len(key) != 3:
            raise ValueError(
                f"_part_unwrap_and_pack: part {part}: old face {global_face} is "
                f"degenerate (repeated vertex index): {corners}. Face and vertex "
                "indices are global (indices into mesh.faces / mesh.vertices)."
            )
        if key in seen:
            raise ValueError(
                f"_part_unwrap_and_pack: part {part}: duplicate face vertex-set "
                "prevents an unambiguous face_map (old faces "
                f"{seen[key]} and {global_face}). Face indices are global "
                "(indices into mesh.faces)."
            )
        seen[key] = global_face


def _check_face_map_bijection(face_map: np.ndarray, n_old_faces: int) -> None:
    """大域 face_map が旧面への全単射であることを検査する(production 不変条件)。

    Args:
        face_map: 新面 → 旧面 `(M_new,) int64`。
        n_old_faces: 旧面数 M_old。

    Raises:
        ValueError: 面数が保存されていない/範囲外の旧面 index がある/同じ旧面を
            2 つ以上の新面が指しているとき(= 黙った面欠落・重複)。
    """
    mapping = np.asarray(face_map, dtype=np.int64)
    if mapping.ndim != 1 or mapping.shape[0] != n_old_faces:
        raise ValueError(
            "_part_unwrap_and_pack: face count is not preserved (M_old="
            f"{n_old_faces}, face_map shape={mapping.shape}); the part path must "
            "not drop or split faces"
        )
    if n_old_faces == 0:
        return
    if int(mapping.min()) < 0 or int(mapping.max()) >= n_old_faces:
        raise ValueError(
            "_part_unwrap_and_pack: face_map contains out-of-range old face indices "
            f"(min={int(mapping.min())} max={int(mapping.max())}, M_old={n_old_faces})"
        )
    seen = np.zeros(n_old_faces, dtype=bool)
    seen[mapping] = True
    if int(seen.sum()) != n_old_faces:
        missing = np.flatnonzero(~seen)
        raise ValueError(
            "_part_unwrap_and_pack: face_map is not a bijection; "
            f"{missing.shape[0]} old face(s) are unreachable, e.g. "
            f"{missing[:8].tolist()}"
        )


def _check_atlas_dims(width: int, height: int, atlas_count: int) -> AtlasDims:
    """アトラス寸法規約(計画v2 §2.3a 手順4)を検査して `AtlasDims` を返す。

    Args:
        width: `atlas.width`。
        height: `atlas.height`。
        atlas_count: `atlas.atlas_count`。

    Returns:
        検証済みの `AtlasDims`。

    Raises:
        ValueError: 寸法が非正、またはアトラスが 1 枚に収まらなかったとき
            (= UV が複数アトラスに分かれ、単一テクスチャへ焼けない)。
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            "_part_unwrap_and_pack: xatlas produced an empty atlas "
            f"(width={width}, height={height})"
        )
    if atlas_count != 1:
        raise ValueError(
            f"_part_unwrap_and_pack: xatlas packed into {atlas_count} atlases "
            f"(width={width}, height={height}); the part path requires a single "
            "atlas because the bake target is one texture. Raise `resolution` or "
            "reduce the number of parts."
        )
    return AtlasDims(width=int(width), height=int(height))


def _check_uv_bounds(uv: np.ndarray, dims: AtlasDims) -> None:
    """UV が規約 (b) の `[0, w/D]×[0, h/D]` に収まっていることを検査する。

    Args:
        uv: 等方スケール後の UV `(N, 2)`。
        dims: アトラス実寸法。

    Raises:
        ValueError: 値域を外れる UV があるとき(実測の min/max を含む)。
    """
    largest = float(max(dims.width, dims.height))
    limit = np.array([dims.width / largest, dims.height / largest], dtype=np.float64)
    lowest = np.asarray(uv, dtype=np.float64).min(axis=0)
    highest = np.asarray(uv, dtype=np.float64).max(axis=0)
    if (lowest < -_UV_BOUND_TOL).any() or (highest > limit + _UV_BOUND_TOL).any():
        raise ValueError(
            "_part_unwrap_and_pack: packed UV leaves the atlas rectangle "
            f"[0, {limit[0]:.6f}] x [0, {limit[1]:.6f}] "
            f"(observed min={lowest.tolist()} max={highest.tolist()}, "
            f"atlas={dims.width}x{dims.height})"
        )


def _part_unwrap_and_pack(
    mesh: MeshData, labels: np.ndarray, *, resolution: int, padding_px: int
) -> tuple[MeshData, np.ndarray, AtlasDims]:
    """部位ごとに xatlas 展開し、単一アトラスへ一括パックする(internal)。

    Args:
        mesh: 入力メッシュ(頂点・面を使う。既存 UV/テクスチャは再展開に用いない)。
            **書き換えない**。
        labels: 面ごとの部位ラベル `(M,) int64`、値は `0..P-1` の連番。
        resolution: xatlas のパッキング解像度(テクセル)。実際のアトラス寸法は
            これを超えることがある(戻り値 `AtlasDims` を参照)。
        padding_px: チャート間のパディング(テクセル。アトラス画素基準)。

    Returns:
        `(new_mesh, face_map, dims)`。

        - `new_mesh`: 新 UV レイアウトの `MeshData`。`vertices`/`source_vertex` は
          元頂点を部位ごとの `vmapping` で複製したもの、`faces` は corner 整列済み
          (契約は `_naive_unwrap_and_pack` と同一)、`uv` は等方スケール後の
          `(N_new, 2) float32`、`maps` は空。
        - `face_map`: `(M_new,) int64` 新面 → 旧面。`mesh.faces[face_map]` が
          `new_mesh.faces` と行・corner 整列する(`bake_maps` の整列済み入力契約)。
        - `dims`: xatlas が実際に確保したアトラス寸法。

    Raises:
        ValueError: 入口検疫に反する `labels`、xatlas が部位内で面数を変えた、
            `_build_face_map` が対応を作れない(部位 index を付けて再送出)、または
            production 不変条件(face_map 全単射 / アイランド–部位整合 / アトラス
            寸法規約 / UV 値域)が破れたとき。
    """
    old_faces = np.asarray(mesh.faces, dtype=np.int64)
    n_old_faces = int(old_faces.shape[0])
    n_parts = _validate_part_labels(labels, n_old_faces)
    label_array = np.asarray(labels, dtype=np.int64)

    old_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    # xatlas は positions=float32 / indices=uint32 を要求する。
    positions = np.ascontiguousarray(old_vertices, dtype=np.float32)
    old_source_vertex = (
        None
        if mesh.source_vertex is None
        else np.asarray(mesh.source_vertex, dtype=np.int64)
    )

    atlas = xatlas.Atlas()
    part_face_ids: list[np.ndarray] = []
    part_vertex_ids: list[np.ndarray] = []
    part_local_faces: list[np.ndarray] = []
    for part in range(n_parts):
        # 昇順の面 index = 決定的な面来歴(provenance)。
        face_ids = np.flatnonzero(label_array == part).astype(np.int64, copy=False)
        faces_part = old_faces[face_ids]
        vertex_ids = np.unique(faces_part)  # 昇順・重複なし
        # `vertex_ids` が昇順なので searchsorted がそのまま局所 reindex になる。
        # 部位ごとに長さ N の逆引き表を作ると O(P*N) になるのでそうしない。
        local_faces = np.searchsorted(vertex_ids, faces_part).astype(np.int64)
        atlas.add_mesh(
            np.ascontiguousarray(positions[vertex_ids]),
            np.ascontiguousarray(local_faces, dtype=np.uint32),
        )
        part_face_ids.append(face_ids)
        part_vertex_ids.append(vertex_ids)
        part_local_faces.append(local_faces)

    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)
    pack_options.padding = int(padding_px)
    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    dims = _check_atlas_dims(
        int(atlas.width), int(atlas.height), int(atlas.atlas_count)
    )
    # 規約 (b): per-axis 正規化をアトラス画素空間へ戻し、一律 1/D で等方スケール。
    largest = float(max(dims.width, dims.height))
    uv_scale = np.array([dims.width, dims.height], dtype=np.float64) / largest

    face_map_chunks: list[np.ndarray] = []
    faces_chunks: list[np.ndarray] = []
    vertices_chunks: list[np.ndarray] = []
    source_chunks: list[np.ndarray] = []
    uv_chunks: list[np.ndarray] = []
    vertex_offset = 0
    for part in range(n_parts):
        # probe 実測: `atlas[p]` は `add_mesh` の追加順に対応する。
        vmapping_raw, new_faces_raw, uvs_raw = atlas[part]
        vmapping = np.asarray(vmapping_raw, dtype=np.int64)
        new_faces = np.asarray(new_faces_raw, dtype=np.int64)
        face_ids = part_face_ids[part]
        if new_faces.shape[0] != face_ids.shape[0]:
            # 面の分割/欠落があると面対応が全単射でなくなる。独断で面を捨てず止める
            # (`xatlas_naive.py:153-159` と同型の fail-loud を部位単位で行う)。
            raise ValueError(
                "_part_unwrap_and_pack: xatlas changed the face count of part "
                f"{part} ({face_ids.shape[0]} -> {new_faces.shape[0]}); "
                "cannot build a face_map"
            )
        # `_build_face_map` が検出できる 2 条件(退化面・部位内の重複面)を先に大域
        # index で報告する。あちらのメッセージは部位ローカル index を「old face N」と
        # 呼ぶため、利用者が問題の面を特定できない(反証レビュー N1)。
        _check_part_face_structure(old_faces[face_ids], face_ids, part)
        try:
            local_face_map, local_aligned = _build_face_map(
                part_local_faces[part], vmapping, new_faces
            )
        except ValueError as error:
            # ここへ来るのは上の 2 条件以外(xatlas が既存の面へ写らない新面を返した
            # 等)。**元メッセージはそのまま前置される** — その中の `old face` /
            # `old vertices` は**部位ローカル**の index なので、そう明記した上で部位
            # index と大域面 index の対応を添える(原因は `from error` で連鎖)。
            raise ValueError(
                f"_part_unwrap_and_pack: part {part}: {error} "
                "(indices in the wrapped message are local to the part; this part "
                f"covers global faces {face_ids[:8].tolist()}"
                f"{'...' if face_ids.shape[0] > 8 else ''})"
            ) from error

        face_map_chunks.append(face_ids[local_face_map])
        # 局所の新頂点 index を、それまでの部位の累積新頂点数だけずらしてから連結する。
        faces_chunks.append(local_aligned + vertex_offset)
        origin = part_vertex_ids[part][vmapping]  # 新頂点 → 元頂点 index
        vertices_chunks.append(old_vertices[origin])
        # source_vertex 合成契約(`xatlas_naive.py:167-171`)を部位経由で適用する。
        source_chunks.append(
            origin.copy() if old_source_vertex is None else old_source_vertex[origin]
        )
        uv_chunks.append(np.asarray(uvs_raw, dtype=np.float64) * uv_scale)
        vertex_offset += int(vmapping.shape[0])

    face_map = np.concatenate(face_map_chunks).astype(np.int64, copy=False)
    new_faces_all = np.concatenate(faces_chunks).astype(np.int64, copy=False)
    new_vertices = np.concatenate(vertices_chunks).astype(np.float64, copy=False)
    new_source_vertex = np.concatenate(source_chunks).astype(np.int64, copy=False)
    new_uv = np.concatenate(uv_chunks)

    # production 不変条件(計画v2 §2.3 手順6)— 違反を黙って下流へ流さない。
    _check_face_map_bijection(face_map, n_old_faces)
    _check_uv_bounds(new_uv, dims)
    _check_island_part_consistency(new_faces_all, label_array[face_map])

    new_mesh = MeshData(
        vertices=new_vertices,
        faces=new_faces_all,
        uv=np.ascontiguousarray(new_uv, dtype=np.float32),
        maps={},
        source_vertex=new_source_vertex,
    )
    return new_mesh, face_map, dims
