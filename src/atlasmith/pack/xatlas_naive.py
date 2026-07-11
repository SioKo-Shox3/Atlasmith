"""xatlas による素朴 UV 再展開+アトラスパッキング(internal・Phase 1 暫定境界)。

`_naive_unwrap_and_pack` は `MeshData` の幾何(頂点+面)から xatlas で新しい UV を
生成し、新旧の面対応表 `face_map` を明示構築して返す。焼き直し(bake)自体はここでは
行わない — 再展開の結果(新 UV レイアウトの `MeshData` と面対応)を返すだけ。

依存方向(横断規約): `pack → types (+xatlas)`。trimesh/PIL/bake/io は import しない。

面対応の契約(裁定3):
- 戻り値 `face_map (M_new,) int64` は「新面 i → 旧面 index」。
- 返す `MeshData.faces` は corner まで整列済み。新面 i の corner k が指す新頂点は、
  `vmapping` を経由して `old_faces[face_map[i], k]` と同一の旧頂点に一致する。よって
  新面と `old_faces[face_map]` は行・corner 整列となり、そのまま `bake_maps` の
  整列済み入力契約(裁定5)を満たす。

xatlas 実測(2026-07-12, xatlas-python / このリポジトリの pin):
- `Atlas.generate` の UV は [0,1] 正規化済み(u,v それぞれ [0,1] に収まる)。
- 面数は保存される(M_new == M_old)。**面の列挙順は保証されない**ため face_map は
  vmapping から明示構築する(順序に依存しない)。
- `vmapping (N_new,)` は「新頂点 → 元頂点 index」。シームで頂点を分割し index を振り直す
  が、面-頂点の接続は保つ。`indices (M_new,3)` は新頂点配列への index、`uvs (N_new,2)`
  は新 UV。`parametrize()` は pack_options を受け取れないので、resolution/padding 制御は
  `Atlas` クラスで行う。
"""

from __future__ import annotations

import numpy as np
import xatlas

from atlasmith.types import MeshData


def _build_face_map(
    old_faces: np.ndarray, vmapping: np.ndarray, new_faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """新面→旧面の対応表と、corner 整列済みの新面配列を構築する。

    引数:
        old_faces: 旧面 `(M, 3) int64`(旧頂点 index)。
        vmapping: 新頂点 → 旧頂点 index `(N_new,) int`。
        new_faces: xatlas の新面 `(M, 3) int`(新頂点 index)。

    戻り値:
        (face_map `(M,) int64`, aligned_faces `(M, 3) int64`)。
        aligned_faces[i, k] は新頂点 index で、`vmapping[aligned_faces[i, k]] ==
        old_faces[face_map[i], k]` が全 i, k で成立する(corner 整列・裁定3)。

    例外:
        ValueError: 面数保存の下でも新旧対応が全単射にならない/旧面に重複頂点集合が
            ある/corner を一意整列できない場合(いずれも Phase 1 のクリーン単一メッシュ
            前提では起こらないが、黙って誤った face_map を返さないため明示的に弾く)。
    """
    n_old = len(old_faces)
    # 旧面を「頂点 index 集合 → 旧面 index」で逆引きする。新面の頂点を vmapping で
    # 旧頂点 index に引き戻すと、ちょうど1つの旧面の頂点集合に一致する。位置ではなく
    # index で突き合わせるので、位置が重複する頂点(cube の 8 隅を 3 面が共有)でも
    # 面を取り違えない — これが「巻き順・座標ではなく index 由来」で堅牢な理由。
    lookup: dict[frozenset[int], int] = {}
    for fi in range(n_old):
        key = frozenset(int(x) for x in old_faces[fi])
        if len(key) != 3:
            raise ValueError(
                f"_naive_unwrap_and_pack: old face {fi} is degenerate "
                f"(repeated vertex index): {old_faces[fi].tolist()}"
            )
        if key in lookup:
            # 同一頂点集合の旧面が2つ(重複面)あると集合突き合わせで face_map を
            # 一意に決められない。単一クリーンメッシュ前提では発生しない。
            raise ValueError(
                "_naive_unwrap_and_pack: duplicate face vertex-set prevents an "
                f"unambiguous face_map (old faces {lookup[key]} and {fi})"
            )
        lookup[key] = fi

    m = len(new_faces)
    face_map = np.empty(m, dtype=np.int64)
    aligned = np.empty((m, 3), dtype=np.int64)
    claimed = np.zeros(n_old, dtype=bool)  # 各旧面が高々1回だけ使われる(全単射)検証。
    for i in range(m):
        nf = new_faces[i]
        old_vs = [int(vmapping[v]) for v in nf]  # 新 corner k → 旧頂点 index
        key = frozenset(old_vs)
        fi = lookup.get(key)
        if fi is None:
            raise ValueError(
                f"_naive_unwrap_and_pack: new face {i} maps to old vertices "
                f"{sorted(old_vs)}, which do not form an old face"
            )
        if claimed[fi]:
            raise ValueError(
                "_naive_unwrap_and_pack: two new faces map to old face "
                f"{fi}; face correspondence is not a bijection"
            )
        claimed[fi] = True
        face_map[i] = fi
        old_face = old_faces[fi]
        # corner 整列(裁定3): 出力新面の corner k には「その旧頂点が old_face[k] に
        # 一致する新頂点」を置く。xatlas は面内 corner を回転/反転し得るため、ここで
        # 旧 corner 順に揃え直す。これで aligned[i, k] は old_faces[face_map[i], k] と
        # 同一頂点(位置)を指し、全 corner で 3D 座標が厳密一致する。
        for k in range(3):
            target = int(old_face[k])
            matches = [j for j in range(3) if old_vs[j] == target]
            if len(matches) != 1:
                raise ValueError(
                    f"_naive_unwrap_and_pack: cannot align corner {k} of new face "
                    f"{i} to old face {fi} (candidate corners={matches})"
                )
            aligned[i, k] = int(nf[matches[0]])
    return face_map, aligned


def _naive_unwrap_and_pack(
    mesh: MeshData, *, resolution: int, padding_px: int
) -> tuple[MeshData, np.ndarray]:
    """xatlas で幾何から UV を再展開・パッキングする(internal)。

    引数:
        mesh: 入力メッシュ(頂点・面を使う。既存 UV/テクスチャは再展開に用いない)。
        resolution: xatlas のパッキング解像度(テクセル)。焼き先テクスチャの一辺と
            揃える。UV 自体は [0,1] 正規化で返るため、この値はチャートのスケールと
            padding の相対量にのみ効く。
        padding_px: チャート間のパディング(テクセル)。bake 側ガターと同じ値を渡し
            単一ソースで同期する(C9)。

    戻り値:
        (new_mesh, face_map)。
        new_mesh: 新 UV レイアウトの `MeshData`。`vertices`/`source_vertex` は元頂点を
            `vmapping` で複製したもの、`faces` は corner 整列済み、`uv` は新 UV
            `(N_new, 2) float32 [0,1]`、`maps` は空(焼き直しは bake が行う)。
        face_map: `(M_new,) int64` 新面 → 旧面。`mesh.faces[face_map]` が new_mesh.faces
            と行・corner 整列する(bake_maps の整列済み入力契約・裁定5)。
    """
    # xatlas は positions=float32 / indices=uint32 を要求する。
    positions = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    old_faces = np.asarray(mesh.faces, dtype=np.int64)
    faces_u32 = np.ascontiguousarray(old_faces, dtype=np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(positions, faces_u32)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)
    pack_options.padding = int(padding_px)
    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    vmapping_raw, new_faces_raw, uvs = atlas[0]
    vmapping = np.asarray(vmapping_raw, dtype=np.int64)
    new_faces = np.asarray(new_faces_raw, dtype=np.int64)
    uv_new = np.ascontiguousarray(uvs, dtype=np.float32)

    if len(new_faces) != len(old_faces):
        # 面の分割/欠落があると新旧の面対応が全単射でなくなり face_map を構築できない。
        # 独断で面を捨てず、停止して呼び出し側に知らせる(計画 UNCERTAIN 3)。
        raise ValueError(
            "_naive_unwrap_and_pack: xatlas changed the face count "
            f"({len(old_faces)} -> {len(new_faces)}); cannot build a face_map"
        )

    face_map, aligned_faces = _build_face_map(old_faces, vmapping, new_faces)

    # 新頂点の 3D 位置は元頂点を vmapping で複製したもの(シームで分割された頂点は
    # 同じ位置を共有する)。float64 のまま複製し MeshData の dtype 契約を満たす。
    new_vertices = np.asarray(mesh.vertices, dtype=np.float64)[vmapping]
    # source_vertex 合成契約(裁定6): 旧 source_vertex があれば vmapping で引き継ぎ、
    # 無ければ vmapping 自体(= 新頂点 → 元頂点)を採用する。
    if mesh.source_vertex is not None:
        new_source_vertex = np.asarray(mesh.source_vertex, dtype=np.int64)[vmapping]
    else:
        new_source_vertex = vmapping.copy()

    new_mesh = MeshData(
        vertices=new_vertices,
        faces=aligned_faces,
        uv=uv_new,
        maps={},
        source_vertex=new_source_vertex,
    )
    return new_mesh, face_map
