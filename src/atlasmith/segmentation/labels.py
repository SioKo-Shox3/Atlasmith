"""ラベル契約検証・union-find・小部位マージ・正規化 relabel(numpy のみ)。

面ラベル `(M,) int64` を扱う共有部品を集めたモジュール。

**`geometric.py` から分離してある WHY**: Step 2-4 の `multiview/fusion.py` が
**同じ union-find・同じ小部位マージ・同じ正規化 relabel** を再利用する
(計画v4 §2.3 末尾)。融合段(段階B/C)は視点間の投票結果を面ラベルへ畳み込む
最後に本モジュールの3つをそのまま通すので、幾何バックエンド側に閉じ込めない。

依存方向(計画v4 §2.1): `segmentation` 配下は **numpy のみ**。
"""

from __future__ import annotations

import heapq

import numpy as np


class UnionFind:
    """経路圧縮つき union-find(scipy を持ち込まないための最小実装)。

    **union by rank/size を使わず、常に小さい根を親にする WHY**: 根が必ず成分内の
    最小 index になるため、`union` を呼ぶ順序に依らず結果がビット単位で一意に
    決まる(計画v2 §2.1 手順3 の「決定的」要求)。木の高さは経路圧縮だけで実用上
    十分に抑えられる。
    """

    __slots__ = ("_parent",)

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")
        self._parent = list(range(size))

    def __len__(self) -> int:
        return len(self._parent)

    def find(self, item: int) -> int:
        """`item` の属する成分の根(= 成分内の最小 index)を返す。"""
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:  # 経路圧縮
            parent[item], item = root, parent[item]
        return root

    def union(self, a: int, b: int) -> bool:
        """`a` と `b` を同一成分にする。実際に併合したときだけ True。"""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        low, high = (root_a, root_b) if root_a < root_b else (root_b, root_a)
        self._parent[high] = low
        return True

    def roots(self) -> np.ndarray:
        """全要素の根を `(size,) int64` で返す。"""
        return np.array(
            [self.find(i) for i in range(len(self._parent))], dtype=np.int64
        )


def union_find_labels(n_items: int, pairs: np.ndarray) -> np.ndarray:
    """`pairs` の各行で結ばれた要素の連結成分を求め、各要素の根を返す。

    Args:
        n_items: 要素数(面数)。
        pairs: 結合するペア `(E, 2)`。`build_face_adjacency` の戻り値をそのまま
            渡せる。空 `(0, 2)` を許容する。

    Returns:
        `(n_items,) int64`。値は成分内の最小 index(連番ではない —
        連番化は `normalize_labels` の仕事)。

    Raises:
        ValueError: `pairs` の shape が `(E, 2)` でないとき。
    """
    pair_array = np.asarray(pairs)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError(f"pairs must have shape (E, 2), got {pair_array.shape}")
    union_find = UnionFind(n_items)
    for item_a, item_b in pair_array:
        union_find.union(int(item_a), int(item_b))
    return union_find.roots()


def validate_labels(labels: np.ndarray, n_faces: int) -> int:
    """`SegmentationBackend` のラベル契約(計画v4 §2.2)を検証し、部位数 P を返す。

    契約: `(n_faces,)` の int64 ndarray で、値集合が `0..P-1` の連番(全面に
    ラベルが付き、欠番も負値も無い)。違反はすべて `ValueError`。

    面数 0 のメッシュでは P == 0 を返す(空集合で契約は自明に成立する)。
    `pack.part_pack` の入口検疫(Step 2-6)はこれに加えて `P >= 1` / `M >= 1` を
    要求するが、それは pack 側の追加要件であって Protocol の契約ではない。

    Args:
        labels: 検証するラベル配列。
        n_faces: 期待する面数 M。

    Returns:
        部位数 P。

    Raises:
        ValueError: 契約に違反しているとき(実際の型/shape/dtype/値をメッセージに含む)。
    """
    if not isinstance(labels, np.ndarray):
        raise ValueError(f"labels must be a numpy ndarray, got {type(labels).__name__}")
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape (M,), got {labels.shape}")
    if labels.shape[0] != n_faces:
        raise ValueError(
            f"labels must have one entry per face ({n_faces}), got {labels.shape[0]}"
        )
    if labels.dtype != np.int64:
        raise ValueError(f"labels must have dtype int64, got {labels.dtype}")
    unique = np.unique(labels)
    n_parts = int(unique.shape[0])
    if not np.array_equal(unique, np.arange(n_parts, dtype=np.int64)):
        raise ValueError(
            f"labels must be consecutive 0..P-1, got {n_parts} distinct values "
            f"with min={unique.min()} max={unique.max()}"
        )
    return n_parts


def merge_small_parts(
    labels: np.ndarray, adjacency: np.ndarray, min_faces: int
) -> np.ndarray:
    """面数 `min_faces` 未満の部位を、共有境界辺数が最大の隣接部位へ反復マージする。

    計画v2 §2.1 手順4 の規約をそのまま実装する:

    - **マージ先のラベルが存続する**(吸収された側のラベル値は消える)。小さい方の
      ラベルへ寄せるのではない。
    - 決定的順序: 面数が最小の部位から処理し、同数ならラベル値の小さい方から。
      マージ先の tie-break(共有境界辺数が同数)もラベル値の小さい方。
    - 隣接部位を持たない孤立部位は `min_faces` 未満でもそのまま残す。

    **`adjacency` にはカット前の全 manifold 隣接を渡すこと**: 二面角でカットした
    後の隣接だけを渡すと、部位は定義上その隣接の連結成分そのものなので、異なる
    部位どうしがけっして隣接せずマージが一度も起きない。

    **実装(逐次セマンティクスは保ったまま増分更新する)**: 上の規約は「1回に1部位ずつ
    吸収する逐次手続き」として定義されているが、毎回ラベル配列を舐め直すと
    O(部位数 x 面数) になる。実測(2026-07-27、当開発機): 断片化した M=20480 /
    マージ前 P=14152 のメッシュで **14.6 秒**、`segment()` 全体の 99.9%。AI 生成
    メッシュ(100k〜1M 面)では実用性が壊れる水準だった。そこで

      - 部位ごとの面数と**部位間の共有辺数**を最初に1回だけ構築し、
      - 候補の取り出しを `(面数, ラベル)` をキーにしたヒープ(遅延無効化つき)で行い、
      - マージのたびに触るのは**吸収される側の隣接だけ**(小部位なので
        `境界辺数 <= 3 * min_faces` で上から抑えられる)、
      - ラベル配列への書き戻しは**最後に1回**

    に変更した。**結果は逐次版とビット単位で同一**(ヒープの取り出し順が
    `(面数昇順, ラベル昇順)` と一致し、マージ先の選び方も同一のため)。等価性は
    `tests/test_segmentation_labels.py` の素朴リファレンスとの無作為突き合わせで
    固定してある。

    Args:
        labels: 面ラベル `(M,)`。値は任意の整数でよい(連番でなくてよい)。
            **書き換えない**(コピーを返す)。
        adjacency: `build_face_adjacency` の戻り値 `(E, 2)`。
        min_faces: この面数**未満**の部位をマージ対象とする。`1` 以下なら無操作。

    Returns:
        `(M,) int64` のマージ後ラベル(連番とは限らない — `normalize_labels` へ渡す)。

    Raises:
        ValueError: `labels` が1次元でない、または `adjacency` の shape が
            `(E, 2)` でないとき。
    """
    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError(f"labels must have shape (M,), got {label_array.shape}")
    pair_array = np.asarray(adjacency)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError(f"adjacency must have shape (E, 2), got {pair_array.shape}")

    merged = label_array.astype(np.int64, copy=True)
    if min_faces <= 1 or merged.size == 0:
        return merged

    parts, part_of_face, counts = np.unique(
        merged, return_inverse=True, return_counts=True
    )
    part_of_face = part_of_face.reshape(merged.shape[0])  # numpy 2.0/2.1 の shape 差
    n_parts = int(parts.shape[0])
    if n_parts <= 1:
        return merged

    neighbours = _build_part_neighbours(part_of_face, pair_array, n_parts)
    sizes = counts.tolist()
    absorbed_into = list(range(n_parts))

    # `parts` は昇順なので **部位 index の昇順 = ラベル値の昇順**。よって
    # `(面数, 部位 index)` タプルの最小取り出しが規約の「面数昇順 → ラベル昇順」に
    # 一致する。面数が変わった部位は新しいエントリを push し、古いエントリは
    # pop 時に `sizes` と突き合わせて捨てる(遅延無効化)。
    pending = [
        (sizes[part], part) for part in range(n_parts) if sizes[part] < min_faces
    ]
    heapq.heapify(pending)
    while pending:
        size, candidate = heapq.heappop(pending)
        if size != sizes[candidate] or absorbed_into[candidate] != candidate:
            continue  # 面数が変わった / 既に吸収済みの古いエントリ。
        if size >= min_faces:
            # 吸収して閾値を超えた部位。面数は増える一方なので二度と候補にならない。
            continue
        near = neighbours[candidate]
        if not near:
            # 孤立部位はそのまま残す。マージで隣接が新しく生まれることはないので、
            # 一度隣接ゼロになった部位を再検査する必要はない。
            continue
        # 共有辺数が最大 → 同数なら部位 index(= ラベル値)が最小。
        destination = min(near.items(), key=lambda item: (-item[1], item[0]))[0]
        _absorb_part(neighbours, candidate, destination)
        absorbed_into[candidate] = destination
        sizes[destination] += sizes[candidate]
        if sizes[destination] < min_faces:
            heapq.heappush(pending, (sizes[destination], destination))

    survivor = _resolve_absorption_chains(absorbed_into)
    return parts[survivor[part_of_face]]


def _build_part_neighbours(
    part_of_face: np.ndarray, adjacency: np.ndarray, n_parts: int
) -> list[dict[int, int]]:
    """部位間の共有境界辺数 `{部位 index: 辺数}` を部位ごとに構築する(無向・両方向)。"""
    part_a = part_of_face[adjacency[:, 0]]
    part_b = part_of_face[adjacency[:, 1]]
    crossing = part_a != part_b
    neighbours: list[dict[int, int]] = [{} for _ in range(n_parts)]
    if not crossing.any():
        return neighbours
    low = np.minimum(part_a[crossing], part_b[crossing]).astype(np.int64)
    high = np.maximum(part_a[crossing], part_b[crossing]).astype(np.int64)
    # 無向ペアを1つの整数キーへ畳んでから C 側で集計する(Python ループを辺の
    # 本数 E ではなく *相異なる部位ペア数* まで減らすため)。n_parts <= M なので
    # n_parts^2 は int64 に収まる(M=1e6 でも 1e12 << 9.2e18)。
    pair_key, weights = np.unique(low * n_parts + high, return_counts=True)
    for key, weight in zip(pair_key.tolist(), weights.tolist()):
        first, second = divmod(key, n_parts)
        neighbours[first][second] = weight
        neighbours[second][first] = weight
    return neighbours


def _absorb_part(
    neighbours: list[dict[int, int]], candidate: int, destination: int
) -> None:
    """`candidate` の隣接を `destination` へ移し替える(`candidate` は以後空になる)。

    触るのは `candidate` の隣接だけ。`candidate` は必ず面数 `< min_faces` の
    小部位なので、その境界辺数は `3 * min_faces` 未満で抑えられる。
    """
    near = neighbours[candidate]
    into = neighbours[destination]
    near.pop(destination, None)  # 2部位を跨いでいた辺は内部辺になる
    into.pop(candidate, None)
    for other, weight in near.items():
        other_map = neighbours[other]
        del other_map[candidate]
        other_map[destination] = other_map.get(destination, 0) + weight
        into[other] = into.get(other, 0) + weight
    near.clear()


def _resolve_absorption_chains(absorbed_into: list[int]) -> np.ndarray:
    """`部位 index -> 最終的に生き残った部位 index` を経路圧縮で解決する。

    吸収先自身がさらに吸収されることがあるため連鎖する。

    **閉路ガード(fail-loud)**: 正しく構築された `absorbed_into` は森なので連鎖は
    必ず `n_parts` ホップ以内で止まる。閉路があるとこのループは**永遠に回る** —
    テストはタイムアウトするまで無出力になり、原因の特定が極めて難しい(実測: 変異
    注入で 7 分間ハング)。`labels.py` は Step 2-4 の `multiview/fusion.py` が
    再利用する共有部品なので、将来の実装者が「静かなハング」ではなく
    「大声の失敗」を受け取れるようにする。
    """
    n_parts = len(absorbed_into)
    survivor = np.empty(n_parts, dtype=np.int64)
    for part in range(n_parts):
        root = part
        hops = 0
        while absorbed_into[root] != root:
            root = absorbed_into[root]
            hops += 1
            if hops > n_parts:
                raise ValueError(
                    "merge bookkeeping is corrupt: the absorption chain starting at "
                    f"part {part} did not terminate within {n_parts} hops, so "
                    "`absorbed_into` contains a cycle. This is an internal invariant "
                    "violation of merge_small_parts (each absorbed part must point at "
                    "a still-living part), not a rejected input."
                )
        node = part
        while absorbed_into[node] != root:  # 経路圧縮
            absorbed_into[node], node = root, absorbed_into[node]
        survivor[part] = root
    return survivor


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """ラベルを「所属する最小の面 index」の昇順で `0..P-1` に振り直す。

    計画v2 §2.1 手順5。面の並びだけで順序が決まるので、上流のラベル値
    (union-find の根や融合の投票 id)に依らず出力が一意になる。

    Args:
        labels: 面ラベル `(M,)`。任意の整数値でよい。**書き換えない**。

    Returns:
        `(M,) int64`、値は `0..P-1` の連番。

    Raises:
        ValueError: `labels` が1次元でないとき。
    """
    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError(f"labels must have shape (M,), got {label_array.shape}")
    n_faces = label_array.shape[0]
    if n_faces == 0:
        return np.zeros(0, dtype=np.int64)

    values, inverse = np.unique(label_array, return_inverse=True)
    inverse = inverse.reshape(n_faces)
    # 各ラベルが最初に現れる面 index。`np.unique(return_index=True)` の「最初の
    # 出現」保証に依らず、最小値集約で明示的に求める。
    first_face = np.full(values.shape[0], n_faces, dtype=np.int64)
    np.minimum.at(first_face, inverse, np.arange(n_faces, dtype=np.int64))
    order = np.argsort(first_face, kind="stable")
    rank = np.empty(values.shape[0], dtype=np.int64)
    rank[order] = np.arange(values.shape[0], dtype=np.int64)
    return rank[inverse].astype(np.int64, copy=False)
