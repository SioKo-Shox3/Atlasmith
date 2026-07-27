"""幾何バックエンド `DihedralSegmenter` — 位置weld + 二面角しきい値 + 連結成分。

計画v2 §2.1(= 計画v4 §2.3 が「無改訂で流用」と定めた一次資料)の設計をそのまま
実装する。weld・面隣接・二面角は `adjacency.py`、契約検証・union-find・小部位
マージ・正規化 relabel は `labels.py` に置いてあり、本モジュールはそれらを
「二面角しきい値でカットして連結成分を取る」順序に並べるだけ。

依存方向(計画v4 §2.1): `segmentation` 配下は **numpy のみ**。
"""

from __future__ import annotations

import numpy as np

from atlasmith.segmentation.adjacency import (
    build_face_adjacency,
    smooth_edge_mask,
    validate_angle_deg,
    weld_vertices,
)
from atlasmith.segmentation.labels import (
    merge_small_parts,
    normalize_labels,
    union_find_labels,
    validate_labels,
)
from atlasmith.types import MeshData

# `min_faces=None` のときの自動値 `max(2, M // _MIN_FACES_AUTO_DIVISOR)` の除数
# (計画v2 §2.1 の初期提案)。「全体の 1% 未満の部位は雑音」という経験則。
_MIN_FACES_AUTO_DIVISOR = 100
_MIN_FACES_AUTO_FLOOR = 2


class DihedralSegmenter:
    """二面角しきい値で面をクラスタリングする幾何部位分割バックエンド。

    `SegmentationBackend`(`atlasmith.segmentation.__init__`)の構造的契約を満たす。

    手順(計画v2 §2.1):
      0. 位置weld — 面の辺を weld 代表 index で表す。
      1. 面隣接 — ちょうど2面が共有する辺のみ隣接(非多様体辺は必ずカット)。
      2. 二面角 — 隣接ペアの法線どうしの角。零面積面が絡むペアは常にカット。
      3. union-find — `角度 <= angle_deg` の隣接だけを union。
      4. 小部位マージ — 面数 `min_faces` 未満の部位を隣接部位へ吸収。
      5. 正規化 relabel — `0..P-1` の連番へ。

    **本バックエンドはビット決定的である**: RNG を一切使わず、`np.lexsort` と
    union-find の決定的な tie-break だけで結果が決まる。同一入力に対して同一の
    ラベル配列を返す。(決定性は `SegmentationBackend` Protocol の要求ではなく
    実装ごとの性質 — 2026-07-27 ユーザー裁定 F。だからこそ性質を持つ側が明記する。)

    **期待挙動の正直な記述**(計画v2 §2.1 末尾): 滑らかな単一ブロブでは P=1 に退化し
    Phase 1 と同等になる(優雅な劣化)。価値が出るのは (a) 複数連結成分を持つ AI
    出力、(b) 鋭い折れ目を持つハードサーフェス。意味論的な部位分割は ML
    バックエンド(`MultiViewSegmenter`)の領分。

    引数は非破壊: 渡された `MeshData` とその配列を一切書き換えず、新しい配列だけを返す。
    """

    __slots__ = ("angle_deg", "min_faces")

    def __init__(
        self, *, angle_deg: float = 60.0, min_faces: int | None = None
    ) -> None:
        """パラメータを検証して保持する。

        Args:
            angle_deg: この角度**以下**の二面角を持つ隣接だけを同一部位に繋ぐ
                しきい値(度)。有限かつ `0 < angle_deg <= 180`。
            min_faces: この面数**未満**の部位を隣接部位へ吸収する。`None` なら
                面数 M から `max(2, M // 100)` を自動で決める。

        Raises:
            ValueError: `angle_deg` が非有限・範囲外、または `min_faces` が
                整数でない・1 未満のとき(構築時に落とす = fail-fast)。
        """
        # `smooth_edge_mask` と同じ番人を使う(判定を2箇所に書かない)。
        angle = validate_angle_deg(angle_deg)
        if min_faces is not None:
            if not isinstance(min_faces, (int, np.integer)):
                raise ValueError(
                    f"min_faces must be None or an int, got {type(min_faces).__name__}"
                )
            if int(min_faces) < 1:
                raise ValueError(f"min_faces must be >= 1, got {min_faces}")
            min_faces = int(min_faces)
        self.angle_deg: float = angle
        self.min_faces: int | None = min_faces

    def segment(self, mesh: MeshData) -> np.ndarray:
        """面ごとの部位ラベルを返す。

        Args:
            mesh: 分割対象。`vertices` と `faces` のみ参照する(UV/テクスチャは
                幾何分割に関与しない)。書き換えない。

        Returns:
            `(M,) int64`、値は `0..P-1` の連番。面数 0 のメッシュには `(0,)` を返す。

        Raises:
            ValueError: `mesh` の shape 契約が壊れているとき(`adjacency` 側の
                検証由来)、または算出したラベルが自身の契約を満たさないとき。
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        n_faces = faces.shape[0]
        if n_faces == 0:
            return np.zeros(0, dtype=np.int64)

        weld_map = weld_vertices(vertices)
        adjacency = build_face_adjacency(faces, weld_map)
        # 手順2〜3: 「二面角がしきい値以下 かつ 両側とも零面積でない」辺だけを繋ぐ。
        # 2つの規約は組でしか正しくないので `smooth_edge_mask` に閉じ込めてある
        # (Step 2-4 の fusion.py 段階B も同じ判定を再利用する)。
        keep = smooth_edge_mask(vertices, faces, adjacency, angle_deg=self.angle_deg)

        roots = union_find_labels(n_faces, adjacency[keep])
        # マージにはカット前の全隣接を渡す(カット後の隣接では異なる部位どうしが
        # 定義上隣接しない — `merge_small_parts` の docstring 参照)。
        merged = merge_small_parts(roots, adjacency, self._effective_min_faces(n_faces))
        labels = normalize_labels(merged)

        # production 不変条件: 契約違反のラベルを黙って下流(pack)へ流さない。
        validate_labels(labels, n_faces)
        return labels

    def _effective_min_faces(self, n_faces: int) -> int:
        """`min_faces=None` のときの自動値を解決する。"""
        if self.min_faces is not None:
            return self.min_faces
        return max(_MIN_FACES_AUTO_FLOOR, n_faces // _MIN_FACES_AUTO_DIVISOR)
