"""部位分割層(パイプライン①)の公開境界。

`SegmentationBackend` 契約と、幾何バックエンド `DihedralSegmenter` を公開する。

依存方向(計画v4 §2.1): **このパッケージのソースが import するのは numpy と
`atlasmith.types` だけ**である。trimesh / xatlas / PIL / torch / moderngl / sam2 と
`atlasmith.io` / `atlasmith.pack` / `atlasmith.bake` は、module 直下・関数内・
`TYPE_CHECKING` ブロックのいずれでも書かない。重い ML/GL 依存は Step 2-3 以降に
追加される `segmentation.multiview` の隔離2ファイル(`render.py` /
`sam2_masks.py`)の**関数内**だけに閉じ込める。

**ただし「ソースが import しない」と「実行時に sys.modules へ載らない」は別物**
(実測 2026-07-27):

    >>> import atlasmith.segmentation
    >>> "trimesh" in sys.modules, "xatlas" in sys.modules, "PIL" in sys.modules
    (True, True, True)

サブモジュールの import は Python の規則により親パッケージ `atlasmith/__init__.py`
の実行を強制し、そこが `io` / `pack` を eager import しているためである
(`src/atlasmith/__init__.py`。Phase 2 では変更しない)。したがって本パッケージ
だけを import しても trimesh/xatlas/PIL の読み込みコストは避けられない。

計画v4 §2.1 の機械的ゲートはこの事実と整合している: **`sys.modules` 検査(a)が
見るのは `torch` / `moderngl` / `sam2` の3つだけ**(これらは載らない)で、
trimesh/xatlas/PIL は **AST 検査(b)** — つまり「ソースに書かれていないこと」——
で担保する対象である。ここを取り違えると「numpy しか読み込まれない」という
誤った期待を持つので明記する。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from atlasmith.segmentation.geometric import DihedralSegmenter
from atlasmith.types import MeshData

__all__ = ["DihedralSegmenter", "SegmentationBackend"]


class SegmentationBackend(Protocol):
    """メッシュを面ごとの部位ラベルへ写すバックエンドの構造的契約。

    契約(計画v4 §2.2):
      `segment(mesh)` の戻り値は `(M,) int64`(M = `mesh.faces` の行数)。値は
      `0..P-1` の連番(P = 部位数)で、**全面にラベルが付く**(欠損も -1 も無い)。
      検疫は `pack.part_pack` の入口(Step 2-6)でも行うので、自前バックエンドの
      出力もそこで弾かれる。

    **決定性は「バックエンドの性質」であり、この Protocol の要求ではない。**
      WHY(2026-07-27 ユーザー裁定 F): 主バックエンドとなる `MultiViewSegmenter`
      (Step 2-4)のパイプラインで非決定的なのは **SAM2 のマスク提案 1 段だけ**で、
      その前後 — カメラ配置・レンダリング・面IDデコード・2パス融合 — はすべて
      決定的である。「決定的であること」を契約に含めると、実用上たった1段の
      非決定要素のために主バックエンドが自分の Protocol を満たせなくなる。
      よって決定性は契約から外し、**各実装が自分の docstring で宣言する**
      (`DihedralSegmenter` はビット決定的、`MultiViewSegmenter` はマスク提案が
      同一なら決定的)。

    Protocol なので継承は不要 — `segment` を持つ任意のオブジェクトが適合する
    (レジストリも ABC 階層も作らない、YAGNI 維持)。
    """

    def segment(self, mesh: MeshData) -> np.ndarray:
        """`mesh` の面ごとの部位ラベル `(M,) int64`(`0..P-1` の連番)を返す。"""
        ...
