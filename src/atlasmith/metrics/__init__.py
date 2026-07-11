"""メトリクス層: 焼き直し品質の定量評価。

numpy と標準ライブラリのみに依存する(横断規約の依存方向: `bake/metrics → numpy のみ`。
trimesh/xatlas/PIL は import しない)。公開 API は `masked_psnr` の1関数
(再展開後オラクルの合否判定に用いる)。
"""

from atlasmith.metrics.quality import masked_psnr

__all__ = ["masked_psnr"]
