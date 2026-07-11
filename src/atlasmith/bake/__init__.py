"""焼き直し層: 旧 UV/旧テクスチャ → 新 UV への画素転写(ガター膨張・複数マップ対応)。

numpy と標準ライブラリのみに依存する(横断規約の依存方向: `bake/metrics → numpy のみ`。
trimesh/xatlas/PIL は import しない)。公開 API は `bake_maps` と結果型 `BakeResult`。
"""

from atlasmith.bake.transfer import BakeResult, bake_maps

__all__ = ["BakeResult", "bake_maps"]
