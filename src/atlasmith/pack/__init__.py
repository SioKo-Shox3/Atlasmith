"""パッキング層: xatlas による UV 再展開+アトラスパッキング。

展開③とパッキング④を一体化した内部ヘルパを 2 つ提供する(横断規約の依存方向:
`pack → types (+xatlas)`。trimesh/PIL/bake/io/segmentation は import しない):

- `_naive_unwrap_and_pack`(Phase 1)— メッシュ全体を 1 回で展開・パックする。
- `_part_unwrap_and_pack`(Phase 2)— 面ラベルで分けた部位ごとに展開し、単一
  アトラスへ一括パックする。「各 UV アイランド ⊆ 単一部位」を保証し、アトラスの
  実寸法 `AtlasDims` も返す。

**どちらも公開 API ではない**(先頭アンダースコアが internal を示す)。展開③+
パッキング④の一体化は依然として暫定であり、安定境界は Phase 3 で「unwrap 済み UV を
受けるパッキング」として再設計する前提(計画 v3 C11)。それまでは internal に留め、
`rebake` からのみ使う。
"""

# 内部利用者(atlasmith.rebake / オラクルテスト)が `from atlasmith.pack import
# _naive_unwrap_and_pack` で参照するための re-export。redundant alias 形式は
# 「意図的な再輸出」を ruff に明示する(公開 API 契約には含めない)。
from atlasmith.pack.part_pack import (
    AtlasDims as AtlasDims,
)
from atlasmith.pack.part_pack import (
    _part_unwrap_and_pack as _part_unwrap_and_pack,
)
from atlasmith.pack.xatlas_naive import (
    _naive_unwrap_and_pack as _naive_unwrap_and_pack,
)

# 公開シンボルは無い(3 つとも internal。`AtlasDims` に先頭アンダースコアが無いのは
# 「型名」だからで、`__all__` に載せない = 公開 API 契約には含めない)。
__all__: list[str] = []
