"""パッキング層: xatlas による UV 再展開+アトラスパッキング。

Phase 1 では展開③とパッキング④を一体化した内部ヘルパ `_naive_unwrap_and_pack`
のみを提供する(横断規約の依存方向: `pack → types (+xatlas)`。trimesh/PIL/bake/io
は import しない)。

**公開 API ではない**(先頭アンダースコアが internal を示す)。展開③+パッキング④の
一体化は暫定であり、安定境界は Phase 2/3 で「unwrap 済み UV を受けるパッキング」として
再設計する前提(計画 v3 C11)。それまでは internal に留め、`rebake` からのみ使う。
"""

# 内部利用者(atlasmith.rebake / オラクルテスト)が `from atlasmith.pack import
# _naive_unwrap_and_pack` で参照するための re-export。redundant alias 形式は
# 「意図的な再輸出」を ruff に明示する(公開 API 契約には含めない)。
from atlasmith.pack.xatlas_naive import (
    _naive_unwrap_and_pack as _naive_unwrap_and_pack,
)

# 公開シンボルは無い(_naive_unwrap_and_pack は internal)。
__all__: list[str] = []
