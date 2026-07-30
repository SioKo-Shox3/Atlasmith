"""多視点(SAM2)部位分割バックエンドの公開境界と寿命契約。

この時点(Step 2-3)で公開するのは**データ契約と Protocol だけ**:
`Camera` / `RenderedView` / `MeshRenderer` / `MaskProposer`。融合(`fusion.py`)
と SAM2 アダプタ(`sam2_masks.py`)、それらを束ねる `MultiViewSegmenter` は
Step 2-4 / 2-5 で追加される。

**`render` をここで import しない**(計画v4 §2.1 規約2)。`render.py` は
moderngl を触る隔離モジュールで、module 直下 import があると
`import atlasmith.segmentation.multiview` だけで GL 依存が `sys.modules` へ
載ってしまう。それを機械的に守っているのが `tests/test_import_isolation.py`。
レンダラが要る側(Step 2-4 の `MultiViewSegmenter` と GL テスト)は
`atlasmith.segmentation.multiview.render` を**関数内で**明示的に import する。

依存方向(計画v4 §2.1): このモジュールは numpy と `cameras` だけに依存する。
"""

from __future__ import annotations

from types import TracebackType
from typing import NamedTuple, Protocol

import numpy as np

from atlasmith.segmentation.multiview.cameras import Camera

__all__ = ["Camera", "MaskProposer", "MeshRenderer", "RenderedView"]


class RenderedView(NamedTuple):
    """1視点ぶんのレンダ結果(計画v4 §2.1 の完全契約)。

    フィールド:
        face_id: `(H, W) int32`。値は面コード、背景は **-1**。
            **row 0 = 画面上端**(`glReadPixels` は左下原点なので、レンダラが
            読み戻し後に行方向を反転してからここへ入れる)。
        color: `(H, W, 3) uint8`。row 順は `face_id` と同一。
        coverage: `(H, W) bool`。前景 = True。**alpha アタッチメント由来で
            色値からは独立**(真っ黒な前景画素を背景と誤認しない)。

    3 つの配列は同じ `(H, W)` を共有し、`coverage <=> (face_id >= 0)` が
    全画素で成立する(レンダラが production 不変条件として検査する)。
    """

    face_id: np.ndarray
    color: np.ndarray
    coverage: np.ndarray


class MeshRenderer(Protocol):
    """メッシュを視点ごとの `RenderedView` へ写すレンダラの構造的契約。

    **寿命契約(計画v4 §2.1・`architecture.md:24-33` への追補)**: 実装は GL
    コンテキストのような長寿命 GPU リソースを持ちうるので、context manager と
    して生成/破棄の正規ルートを定める。

      - `__enter__` でリソースを確保し、`__exit__` で**生成の逆順に**解放する。
      - `__enter__` 前 / `__exit__` 後の `render_view` は `RuntimeError`
        (未入場・入場中・退場後の 3 状態を実装が明示的に管理する
        — 2026-07-29 オーケストレーター裁定2)。
      - 同一インスタンスの `__enter__` 再入も `RuntimeError`。

    **注入口(Step 2-4 の想定 — ここでは Protocol の側から契約だけ書いておく)**:
    Step 2-4 の `MultiViewSegmenter` は
    `renderer_factory: Callable[[MeshData], MeshRenderer]` としてこの実装を
    受け取る(§0-A 条件9)。renderer はメッシュに依存するので長寿命化せず、
    **`segment()` 呼び出しごとに `with` で生成・破棄する** —
    つまり注入された renderer の `__enter__` / `__exit__` を呼ぶのは
    `segment()` であって、注入した側ではない。`MultiViewSegmenter` 自身は
    Step 2-4 で追加する(この時点では存在しない)。
    """

    def __enter__(self) -> MeshRenderer:
        """リソースを確保して自分自身を返す。再入は `RuntimeError`。"""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """生成の逆順にリソースを解放する(例外を握り潰さない)。"""
        ...

    def render_view(self, camera: Camera) -> RenderedView:
        """`camera` から見た `RenderedView` を返す(入場中のみ有効)。"""
        ...


class MaskProposer(Protocol):
    """`RenderedView` からマスク候補を提案する層の構造的契約(Step 2-5 で実装)。

    **パイプライン中で非決定的なのはこの 1 段だけ**(計画v4 §2.4)。前後
    (カメラ・レンダ・面IDデコード・視点間融合)はすべて決定的なので、この
    Protocol を差し替えれば GPU も重みも無しに下流を厳密ゲートにかけられる。

    寿命: モデル重み等を保持しうるので `close()` を持つ。所有者は
    `MultiViewSegmenter`(Step 2-4)であり、そこが `close()` を呼ぶ。
    """

    def propose(self, view: RenderedView) -> np.ndarray:
        """`(K, H, W) bool` のマスク候補を返す(K は視点ごとに変わる)。"""
        ...

    def close(self) -> None:
        """保持しているリソースを解放する(冪等であること)。"""
        ...
