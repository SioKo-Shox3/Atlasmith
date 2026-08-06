"""多視点(SAM2)部位分割バックエンドの公開境界と寿命契約。

公開するのは**データ契約・Protocol・決定的バックエンド**:
`Camera` / `RenderedView` / `MeshRenderer` / `MaskProposer` /
`MultiViewSegmenter`、および SAM2 アダプタ(`sam2_masks.py`、Step 2-5)を
束ねるファクトリ `make_sam2_segmenter()`。

**`render` / `sam2_masks` をここで import しない**(計画v4 §2.1 規約2)。
どちらも第三者 GL/ML ライブラリを触る隔離モジュールで、module 直下 import が
あると `import atlasmith.segmentation.multiview` だけで重い依存が
`sys.modules` へ載る道が開く。それを機械的に守っているのが
`tests/test_import_isolation.py`。レンダラが要る側(Step 2-5 の
`make_sam2_segmenter` と GL テスト)は
`atlasmith.segmentation.multiview.render` を**関数内で**明示的に import する。
**`fusion` は numpy のみなので module 直下で import してよい。**

依存方向(計画v4 §2.1): このモジュールは numpy と `cameras` / `faceid` /
`fusion`(いずれも numpy のみ)だけに依存する。
"""

from __future__ import annotations

import logging
import warnings
from types import TracebackType
from typing import Any, Callable, NamedTuple, Protocol

import numpy as np

from atlasmith.segmentation.adjacency import validate_angle_deg
from atlasmith.segmentation.multiview import fusion
from atlasmith.segmentation.multiview.cameras import (
    DEFAULT_N_VIEWS,
    PROJECTIONS,
    Camera,
    build_cameras,
)
from atlasmith.segmentation.multiview.faceid import validate_face_count
from atlasmith.types import MeshData

# 公開契約は 6 件(2026-08-07 外部レビュー裁定)。数を固定しているのは
# `tests/test_multiview_sam2.py::test_public_symbols_are_the_six_approved_names`。
#
# **`Camera` を残す WHY**(将来また「承認外の公開シンボル」として指摘されない
# ように理由をここに置く): `MeshRenderer.render_view(camera: Camera)` の**引数型**
# であり、`MeshRenderer` は外部が実装しうる Protocol である。`Camera` を隠すと
# 注入用レンダラを書く側が引数の型を名指しできず、Protocol の公開が意味を失う。
#
# **`DEFAULT_IMAGE_SIZE` / `DEFAULT_PROJECTION` / `DEFAULT_SHADING` を外す WHY**:
# `__all__` から外すのは star-import の公開契約から外すことだけで、module 属性
# としては残るので `from atlasmith.segmentation.multiview import DEFAULT_IMAGE_SIZE`
# は従来どおり通る(`sam2_masks` と ML テストが実際にそう import している)。
# 既定値は「注入する側が参照する内部の出発点」であって、外部へ約束する型・
# 入口ではない。
__all__ = [
    "Camera",
    "MaskProposer",
    "MeshRenderer",
    "MultiViewSegmenter",
    "RenderedView",
    "make_sam2_segmenter",
]

_LOGGER = logging.getLogger(__name__)

# 既定の投影方式(2026-07-29 オーケストレーター裁定3)。**WHY perspective**:
# Step 2-1.5 spike で実測した投影はこれだけで、正射影は未実測。実測していない
# 方式を既定にしない。
DEFAULT_PROJECTION = "perspective"
# 既定のレンダ解像度と色付け(同裁定3)。**WHY texture_normal**: spike の実測で
# assigned_ratio が 0.721 → 0.980、観測辺が 3,491 → 6,986 に倍増し、追加コストは
# 実質ゼロだった(24.1s vs 25.1s は誤差)。計画v4 §2.4.3 の planner 推奨と一致する。
#
# **この 2 つが `MultiViewSegmenter.__init__` の引数ではない WHY**: レンダラは
# `renderer_factory` として注入され(§0-A 条件9)、解像度と色付けは**その factory が
# 閉じ込める**値である。segmenter 側に同名の引数を置くと、注入された factory が
# 実際に使う値と食い違っても誰にも分からない「嘘の引数」になる。値の検証も
# `_ModernglRenderer.__init__`(`MIN_IMAGE_SIZE` / `SHADING_MODES`)が唯一の
# 番人であり、ここでは複製しない。Step 2-5 の `make_sam2_segmenter` がこの 2 定数を
# 既定値として読み、factory へ焼き込む。
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_SHADING = "texture_normal"


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
    `segment()` であって、注入した側ではない(Step 2-4 で実装済み — 下の
    `MultiViewSegmenter.segment`)。
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


class MultiViewSegmenter:
    """多視点マスク投票による部位分割バックエンド(計画v4 §2.4)。

    `SegmentationBackend`(`atlasmith.segmentation`)の構造的契約を満たす。
    パイプライン:

    ```
    mesh --cameras--> Camera[V]
         --renderer_factory(mesh)--> RenderedView(face_id, color, coverage)
         --proposer--> masks (K, H, W) bool          <-- ここだけ非決定的
         --fusion 段階A--> view_segment (V, M) int32
         --fusion 段階B〜E--> labels (M,) int64
    ```

    **決定性**: マスク提案が同一なら**ビット決定的**(カメラ配置・面IDデコード・
    2パス融合はすべて RNG 不使用)。SAM2 込みでは「安定」であって「決定的」では
    ない — だから `MaskProposer` を Protocol で切り離してある(§2.2 の裁定 F)。

    **所有権/寿命(§2.1)**:

      - **`proposer` を所有する。** `close()` / `__exit__` が `proposer.close()`
        を呼ぶ。重み数百 MB を持ちうるので、使い終わったら閉じること。
      - **`renderer` は所有しない。** `renderer_factory(mesh)` で
        `segment()` 呼び出しごとに生成し、`with` で囲んで破棄する。レンダラは
        メッシュに束縛される(頂点バッファも面IDも mesh 依存)ので長寿命化できない。
        **`__enter__` / `__exit__` を呼ぶのは `segment()`** であって、
        factory を渡した側ではない(§0-A 条件9)。

    **未 `__enter__` でも `segment()` を許す(§0-A 条件10・2026-07-29 裁定2)**:

        segmenter = MultiViewSegmenter(proposer, factory)
        labels = segmenter.segment(mesh)   # OK。`with` は必須ではない

    これは `MeshRenderer` の契約(未入場の `render_view` は `RuntimeError`)と
    **意図的に非対称**である。理由が違う:

      - レンダラは `__enter__` で初めて GL リソースを作るので、未入場では
        **物理的に動けない**。
      - segmenter は `__init__` で受け取った proposer だけで**動ける**。
        `__enter__` は「proposer を閉じる責任を引き受ける」という宣言であって、
        使用開始の前提ではない。
      - 決め手は API 契約: 計画v4 §2.1 は「**`rebake` は注入された backend を
        閉じない**」と定めているので、`rebake(segmentation=make_sam2_segmenter())`
        を `with` 無しで呼ぶ経路が**現実に踏まれる**。ここを `RuntimeError` に
        すると、その公開 API が壊れる。

    **ただし `__exit__` / `close()` 後の `segment()` は `RuntimeError`** —
    proposer が閉じられており、黙って壊れた結果を返すよりは落ちるべきだから。

    **スレッド**: シングルスレッド前提(`architecture.md:29-33`)。GL コンテキストも
    torch 推論も呼び出しスレッドに束縛される。

    引数は非破壊: 渡された `MeshData` とその配列を書き換えず、新しい配列だけを返す。
    """

    __slots__ = (
        "_closed",
        "_proposer",
        "_renderer_factory",
        "angle_deg",
        "assign_ratio",
        "assigned_warn",
        "max_masks_per_view",
        "merge_threshold",
        "min_faces",
        "min_votes",
        "n_views",
        "projection",
        "visible_warn",
    )

    def __init__(
        self,
        proposer: MaskProposer,
        renderer_factory: Callable[[MeshData], MeshRenderer],
        *,
        n_views: int = DEFAULT_N_VIEWS,
        projection: str = DEFAULT_PROJECTION,
        assign_ratio: float = fusion.DEFAULT_ASSIGN_RATIO,
        min_votes: int = fusion.DEFAULT_MIN_VOTES,
        merge_threshold: float = fusion.DEFAULT_MERGE_THRESHOLD,
        angle_deg: float = fusion.DEFAULT_ANGLE_DEG,
        min_faces: int | None = None,
        visible_warn: float = fusion.DEFAULT_VISIBLE_WARN,
        assigned_warn: float = fusion.DEFAULT_ASSIGNED_WARN,
        max_masks_per_view: int = fusion.DEFAULT_MAX_MASKS_PER_VIEW,
    ) -> None:
        """パラメータを**構築時に**検証して保持する(fail-fast)。

        Args:
            proposer: マスク提案器。**所有する**(`close()` を呼ぶ)。
            renderer_factory: `mesh` から `MeshRenderer` を作る callable。
                `segment()` が `with factory(mesh) as renderer:` で使う。
                解像度・色付けはこの callable が閉じ込める(`DEFAULT_IMAGE_SIZE` /
                `DEFAULT_SHADING` の WHY コメント参照)。
            n_views: 視点数(既定 24)。
            projection: `"perspective"` / `"orthographic"`(既定は前者 — 裁定3)。
            assign_ratio: 段階A の占有率しきい値(`(0, 1]`、判定は `>=`)。
            min_votes: 観測辺とみなす最小票数(1 以上、既定 2)。**`n_views` 以下で
                なければならない**(超えるとどの辺も観測されず幾何プライアだけに
                なるため — 下のガード参照)。
            merge_threshold: パス1 の結合しきい値(`[0, 1]`、判定は `>=`)。
            angle_deg: 幾何プライアの二面角しきい値(`(0, 180]`)。
            min_faces: 小部位マージのしきい値。`None` なら `max(2, M // 100)`。
            visible_warn: 可視面率の警告しきい値。
            assigned_warn: 可視面のうち割当を得た率の警告しきい値。
            max_masks_per_view: 1 視点あたりのマスク数上限(超過は `ValueError`)。

        Raises:
            ValueError: いずれかのパラメータが範囲外・型違いのとき、または
                `min_votes > n_views`(構造的に ML が寄与できない組み合わせ)のとき。
        """
        # `n_views` / `projection` は `cameras` 側の述語と同じものを先に適用する。
        # 実際に使うのは `segment()` 内の `build_cameras` だが、構築時に落とせる
        # ものは構築時に落とす(不正な設定のまま数分のレンダを始めない)。
        if isinstance(n_views, bool) or not isinstance(n_views, (int, np.integer)):
            raise ValueError(f"n_views must be an int, got {type(n_views).__name__}")
        if int(n_views) < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}")
        if projection not in PROJECTIONS:
            raise ValueError(
                f"unknown projection {projection!r}, expected one of "
                f"{list(PROJECTIONS)}"
            )
        self.n_views: int = int(n_views)
        self.projection: str = projection
        # 融合側のパラメータは `fusion` の検証関数を通す(同じ述語を 2 箇所に
        # 書かない — `fusion` 内の各段も同じ関数で自衛している)。
        self.assign_ratio: float = fusion.validate_assign_ratio(assign_ratio)
        self.min_votes: int = fusion.validate_min_votes(min_votes)
        # **2 つのパラメータの関係も見る**(2026-07-30 反証レビュー B2): 1 つの辺が
        # 得られる票の上限は視点数なので、`min_votes > n_views` では**どの辺も
        # 観測辺になれず**、出力は必ず `DihedralSegmenter` と同一になる。ML を
        # 指定したのに幾何しか働かない設定を黙って受けない(例: 既定 `min_votes=2`
        # のまま `n_views=1` を渡す)。実行時に同じ結末へ落ちる別経路
        # (マスクが票に届かない等)は `fusion._warn_if_nothing_observed` が警告する。
        if self.min_votes > self.n_views:
            raise ValueError(
                f"min_votes={self.min_votes} exceeds n_views={self.n_views}: an edge "
                "can never collect more votes than there are views, so no edge would "
                "ever be observed and the result would be the geometric prior alone. "
                "Lower min_votes (1 disables the guard against single-view errors) or "
                "raise n_views."
            )
        self.merge_threshold: float = fusion.validate_merge_threshold(merge_threshold)
        self.angle_deg: float = validate_angle_deg(angle_deg)
        self.min_faces: int | None = fusion.validate_min_faces(min_faces)
        self.visible_warn: float = fusion.validate_warn_ratio(
            visible_warn, "visible_warn"
        )
        self.assigned_warn: float = fusion.validate_warn_ratio(
            assigned_warn, "assigned_warn"
        )
        self.max_masks_per_view: int = fusion.validate_max_masks_per_view(
            max_masks_per_view
        )
        self._proposer = proposer
        self._renderer_factory = renderer_factory
        self._closed = False

    def __enter__(self) -> MultiViewSegmenter:
        """`proposer` を閉じる責任を引き受ける。閉じ済みなら `RuntimeError`。"""
        if self._closed:
            raise RuntimeError(
                "this MultiViewSegmenter has been closed; its mask proposer is "
                "already released, so it cannot be entered again. Build a new "
                "segmenter (make_sam2_segmenter reloads the weights)."
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """`close()` を呼ぶ(例外は握り潰さない)。"""
        self.close()

    def close(self) -> None:
        """所有する `proposer` を解放する。**冪等**(2 回目以降は無操作)。

        **解放が成功してからフラグを立てる**(2026-08-07 外部レビュー指摘で
        順序を反転): 逆順(フラグ先行)だと、`proposer.close()` が**解放前**や
        **部分解放**の時点で例外を投げたとき、リソース(SAM2 の重み = GPU メモリ
        数 GB)が残ったまま segmenter が「閉じた」ことになり、**二度と解放
        できなくなる**。`MaskProposer.close()` は冪等が契約(この module の
        Protocol 定義)なので、失敗後の再呼び出しは安全であり、再試行できる
        状態を保つ方が確実に安全側に倒れる。

        例外は握り潰さず呼び出し側へ伝える(`__exit__` 経由でも同じ)。その場合
        `_closed` は False のままなので、呼び出し側は `close()` を再実行できる。
        """
        if self._closed:
            return
        self._proposer.close()
        self._closed = True

    def segment(self, mesh: MeshData) -> np.ndarray:
        """面ごとの部位ラベルを返す(視点をストリーミングして融合する)。

        視点ごとに `render -> propose -> 段階A` を回し、保持するのは
        `view_segment (V, M) int32` と `view_visible (V, M) bool` だけ
        (計画v4 §2.4.6)。画像は 1 視点ぶんしかメモリに載らない。

        Args:
            mesh: 分割対象。`vertices` / `faces` を読み、レンダラへそのまま渡す。
                書き換えない。

        Returns:
            `(M,) int64`、値は `0..P-1` の連番。面数 0 のメッシュには `(0,)` を
            返す(レンダラも生成しない)。

        Raises:
            RuntimeError: `close()` / `__exit__` の後に呼ばれたとき。
            ValueError: 面数が 24bit 面IDの上限を超えるとき、カメラが全頂点を
                収められないとき、注入されたレンダラ/提案器の出力が契約
                (shape・dtype・`coverage <=> face_id >= 0`)を破っているとき、
                または算出したラベルが契約を満たさないとき。
        """
        if self._closed:
            raise RuntimeError(
                "this MultiViewSegmenter has been closed; its mask proposer is "
                "already released so segment() cannot run. Note that entering the "
                "context manager is optional (an unentered segmenter works), but "
                "leaving it is final."
            )
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        n_faces = int(faces.shape[0])
        if n_faces == 0:
            # 面が無ければ描くものも投票するものも無い。レンダラを起こさずに返す
            # (GL コンテキストの生成は失敗しうる副作用なので、無駄には踏まない)。
            return np.zeros(0, dtype=np.int64)
        validate_face_count(n_faces)

        cameras = build_cameras(
            vertices, n_views=self.n_views, projection=self.projection
        )
        view_segment = np.full(
            (len(cameras), n_faces), fusion.UNASSIGNED, dtype=np.int32
        )
        view_visible = np.zeros((len(cameras), n_faces), dtype=bool)
        total_masks = 0
        # renderer は mesh に束縛されるので `segment()` ごとに生成・破棄する
        # (§2.1 / §0-A 条件9)。例外経路でも `__exit__` を必ず通す。
        with self._renderer_factory(mesh) as renderer:
            for camera in cameras:
                view = renderer.render_view(camera)
                context = f"view {camera.index}"
                masks = fusion.normalize_masks(
                    self._proposer.propose(view),
                    max_masks_per_view=self.max_masks_per_view,
                    context=context,
                )
                total_masks += int(masks.shape[0])
                assignment = fusion.assign_view_faces(
                    masks,
                    view.face_id,
                    view.coverage,
                    n_faces=n_faces,
                    assign_ratio=self.assign_ratio,
                    context=context,
                )
                view_segment[camera.index] = assignment.segment
                view_visible[camera.index] = assignment.visible
                _LOGGER.info(
                    "view %d/%d: %d mask(s), %d/%d faces assigned",
                    camera.index + 1,
                    len(cameras),
                    masks.shape[0],
                    int((assignment.segment >= 0).sum()),
                    n_faces,
                )
        if total_masks == 0:
            # §2.6: ML が 1 枚もマスクを出さなかったとき。結果は幾何バックエンド
            # 相当になる(全辺 votes=0 → パス2-1 のみ)ので、黙って「ML で分割
            # した」ことにしない。
            warnings.warn(
                "the mask proposer returned no masks for any of the "
                f"{len(cameras)} views, so this segmentation is the geometric "
                "prior alone (identical to DihedralSegmenter with the same "
                "angle_deg/min_faces). Check the proposer thresholds, or use "
                "`--segmenter geometric` deliberately.",
                stacklevel=2,
            )
        return fusion.fuse_view_segments(
            vertices,
            faces,
            view_segment,
            view_visible,
            angle_deg=self.angle_deg,
            min_votes=self.min_votes,
            merge_threshold=self.merge_threshold,
            min_faces=self.min_faces,
            visible_warn=self.visible_warn,
            assigned_warn=self.assigned_warn,
        )


def make_sam2_segmenter(**kwargs: Any) -> MultiViewSegmenter:
    """SAM2 自動マスク生成を結線した `MultiViewSegmenter` を作る(Step 2-5)。

    戻り値は proposer(= SAM2 の重み)を**所有する** context manager なので、
    使い終わったら `with` か `close()` で必ず閉じること:

        with make_sam2_segmenter() as segmenter:
            labels = segmenter.segment(mesh)

    パラメータはすべてキーワードで `sam2_masks.build_sam2_segmenter` へ透過する。
    主なもの(既定値と実測根拠は `sam2_masks.py` の `DEFAULT_*` を参照):

      - `model_id`(既定 `"facebook/sam2.1-hiera-large"`)/ `device`(既定は
        cuda 自動判定、cpu へ落ちるとき `UserWarning`)
      - `pred_iou_thresh` / `stability_score_thresh`(AMG の閾値 — 既定 0.5 / 0.7)
      - `crop_n_layers`: **0 のみ受理**(既定 0)。非 0 は AMG が
        `point_grids[crop_layer]` を引くのに対し、この proposer が視点ごとに
        グリッドを 1 つしか差し替えないため成立しない — 入口で `ValueError`。
      - `grid_side`(シルエット内点グリッド)/ `area_band` / `channels`
        (既定 `("sdf", "shading")` — SDF チャンネルが品質の主因)
      - `image_size` / `shading`(renderer_factory へ焼き込む —
        `DEFAULT_IMAGE_SIZE` / `DEFAULT_SHADING` の WHY コメント参照)
      - ほかは `MultiViewSegmenter.__init__` の引数(`n_views` 等)へ透過。

    **検証済みなのは既定パラメータだけ**: `n_views` / `image_size` を既定より
    下げると構築時に `UserWarning` が出る(禁止はしない)。実測では
    `n_views=8` 単独・`image_size=512` 単独のどちらでも accuracy が 0.9574 →
    0.5000(P=1)へ崩れ、幾何プライアと同値になった。結果が単一部位になった
    ときも `segment()` が `UserWarning` を出す。

    Raises:
        ModuleNotFoundError: torch / sam2 が**未導入**のとき。メッセージが
            `uv sync --extra ml` / `pip install "atlasmith[ml]"` /
            `--segmenter geometric` の 3 経路を提示する。**不在だけがこの型**で
            あり、CLI の既定フォールバックはこれだけを条件にする。
        ImportError: torch / sam2 は導入済みだが import が失敗したとき(ABI 不整合・
            推移的依存の欠落 = 壊れたインストール)。別アルゴリズムでの代替は
            行わず伝播する(`sam2_masks._is_absent` の WHY)。
        ValueError: パラメータが契約外のとき(`crop_n_layers != 0` を含む)、または
            **未知のキーワード名**が渡されたとき。名前の検疫は重みロードより前に
            走る(`sam2_masks._validate_segmenter_kwargs`)。

    **WHY 本体が `sam2_masks.py` で、ここは委譲だけか**: `sam2_masks` は隔離
    モジュール2(計画v4 §2.1)。module 直下で import すると
    `import atlasmith.segmentation.multiview` だけで隔離ファイルが読み込まれ、
    「重い依存に触るコードは要るときだけ触る」境界が崩れる(torch 自体は
    遅延 import でも、境界の規約はファイル単位で守る — 冒頭 docstring と
    `tests/test_import_isolation.py` 参照)。だから import は**この関数の中**。
    """
    from atlasmith.segmentation.multiview import sam2_masks

    return sam2_masks.build_sam2_segmenter(**kwargs)
