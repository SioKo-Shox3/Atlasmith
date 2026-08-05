"""`sam2_masks.py` と `make_sam2_segmenter` のゲート(計画v4 §5 Step 2-5)。

2 層構成:

  - **非 ML(CI でも必ず実行)**: torch/sam2 を偽物に差し替え、未導入エラー経路・
    CPU 警告・パラメータ検証・寿命契約・SDF 配管(裁定A)・面積帯/チャンネル
    規約(裁定B)を GPU 無しで検査する。
  - **ML(`@pytest.mark.gl @pytest.mark.ml`)**: 実重み(既定 sam2.1-hiera-large)
    での疎通ゲート(5/6 — 品質主張はしない)、主品質ゲート(7)、負の対照(8)、
    安定性(9)、既定パラメータ経路(10)、決定的層の非退行(11)。ゲート 7〜10 は
    **module スコープ fixture で SAM2 の segment を 2 回だけ実行して共有する**
    (裁定F — 1 回数分かかるため)。閾値は 2026-08-01 オーケストレーター裁定E:
    accuracy >= 0.85 / Δ >= 0.30 / 安定性 >= 0.98 / P <= min(T*3, 8)。
    **未達なら閾値を下げずに落とす**(停止条件 — 計画v4 Step 2-5 リスク欄)。
"""

from __future__ import annotations

import sys
import time
import types
import warnings
from typing import Any, Callable

import numpy as np
import pytest

from atlasmith.segmentation.labels import validate_labels
from atlasmith.segmentation.multiview import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SHADING,
    MeshRenderer,
    MultiViewSegmenter,
    RenderedView,
    make_sam2_segmenter,
    sam2_masks,
)
from atlasmith.segmentation.multiview.thickness import thickness_to_image
from atlasmith.types import MeshData

# 未導入エラーのメッセージに揃って含まれるべき 3 経路(計画v4 Step 2-5 ゲート1)。
_INSTALL_PHRASES = (
    "uv sync --extra ml",
    'pip install "atlasmith[ml]"',
    "--segmenter geometric",
)


# ---------------------------------------------------------------------------
# 偽物(非 ML テスト用): CI に torch/sam2 が無くても全経路を踏めるようにする
# ---------------------------------------------------------------------------


class _FakeGenerator:
    """`SAM2AutomaticMaskGenerator` の使用面(point_grids 属性 + generate)を再現。

    `records` に仕込んだ辞書列(`segmentation` / `point_coords`)をそのまま返す。
    受け取った画像は `generate_images` に記録する(チャンネル規約の検査用)。
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.point_grids = kwargs.get("point_grids")
        self.records: list[dict[str, Any]] = []
        self.generate_images: list[np.ndarray] = []

    def generate(self, image: np.ndarray) -> list[dict[str, Any]]:
        self.generate_images.append(np.asarray(image).copy())
        return list(self.records)


class _FakeAmgFactory:
    """`_import_sam2()` の戻り値(クラス)の代役。構築呼び出しを記録する。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.generators: list[_FakeGenerator] = []

    def from_pretrained(self, model_id: str, **kwargs: Any) -> _FakeGenerator:
        self.calls.append((model_id, dict(kwargs)))
        generator = _FakeGenerator(**kwargs)
        self.generators.append(generator)
        return generator


def _fake_torch(cuda_available: bool) -> Any:
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available)
    )


@pytest.fixture
def fake_ml(monkeypatch: pytest.MonkeyPatch) -> _FakeAmgFactory:
    """torch(cuda あり)と sam2 を偽物へ差し替える。"""
    factory = _FakeAmgFactory()
    monkeypatch.setattr(sam2_masks, "_import_torch", lambda: _fake_torch(True))
    monkeypatch.setattr(sam2_masks, "_import_sam2", lambda: factory)
    return factory


def _make_view(face_id: np.ndarray, color: np.ndarray | None = None) -> RenderedView:
    """`face_id` から契約どおりの `RenderedView` を組む(coverage は導出)。"""
    ids = np.asarray(face_id, dtype=np.int32)
    if color is None:
        color = np.zeros((*ids.shape, 3), dtype=np.uint8)
    return RenderedView(face_id=ids, color=color, coverage=ids >= 0)


class _CountingProposer:
    """K=0 を返しつつ `set_face_thickness` / `close` を数えるスタブ。"""

    def __init__(self) -> None:
        self.close_count = 0
        self.thickness_history: list[np.ndarray] = []

    def set_face_thickness(self, thickness01: np.ndarray) -> None:
        self.thickness_history.append(np.asarray(thickness01).copy())

    def propose(self, view: RenderedView) -> np.ndarray:
        height, width = view.face_id.shape
        return np.zeros((0, height, width), dtype=bool)

    def close(self) -> None:
        self.close_count += 1


# ---------------------------------------------------------------------------
# 非 ML 1: 未導入エラー経路(ゲート1)
# ---------------------------------------------------------------------------


def test_missing_torch_raises_an_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch 不在で `ImportError` + 3 つの行動可能な経路が案内される。

    `sys.modules` へ `None` を差し込むと `import torch` が `ImportError` になる
    (`tests/test_multiview_render.py` の moderngl 版と同じ手口 — 導入済みの
    開発機でも未導入経路を踏める)。
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError) as excinfo:
        make_sam2_segmenter()
    for phrase in _INSTALL_PHRASES:
        assert phrase in str(excinfo.value)


def test_missing_sam2_raises_an_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch はあるが sam2 が無い環境でも同じ 3 経路が案内される。"""
    monkeypatch.setattr(sam2_masks, "_import_torch", lambda: _fake_torch(True))
    monkeypatch.setitem(sys.modules, "sam2", None)
    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", None)
    with pytest.raises(ImportError) as excinfo:
        make_sam2_segmenter()
    for phrase in _INSTALL_PHRASES:
        assert phrase in str(excinfo.value)


# ---------------------------------------------------------------------------
# 非 ML 2: CPU 警告(ゲート2)
# ---------------------------------------------------------------------------


def test_cpu_fallback_warns(
    fake_ml: _FakeAmgFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUDA が見えないとき、自動解決は cpu + `UserWarning`(計画v4 §2.4.4)。"""
    monkeypatch.setattr(sam2_masks, "_import_torch", lambda: _fake_torch(False))
    with pytest.warns(UserWarning, match="CPU"):
        segmenter = make_sam2_segmenter()
    segmenter.close()
    assert fake_ml.calls[0][1]["device"] == "cpu"


def test_explicit_cpu_device_does_not_warn(
    fake_ml: _FakeAmgFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """明示的な `device="cpu"` は意図的な選択なので警告しない。"""
    monkeypatch.setattr(sam2_masks, "_import_torch", lambda: _fake_torch(False))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        segmenter = make_sam2_segmenter(device="cpu")
    segmenter.close()
    assert fake_ml.calls[0][1]["device"] == "cpu"


# ---------------------------------------------------------------------------
# 非 ML 3: パラメータ検証(ゲート3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"model_id": ""},
        {"model_id": 5},
        {"pred_iou_thresh": -0.1},
        {"pred_iou_thresh": 1.5},
        {"stability_score_thresh": 2.0},
        {"crop_n_layers": -1},
        {"crop_n_layers": 0.5},
        # B-1(2026-08-03 反証レビュー): 1 以上は AMG 内部で IndexError になるので
        # 入口で拒否する(実重みで「数分後の IndexError」を実測済み)。
        {"crop_n_layers": 1},
        {"crop_n_layers": 2},
        {"grid_side": 0},
        {"grid_side": True},
        {"area_band": (0.5, 0.5)},
        {"area_band": (-0.1, 0.5)},
        {"area_band": (0.9, 0.5)},
        {"area_band": (0.1,)},
        {"area_band": "wide"},
        {"channels": ()},
        {"channels": ("rgb",)},
        {"channels": ("sdf", "sdf")},
        {"channels": "sdf"},
        {"image_size": 7},
        {"image_size": 64.0},
        {"shading": "phong"},
        {"device": ""},
        {"n_views": 0},
    ],
    ids=lambda kwargs: "-".join(f"{k}={v!r}" for k, v in kwargs.items()),
)
def test_bad_parameters_raise_value_error(
    fake_ml: _FakeAmgFactory, bad_kwargs: dict[str, Any]
) -> None:
    """契約外のパラメータは `ValueError`(`area_band` は `0 <= lo < hi`)。"""
    with pytest.raises(ValueError):
        make_sam2_segmenter(**bad_kwargs)


def test_parameter_errors_fire_before_the_weights_load(
    fake_ml: _FakeAmgFactory,
) -> None:
    """モデル固有パラメータの検証は `from_pretrained` より先(fail-fast)。

    数百 MB の重み DL の後に typo の `ValueError` を出さない、という設計意図の
    実証。`n_views` 等の融合側パラメータは親の検証(構築の最終段)なので対象外。
    """
    for bad_kwargs in ({"area_band": (0.9, 0.1)}, {"shading": "phong"}):
        with pytest.raises(ValueError):
            make_sam2_segmenter(**bad_kwargs)
    assert fake_ml.calls == []


def test_crop_n_layers_is_rejected_at_the_door(fake_ml: _FakeAmgFactory) -> None:
    """★ B-1(2026-08-03 反証レビュー): `crop_n_layers != 0` は入口で `ValueError`。

    旧実装は 0 以上を受理していたが、`Sam2MaskProposer.propose` は視点ごとに
    `point_grids` を**1 つだけ**差し替える一方、AMG は crop レイヤ index で
    `point_grids[i]` を引く。実重みでは**重みロードとレンダを終えた数分後に**
    `IndexError: list index out of range` で落ちた。

    どう壊れたら落ちるか: 検証が「0 以上」に戻った瞬間、この test が通らなくなる
    (= 数分後の IndexError が復活する)。メッセージが理由と唯一の合法値を
    述べていることも固定する — 偽 generator ではこの破綻を検出できないので、
    **入口の拒否そのものがゲート**である。
    """
    with pytest.raises(ValueError, match="crop_n_layers must be 0") as excinfo:
        make_sam2_segmenter(crop_n_layers=1)
    message = str(excinfo.value)
    assert "point_grids" in message
    assert "IndexError" in message
    assert fake_ml.calls == []  # 重みをロードする前に落ちている


def test_amg_kwargs_pass_through_to_the_generator(fake_ml: _FakeAmgFactory) -> None:
    """裁定D の kwargs(閾値・crop)が generator 構築へそのまま届く。

    **`crop_n_layers` に 1 を渡さない WHY**(B-1): 偽 generator は
    `point_grids[crop_layer_idx]` を引かないので、実重みなら数分後に IndexError
    になる値でも「generator に届いた」と緑になってしまう。**このテストが検出
    できない値をこのテストで正当化しない** — 非 0 の拒否は
    `test_crop_n_layers_is_rejected_at_the_door` が受け持つ。
    """
    segmenter = make_sam2_segmenter(
        model_id="facebook/sam2-hiera-tiny",
        pred_iou_thresh=0.25,
        stability_score_thresh=0.6,
        crop_n_layers=0,
        grid_side=8,
    )
    segmenter.close()
    model_id, kwargs = fake_ml.calls[0]
    assert model_id == "facebook/sam2-hiera-tiny"
    assert kwargs["pred_iou_thresh"] == 0.25
    assert kwargs["stability_score_thresh"] == 0.6
    assert kwargs["crop_n_layers"] == 0
    assert kwargs["points_per_side"] is None
    assert np.asarray(kwargs["point_grids"][0]).shape == (64, 2)


def test_bad_fusion_kwargs_still_close_the_proposer(
    fake_ml: _FakeAmgFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """親(`MultiViewSegmenter`)の検証で落ちても重み保持側はリークしない。

    どう壊れたら落ちるか: `build_sam2_segmenter` の try/except が消えると、
    構築済み proposer が開いたまま例外だけ飛ぶ。**N-3(2026-08-03 反証レビュー):
    旧版は `close()` を一切見ておらず、try/except を消しても通っていた** ので、
    ここでは `close()` が呼ばれたことと、閉じた結果 proposer が実際に使えなく
    なっていること(`propose` が `RuntimeError`)まで観測する。
    """
    closed: list[Any] = []
    real_close = sam2_masks.Sam2MaskProposer.close

    def spy_close(self: Any) -> None:
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(sam2_masks.Sam2MaskProposer, "close", spy_close)
    with pytest.raises(ValueError, match="min_votes"):
        make_sam2_segmenter(n_views=1)  # 既定 min_votes=2 > n_views=1
    assert len(fake_ml.calls) == 1  # 重みロード後に親の検証で落ちた経路
    assert len(closed) == 1, "the proposer holding the weights was never closed"
    view, _ = _foreground_view()
    with pytest.raises(RuntimeError, match="close"):
        closed[0].propose(view)


# ---------------------------------------------------------------------------
# 非 ML 3.5: 「ML が寄与しなかった」経路の告知(B-3 — 2026-08-03 反証レビュー)
#
# 実測(既定から片方だけ変更): n_views=8 単独でも image_size=512 単独でも
# accuracy 0.9574 -> 0.5000 / P=2 -> 1 に崩れ、**警告は 0 件**だった。0.5000 / P=1 は
# 幾何プライアと完全に同値(= ML 寄与ゼロ)。ここはその 2 経路の告知を固定する。
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weak_kwargs, expected_fragment",
    [
        ({"n_views": 8}, "n_views=8"),
        ({"image_size": 512}, "image_size=512"),
        ({"n_views": 8, "image_size": 512}, "n_views=8"),
    ],
    ids=["n_views", "image_size", "both"],
)
def test_below_default_parameters_warn(
    fake_ml: _FakeAmgFactory,
    weak_kwargs: dict[str, Any],
    expected_fragment: str,
) -> None:
    """★ B-3: 検証済み既定を下回る構成は**警告する**(禁止はしない)。

    どう壊れたら落ちるか: 崩壊領域の告知が消えた瞬間 — つまり「ML を選んだのに
    幾何プライアと同値の結果が黙って返る」状態に戻った瞬間に落ちる。
    """
    with pytest.warns(UserWarning, match="below its validated defaults") as captured:
        segmenter = make_sam2_segmenter(**weak_kwargs)
    segmenter.close()
    message = next(
        str(record.message)
        for record in captured
        if "below its validated defaults" in str(record.message)
    )
    assert expected_fragment in message
    # 実測値を文面に残す(「未検証」だけでは行動できない)。
    assert "0.9574" in message and "0.5000" in message


def test_default_parameters_do_not_warn(fake_ml: _FakeAmgFactory) -> None:
    """負の対照: 既定(n_views=24 / image_size=1024)は警告しない。

    これが無いと「常に警告する」実装でも上の test が通ってしまう。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        segmenter = make_sam2_segmenter()
    segmenter.close()


def test_above_default_parameters_do_not_warn(fake_ml: _FakeAmgFactory) -> None:
    """既定を**上回る**構成も警告しない(下回るときだけの告知である)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        segmenter = make_sam2_segmenter(n_views=32, image_size=2048)
    segmenter.close()


# ---------------------------------------------------------------------------
# 非 ML 4: 寿命契約(ゲート4)
# ---------------------------------------------------------------------------


def test_segmenter_lifetime_contract(
    fake_ml: _FakeAmgFactory, cube_mesh: MeshData
) -> None:
    """`__exit__` 後の `segment` は `RuntimeError`、`close()` は冪等。"""
    segmenter = make_sam2_segmenter()
    with segmenter:
        pass
    with pytest.raises(RuntimeError, match="closed"):
        segmenter.segment(cube_mesh)
    segmenter.close()
    segmenter.close()
    with pytest.raises(RuntimeError):
        with segmenter:
            pass  # pragma: no cover - 再入場は入口で落ちる


def test_proposer_is_closed_exactly_once() -> None:
    """所有権: `__exit__` がスタブ proposer の `close()` をちょうど 1 回呼ぶ。"""
    proposer = _CountingProposer()
    segmenter = sam2_masks._Sam2MultiViewSegmenter(
        proposer, lambda mesh: None, needs_thickness=False
    )
    with segmenter:
        pass
    assert proposer.close_count == 1
    segmenter.close()
    assert proposer.close_count == 1


# ---------------------------------------------------------------------------
# 非 ML 5: SDF 配管(裁定A — サブクラスの厚み受け渡し)
# ---------------------------------------------------------------------------


def _block_view_factory(
    build_block_view: Callable[..., RenderedView],
    static_renderer: Callable[..., Any],
    n_views: int,
) -> Callable[[MeshData], MeshRenderer]:
    """メッシュの面数に合わせた合成ビューを返す renderer_factory を組む。"""

    def factory(mesh: MeshData) -> MeshRenderer:
        return static_renderer([build_block_view(len(mesh.faces))] * n_views)

    return factory


def test_subclass_pushes_fresh_thickness_on_every_segment(
    cube_mesh: MeshData,
    capped_cylinder_mesh: MeshData,
    build_block_view: Callable[..., RenderedView],
    static_renderer: Callable[..., Any],
) -> None:
    """裁定A: `segment()` ごとに厚みが計算し直され proposer へ渡る(stale 防止)。

    メッシュを替えて 2 回 `segment` し、渡った厚み配列の長さがそれぞれの面数に
    追従することを見る — 1 回目の厚みが残っていたら 2 回目の長さが合わない。
    """
    proposer = _CountingProposer()
    factory = _block_view_factory(build_block_view, static_renderer, n_views=2)
    segmenter = sam2_masks._Sam2MultiViewSegmenter(
        proposer, factory, needs_thickness=True, n_views=2
    )
    with segmenter, pytest.warns(UserWarning):  # K=0 → 「幾何プライアのみ」警告
        labels_cube = segmenter.segment(cube_mesh)
        labels_cylinder = segmenter.segment(capped_cylinder_mesh)
    assert validate_labels(labels_cube, len(cube_mesh.faces)) >= 1
    assert validate_labels(labels_cylinder, len(capped_cylinder_mesh.faces)) >= 1
    assert [len(t) for t in proposer.thickness_history] == [
        len(cube_mesh.faces),
        len(capped_cylinder_mesh.faces),
    ]
    for thickness01 in proposer.thickness_history:
        assert float(thickness01.min()) >= 0.0
        assert float(thickness01.max()) <= 1.0


def test_thickness_is_skipped_without_the_sdf_channel(
    cube_mesh: MeshData,
    build_block_view: Callable[..., RenderedView],
    static_renderer: Callable[..., Any],
) -> None:
    """`channels` に `"sdf"` が無い構成ではレイキャストを丸ごと払わない。"""
    proposer = _CountingProposer()
    factory = _block_view_factory(build_block_view, static_renderer, n_views=2)
    segmenter = sam2_masks._Sam2MultiViewSegmenter(
        proposer, factory, needs_thickness=False, n_views=2
    )
    with segmenter, pytest.warns(UserWarning):
        segmenter.segment(cube_mesh)
    assert proposer.thickness_history == []


def test_single_part_result_warns_that_ml_did_not_contribute(
    cube_mesh: MeshData,
    build_block_view: Callable[..., RenderedView],
    static_renderer: Callable[..., Any],
) -> None:
    """★ B-3: 結果が P==1 なら「幾何フォールバックと区別できない」と警告する。

    親 `MultiViewSegmenter` の既存ガードは `total_masks == 0` でしか発火せず、
    「マスクは出たが融合後に 1 部位へ潰れた」経路(実測の崩壊形 —
    n_views=8 / image_size=512 で accuracy 0.5000 / P=1 / 警告 0 件)を捉えない。
    **本体は凍結**なのでサブクラス側の告知を固定する。

    `angle_deg=180` は「幾何プライアが全面を 1 部位へ結合する」状況を立方体で
    安価に作るための設定(P==1 の再現手段であって既定の推奨ではない)。
    """
    proposer = _CountingProposer()
    factory = _block_view_factory(build_block_view, static_renderer, n_views=2)
    segmenter = sam2_masks._Sam2MultiViewSegmenter(
        proposer, factory, needs_thickness=False, n_views=2, angle_deg=180.0
    )
    with segmenter, pytest.warns(UserWarning, match="single part") as captured:
        labels = segmenter.segment(cube_mesh)
    assert validate_labels(labels, len(cube_mesh.faces)) == 1
    single = [
        str(record.message)
        for record in captured
        if "single part" in str(record.message)
    ]
    assert len(single) == 1, single
    # 検証済み既定の実測値(何が期待値か)と、**この実行の現在値**(何で走ったか)を
    # 両方要求する。後者が無いと利用者は原因追跡に 1 往復増える(B-3)。
    assert "n_views=24" in single[0] and "image_size=1024" in single[0]
    assert "this run: n_views=2;" in single[0]
    assert "--segmenter geometric" in single[0]


def test_multi_part_result_does_not_warn_about_collapse(
    cube_mesh: MeshData,
    build_block_view: Callable[..., RenderedView],
    static_renderer: Callable[..., Any],
) -> None:
    """負の対照: P >= 2 では崩壊警告を出さない(常時警告する実装なら落ちる)。

    既定 `angle_deg=60` の立方体は facet 境界(90 度)で割れるので P > 1。
    """
    proposer = _CountingProposer()
    factory = _block_view_factory(build_block_view, static_renderer, n_views=2)
    segmenter = sam2_masks._Sam2MultiViewSegmenter(
        proposer, factory, needs_thickness=False, n_views=2
    )
    with segmenter, warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        labels = segmenter.segment(cube_mesh)
    assert validate_labels(labels, len(cube_mesh.faces)) > 1
    assert [str(r.message) for r in records if "single part" in str(r.message)] == []


# ---------------------------------------------------------------------------
# 非 ML 6: Sam2MaskProposer の単体規約(裁定B — 面積帯・チャンネル・シード検査)
# ---------------------------------------------------------------------------


def _foreground_view(size: int = 16) -> tuple[RenderedView, int]:
    """中央 8x8 が面 0 の合成ビュー(シルエット 64 画素)を組む。"""
    face_id = np.full((size, size), -1, dtype=np.int32)
    face_id[4:12, 4:12] = 0
    return _make_view(face_id), 64


def _record(mask: np.ndarray, point: tuple[float, float]) -> dict[str, Any]:
    """AMG の返却レコード(検査で使うキーだけ)を組む。"""
    return {"segmentation": mask, "point_coords": [list(point)]}


def test_proposer_rejects_a_generator_without_point_grids() -> None:
    with pytest.raises(ValueError, match="point_grids"):
        sam2_masks.Sam2MaskProposer(object())


def test_proposer_propose_after_close_raises() -> None:
    proposer = sam2_masks.Sam2MaskProposer(_FakeGenerator())
    proposer.close()
    proposer.close()  # 冪等
    view, _ = _foreground_view()
    with pytest.raises(RuntimeError, match="close"):
        proposer.propose(view)


def test_proposer_returns_no_masks_for_an_empty_silhouette() -> None:
    """空シルエット = その視点でメッシュが見えないだけ → K=0(AMG は呼ばない)。"""
    generator = _FakeGenerator()
    proposer = sam2_masks.Sam2MaskProposer(generator, channels=("shading",))
    view = _make_view(np.full((8, 8), -1, dtype=np.int32))
    masks = proposer.propose(view)
    assert masks.shape == (0, 8, 8)
    assert masks.dtype == np.bool_
    assert generator.generate_images == []


def test_proposer_sdf_channel_requires_thickness() -> None:
    proposer = sam2_masks.Sam2MaskProposer(_FakeGenerator(), channels=("sdf",))
    view, _ = _foreground_view()
    with pytest.raises(RuntimeError, match="set_face_thickness"):
        proposer.propose(view)


def test_proposer_applies_the_area_band() -> None:
    """裁定B: 候補は面積帯 `[lo, hi]`(シルエット比)で選別される。

    シルエット 64 画素に対し 2 画素(3%)と 60 画素(94%)は帯 (0.05, 0.85) の
    外、32 画素(50%)だけが通る。
    """
    view, silhouette_area = _foreground_view()
    tiny = np.zeros((16, 16), dtype=bool)
    tiny[4, 4:6] = True  # 2 px / 64 = 0.031
    mid = np.zeros((16, 16), dtype=bool)
    mid[4:8, 4:12] = True  # 32 px / 64 = 0.5
    huge = np.zeros((16, 16), dtype=bool)
    huge[4:12, 4:12] = True
    huge[4, 4:8] = False  # 60 px / 64 = 0.94
    generator = _FakeGenerator()
    generator.records = [
        _record(tiny, (8.0, 4.0)),
        _record(mid, (6.0, 6.0)),
        _record(huge, (8.0, 8.0)),
    ]
    proposer = sam2_masks.Sam2MaskProposer(generator, channels=("shading",))
    masks = proposer.propose(view)
    assert masks.shape == (1, 16, 16)
    assert np.array_equal(masks[0], mid)


def test_proposer_runs_amg_once_per_channel_with_the_right_images() -> None:
    """裁定B: 既定 `("sdf", "shading")` は AMG を 2 回回す — 1 回目は厚み合成の
    グレースケール、2 回目は `view.color`。"""
    face_id = np.full((16, 16), -1, dtype=np.int32)
    face_id[4:12, 4:12] = 0
    color = np.zeros((16, 16, 3), dtype=np.uint8)
    color[4:12, 4:12] = (10, 200, 30)
    view = _make_view(face_id, color)
    thickness01 = np.array([0.25])
    generator = _FakeGenerator()
    proposer = sam2_masks.Sam2MaskProposer(generator)  # 既定 channels
    proposer.set_face_thickness(thickness01)
    masks = proposer.propose(view)
    assert masks.shape == (0, 16, 16)  # 偽 AMG は候補ゼロ
    assert len(generator.generate_images) == 2
    assert np.array_equal(
        generator.generate_images[0], thickness_to_image(face_id, thickness01)
    )
    assert np.array_equal(generator.generate_images[1], color)


def test_proposer_swaps_the_point_grid_to_the_silhouette() -> None:
    """propose のたびに `point_grids` が視点のシルエット内グリッドへ差し替わる。"""
    view, _ = _foreground_view()
    generator = _FakeGenerator()
    proposer = sam2_masks.Sam2MaskProposer(
        generator, channels=("shading",), grid_side=4
    )
    proposer.propose(view)
    grid = np.asarray(generator.point_grids[0])
    assert grid.shape == (16, 2)
    # 正規化座標を画素へ戻すと全点が前景に載る(シルエット内グリッドの定義)。
    xs = np.clip(np.round(grid[:, 0] * 16).astype(int), 0, 15)
    ys = np.clip(np.round(grid[:, 1] * 16).astype(int), 0, 15)
    assert view.coverage[ys, xs].all()


def test_proposer_detects_an_ineffective_grid_swap() -> None:
    """返却シードがシルエット外に散っていたら黙って続けず `RuntimeError`。

    (sam2 の将来版が generate 時に `point_grids` 属性を読まなくなる、という
    実在する壊れ方の検出器 — probe 実証済みの検査。)
    """
    view, _ = _foreground_view()
    outside = np.zeros((16, 16), dtype=bool)
    outside[4:8, 4:12] = True
    generator = _FakeGenerator()
    generator.records = [_record(outside, (0.0, 0.0))]  # シード = 背景の隅
    proposer = sam2_masks.Sam2MaskProposer(generator, channels=("shading",))
    with pytest.raises(RuntimeError, match="point_grids"):
        proposer.propose(view)


def test_set_face_thickness_validates_its_input() -> None:
    proposer = sam2_masks.Sam2MaskProposer(_FakeGenerator())
    with pytest.raises(ValueError, match=r"\(M,\)"):
        proposer.set_face_thickness(np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        proposer.set_face_thickness(np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        proposer.set_face_thickness(np.array([np.nan]))


def test_silhouette_point_grid_contract() -> None:
    coverage = np.zeros((8, 8), dtype=bool)
    coverage[2:6, 3:7] = True
    grid = sam2_masks.silhouette_point_grid(coverage, 3)
    assert grid.shape == (9, 2)
    assert grid.dtype == np.float64
    xs = np.clip(np.round(grid[:, 0] * 8).astype(int), 0, 7)
    ys = np.clip(np.round(grid[:, 1] * 8).astype(int), 0, 7)
    assert coverage[ys, xs].all()
    with pytest.raises(ValueError, match="foreground"):
        sam2_masks.silhouette_point_grid(np.zeros((8, 8), dtype=bool), 3)
    with pytest.raises(ValueError, match="bool"):
        sam2_masks.silhouette_point_grid(np.zeros((8, 8), dtype=np.int32), 3)


# ---------------------------------------------------------------------------
# ML 層のヘルパ
# ---------------------------------------------------------------------------


def _one_to_one_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """重み降順・貪欲な一対一マッチング精度(計画v4 Step 2-5 ゲート7 の定義)。

    混同行列 `C (P, T)` を作り、重み降順(tie は小さい pred → 小さい true)で
    pred–true を一対一に対応付け、`Σ matched / M` を返す。未マッチの予測部位の
    面はすべて誤りに算入される。scipy(Hungarian)は使わない(BL-5(a))。
    """
    pred_array = np.asarray(pred, dtype=np.int64)
    true_array = np.asarray(true, dtype=np.int64)
    assert pred_array.shape == true_array.shape and pred_array.ndim == 1
    assert pred_array.size > 0
    assert int(pred_array.min()) >= 0 and int(true_array.min()) >= 0
    n_pred = int(pred_array.max()) + 1
    n_true = int(true_array.max()) + 1
    confusion = np.bincount(
        pred_array * n_true + true_array, minlength=n_pred * n_true
    ).reshape(n_pred, n_true)
    entries = sorted(
        (
            (int(confusion[p, t]), p, t)
            for p in range(n_pred)
            for t in range(n_true)
            if confusion[p, t] > 0
        ),
        key=lambda entry: (-entry[0], entry[1], entry[2]),
    )
    used_pred: set[int] = set()
    used_true: set[int] = set()
    matched = 0
    for count, p, t in entries:
        if p in used_pred or t in used_true:
            continue
        used_pred.add(p)
        used_true.add(t)
        matched += count
    return matched / pred_array.size


class _NullProposer:
    """常に K=0(幾何ベースライン用 — probe と同じ対照)。"""

    def propose(self, view: RenderedView) -> np.ndarray:
        height, width = view.face_id.shape
        return np.zeros((0, height, width), dtype=bool)

    def close(self) -> None:
        pass


class _RecordingProposer:
    """実 proposer の出力を視点順に記録する(ゲート11 のキャッシュ)。"""

    def __init__(self, inner: sam2_masks.Sam2MaskProposer) -> None:
        self._inner = inner
        self.masks: list[np.ndarray] = []

    def set_face_thickness(self, thickness01: np.ndarray) -> None:
        self._inner.set_face_thickness(thickness01)

    def propose(self, view: RenderedView) -> np.ndarray:
        masks = self._inner.propose(view)
        self.masks.append(masks)
        return masks

    def close(self) -> None:
        self._inner.close()


def _gl_renderer_factory(
    image_size: int, shading: str
) -> Callable[[MeshData], MeshRenderer]:
    def factory(mesh: MeshData) -> MeshRenderer:
        from atlasmith.segmentation.multiview.render import _ModernglRenderer

        return _ModernglRenderer(mesh, image_size=image_size, shading=shading)

    return factory


def _build_real_proposer(**proposer_kwargs: Any) -> sam2_masks.Sam2MaskProposer:
    """実重みの AMG を production と同じ既定(モデル/閾値/グリッド)で構築する。"""
    generator_cls = sam2_masks._import_sam2()
    generator = generator_cls.from_pretrained(
        sam2_masks.DEFAULT_MODEL_ID,
        points_per_side=None,
        point_grids=[sam2_masks._uniform_point_grid(sam2_masks.DEFAULT_GRID_SIDE)],
        pred_iou_thresh=sam2_masks.DEFAULT_PRED_IOU_THRESH,
        stability_score_thresh=sam2_masks.DEFAULT_STABILITY_SCORE_THRESH,
        crop_n_layers=sam2_masks.DEFAULT_CROP_N_LAYERS,
        device="cuda",
    )
    return sam2_masks.Sam2MaskProposer(generator, **proposer_kwargs)


@pytest.fixture(scope="module")
def sam2_peanut_runs(peanut_mesh_module: MeshData) -> dict[str, Any]:
    """既定パラメータの SAM2 segment を **2 回だけ** 実行して共有する(裁定F)。

    幾何ベースライン(K=0 proposer・同一パラメータ)も 1 回実行する。
    ゲート 7〜10 はこの結果を読むだけで SAM2 を再実行しない(1 回数分かかる)。
    `make_sam2_segmenter()` は**無引数** = 全て既定(ゲート10 の要件)。
    """
    timings: dict[str, float] = {}
    started = time.perf_counter()
    segmenter = make_sam2_segmenter()
    timings["make_sam2_segmenter"] = time.perf_counter() - started
    with segmenter:
        started = time.perf_counter()
        labels_first = segmenter.segment(peanut_mesh_module)
        timings["segment_first"] = time.perf_counter() - started
        started = time.perf_counter()
        labels_second = segmenter.segment(peanut_mesh_module)
        timings["segment_second"] = time.perf_counter() - started
    started = time.perf_counter()
    with warnings.catch_warnings():
        # K=0 は意図した対照実験なので「no masks」系の警告は想定内。
        warnings.simplefilter("ignore")
        with MultiViewSegmenter(
            _NullProposer(),
            _gl_renderer_factory(DEFAULT_IMAGE_SIZE, DEFAULT_SHADING),
        ) as geometric:
            labels_geometric = geometric.segment(peanut_mesh_module)
    timings["segment_geometric"] = time.perf_counter() - started
    print(
        "\n[step2-5 ml fixture] timings_sec="
        + str({key: round(value, 1) for key, value in timings.items()})
    )
    return {
        "labels_first": labels_first,
        "labels_second": labels_second,
        "labels_geometric": labels_geometric,
        "timings": timings,
        "make_kwargs": {},
    }


# ---------------------------------------------------------------------------
# ML 5/6: 疎通ゲート(品質主張はしない)
# ---------------------------------------------------------------------------


@pytest.mark.gl
@pytest.mark.ml
def test_sam2_smoke_three_rects() -> None:
    """SAM2 疎通ゲート — **疎通確認であり品質ゲートではない**(ゲート5)。

    単色背景に 3 矩形の合成画像は色境界だけで割れるため、通っても部位分割の
    品質は何も主張しない。落ちたら「重み・推論・マスク返却の形」のどれかが
    壊れている、という切り分けにだけ使う。
    """
    size = 512
    rects = ((50, 50, 150, 150), (80, 300, 160, 420), (300, 150, 440, 240))
    face_id = np.full((size, size), -1, dtype=np.int32)
    color = np.zeros((size, size, 3), dtype=np.uint8)
    palette = ((220, 60, 60), (60, 220, 60), (60, 60, 220))
    for index, (r0, c0, r1, c1) in enumerate(rects):
        face_id[r0:r1, c0:c1] = index
        color[r0:r1, c0:c1] = palette[index]
    view = _make_view(face_id, color)

    proposer = _build_real_proposer(channels=("shading",))
    try:
        masks = proposer.propose(view)
    finally:
        proposer.close()

    assert masks.shape[0] >= 3
    for r0, c0, r1, c1 in rects:
        rect = np.zeros((size, size), dtype=bool)
        rect[r0:r1, c0:c1] = True
        best_iou = max(
            float(np.count_nonzero(mask & rect)) / float(np.count_nonzero(mask | rect))
            for mask in masks
        )
        assert best_iou >= 0.90


@pytest.mark.gl
@pytest.mark.ml
def test_multiview_smoke_two_color(
    build_peanut_geometry: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """多視点パイプライン疎通ゲート — **疎通確認であり品質ゲートではない**
    (ゲート6)。2 色に塗り分けた小さめのダンベルで一気通貫が例外なく完走し、
    ラベル契約(連番・全面付与)を満たすことだけを見る。

    **この構成(n_views=6 / image_size=512)は ML の寄与が崩壊する領域である**
    (2026-08-03 反証レビュー B-3 の実測: n_views=8 単独・image_size=512 単独の
    どちらでも accuracy 0.9574 → 0.5000 / P=1 = 幾何プライアと同値)。ここで
    見ているのは**一気通貫が落ちないこと**だけで、品質の主張は一切しない
    (主品質ゲートは 7、負の対照は 8 — どちらも既定パラメータ)。実行が数分で
    済むよう小さい構成を選んでいる、という位置づけ自体は計画どおり。

    したがって `build_sam2_segmenter` の「既定未満」警告が**必ず**出る。
    黙って握り潰さず `pytest.warns` で受ける(出なくなったら告知が壊れた
    ということなので、この test が落ちて知らせる)。P==1 の崩壊警告も同時に
    出うるが、そちらは実行ごとに変わるので存在を要求しない。
    """
    vertices, faces, uv = build_peanut_geometry(24, 32)
    texture = np.zeros((64, 64, 3), dtype=np.float32)
    texture[:32] = (0.9, 0.15, 0.15)  # V < 0.5 = 北ローブ
    texture[32:] = (0.15, 0.15, 0.9)  # V >= 0.5 = 南ローブ
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps={"basecolor": texture},
        source_vertex=np.arange(len(vertices), dtype=np.int64),
    )
    with pytest.warns(UserWarning, match="below its validated defaults"):
        with make_sam2_segmenter(n_views=6, image_size=512) as segmenter:
            labels = segmenter.segment(mesh)
    n_parts = validate_labels(labels, len(faces))
    assert n_parts >= 1


# ---------------------------------------------------------------------------
# ML 7〜10: 主品質・負の対照・安定性・既定パラメータ(裁定E/F)
# ---------------------------------------------------------------------------


@pytest.mark.gl
@pytest.mark.ml
@pytest.mark.slow
def test_sam2_splits_the_monochrome_peanut(
    sam2_peanut_runs: dict[str, Any],
    peanut_mesh_module: MeshData,
    peanut_truth_labels: Callable[[MeshData], np.ndarray],
) -> None:
    """★ 主 ML 品質ゲート(ゲート7・裁定E)。閾値は下げない(停止条件)。"""
    truth = peanut_truth_labels(peanut_mesh_module)
    labels_geometric = sam2_peanut_runs["labels_geometric"]
    # 前提条件: 幾何プライア単独では割れない(崩れたら本ゲートは空虚)。
    assert int(labels_geometric.max()) + 1 == 1
    labels = sam2_peanut_runs["labels_first"]
    n_parts = int(labels.max()) + 1
    n_true = int(truth.max()) + 1
    accuracy = _one_to_one_accuracy(labels, truth)
    print(f"\n[step2-5 gate7] accuracy={accuracy:.4f} P={n_parts} T={n_true}")
    assert n_parts <= min(n_true * 3, 8)  # 裁定E: 過分割で精度を稼がせない
    assert accuracy >= 0.85  # 裁定E の床(spike 実測 0.9574)


@pytest.mark.gl
@pytest.mark.ml
@pytest.mark.slow
def test_sam2_output_differs_from_and_beats_the_geometric_prior(
    sam2_peanut_runs: dict[str, Any],
    peanut_mesh_module: MeshData,
    peanut_truth_labels: Callable[[MeshData], np.ndarray],
) -> None:
    """負の対照(ゲート8): 出力が幾何プライアと**別物**であることを固定する。

    **N-5(2026-08-03 反証レビュー)— 表現の訂正**: 本ゲートを「ML が寄与した
    ことの唯一の証拠」と呼ぶのは過大だった。peanut の真値は 3008:3008 の完全
    対称なので、P=1 に潰れた幾何プライアの一対一精度は**必ず** 0.5000 になる。
    よって `delta >= 0.30` は `accuracy >= 0.80` と代数的に同値で、ゲート7 の床
    (0.85)より**弱い** — 精度の主張はゲート7 が担う。ゲート7 に含まれない
    独立な情報は `not np.array_equal`(= 同じラベル配列を返していない)だけ
    であり、それがこのゲートの存在理由である。
    """
    truth = peanut_truth_labels(peanut_mesh_module)
    labels_ml = sam2_peanut_runs["labels_first"]
    labels_geometric = sam2_peanut_runs["labels_geometric"]
    assert not np.array_equal(labels_ml, labels_geometric)
    delta = _one_to_one_accuracy(labels_ml, truth) - _one_to_one_accuracy(
        labels_geometric, truth
    )
    print(f"\n[step2-5 gate8] delta={delta:+.4f}")
    assert delta >= 0.30  # 裁定E の床(spike 実測 +0.4574)


@pytest.mark.gl
@pytest.mark.ml
@pytest.mark.slow
def test_sam2_is_stable_across_two_runs(sam2_peanut_runs: dict[str, Any]) -> None:
    """安定性(ゲート9): 同一メッシュ 2 回の一対一マッチング一致率 >= 0.98。

    SAM2 は「決定的」ではなく「安定」(計画v4 §2.2)なので、ビット一致ではなく
    一致率で測る。
    """
    agreement = _one_to_one_accuracy(
        sam2_peanut_runs["labels_first"], sam2_peanut_runs["labels_second"]
    )
    print(f"\n[step2-5 gate9] agreement={agreement:.4f}")
    assert agreement >= 0.98  # 裁定E


@pytest.mark.gl
@pytest.mark.ml
@pytest.mark.slow
def test_default_parameters_ran_and_times_were_recorded(
    sam2_peanut_runs: dict[str, Any],
) -> None:
    """既定パラメータ経路(ゲート10): 7〜9 の共有 fixture が
    `make_sam2_segmenter()` を**無引数**(= n_views=24 / image_size=1024 /
    sam2.1-hiera-large / channels=("sdf","shading"))で実行済みであることと、
    所要時間が記録されたことを固定する。実測値は fixture が print する。"""
    assert sam2_peanut_runs["make_kwargs"] == {}
    timings = sam2_peanut_runs["timings"]
    assert {"make_sam2_segmenter", "segment_first", "segment_second"} <= set(timings)
    assert all(value > 0.0 for value in timings.values())


# ---------------------------------------------------------------------------
# ML 11: 決定的層の非退行(キャッシュ再生)
# ---------------------------------------------------------------------------


@pytest.mark.gl
@pytest.mark.ml
def test_cached_sam2_masks_replay_identically_through_the_deterministic_layer(
    peanut_mesh_module: MeshData,
    static_mask_proposer: Callable[..., Any],
) -> None:
    """ゲート11: SAM2 出力をキャッシュして `_StaticMaskProposer` に流すと、
    決定的層(レンダ・面ID・融合)が**同一ラベル**を再生する。

    Step 2-4 の「マスク提案が同一なら決定的」というゲート群と矛盾しないことの
    実証 — もし再生が食い違ったら、非決定性が proposer の外(決定的層)へ
    漏れている。

    **この構成(n_views=6 / image_size=512)も ML の寄与が崩壊する領域**
    (2026-08-03 反証レビュー B-3)。ここで見ているのは「同じマスクなら同じ
    ラベルが出るか」= 決定的層の非退行だけで、**品質の主張はしない**(その
    比較対象は live 自身なので、崩壊していても意味は変わらない)。よって
    崩壊警告(P==1)は出うるものとして明示的に無視する — 出る/出ないが
    実行ごとに変わる量なので、存在を要求も禁止もしない。
    """
    n_views = 6
    image_size = 512
    factory = _gl_renderer_factory(image_size, DEFAULT_SHADING)
    recorder = _RecordingProposer(_build_real_proposer())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 崩壊警告(P==1)は本ゲートの対象外
        with sam2_masks._Sam2MultiViewSegmenter(
            recorder, factory, needs_thickness=True, n_views=n_views
        ) as live:
            labels_live = live.segment(peanut_mesh_module)
    assert len(recorder.masks) == n_views

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 被覆・割当の警告は本ゲートの対象外
        with MultiViewSegmenter(
            static_mask_proposer(recorder.masks), factory, n_views=n_views
        ) as replay:
            labels_replay = replay.segment(peanut_mesh_module)
    assert np.array_equal(labels_live, labels_replay)
