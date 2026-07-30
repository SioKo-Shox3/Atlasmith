"""面IDエンコード規約のゲート(計画v4 §5 Step 2-3 ゲート1・ゲート10 の一部)。

GPU を一切使わない numpy だけの往復試験。GPU 上での厳密性(固定小数点書き込みが
ディザ/ブレンドで壊れないこと)は `tests/test_multiview_render.py` の sentinel
ゲート(`@pytest.mark.gl`)が別途実証する — 本ファイルは**数値契約そのもの**の
検査で、両方が揃って初めて「面IDが厳密」と言える。
"""

from __future__ import annotations

import numpy as np
import pytest

from atlasmith.segmentation.multiview.faceid import (
    BACKGROUND_FACE_ID,
    FOREGROUND_ALPHA,
    MAX_FACE_CODE,
    MAX_FACE_COUNT,
    decode_face_id,
    encode_face_codes,
    encode_face_codes_rgba,
    validate_coverage_consistency,
    validate_face_count,
)

# 計画v4 §5 Step 2-3 ゲート1 が指定する代表値。バイト境界(R→G は 254/255、
# G→B は 65534/65535 — `e = code + 1` なので境界は本文の 255/65535 ではなく
# こちら側にずれる。§0-A 条件6 の訂正)と 24bit 上限付近を含む。
_ROUND_TRIP_CODES = np.array(
    [0, 1, 2, 254, 255, 256, 65534, 65535, 65536, 65537, 16777213, 16777214],
    dtype=np.int64,
)


def _as_image(rgba: np.ndarray) -> np.ndarray:
    """`(M, 4) uint8` を、デコーダが受け取る `(1, M, 4)` 画像へ整形する。"""
    return np.ascontiguousarray(rgba[np.newaxis, :, :])


def test_encode_decode_round_trip_is_exact() -> None:
    """代表 12 値が `decode(encode(id)) == id` で全件厳密一致する(ゲート1)。

    どう壊れたら落ちるか: シフト量・バイト順・`+1` のオフセットのいずれかが
    ずれた瞬間に、境界値(254/255・65534/65535)から落ちる。
    """
    rgba = encode_face_codes_rgba(_ROUND_TRIP_CODES)
    face_id, coverage = decode_face_id(_as_image(rgba))

    assert face_id.dtype == np.int32
    assert np.array_equal(face_id[0].astype(np.int64), _ROUND_TRIP_CODES)
    assert coverage.all()


def test_encoded_bytes_match_the_24bit_contract() -> None:
    """規約 `R = e & 0xFF` / `G = (e >> 8) & 0xFF` / `B = (e >> 16) & 0xFF` の実値。

    往復だけだと「エンコードとデコードが同じ向きに間違っている」場合に通って
    しまうので、バイト列そのものを独立に書き下して突き合わせる。
    """
    codes = np.array([0, 254, 255, 65534, 65535, 16777214], dtype=np.int64)
    expected = np.array(
        [
            [1, 0, 0, 255],  # e = 1
            [255, 0, 0, 255],  # e = 255      (R の上限)
            [0, 1, 0, 255],  # e = 256      (R -> G のキャリー)
            [255, 255, 0, 255],  # e = 65535    (G の上限)
            [0, 0, 1, 255],  # e = 65536    (G -> B のキャリー)
            [255, 255, 255, 255],  # e = 16777215 (24bit の上限)
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(encode_face_codes_rgba(codes), expected)


def test_float_attribute_matches_the_byte_contract() -> None:
    """頂点属性用の float32 色が、GL の unorm8 量子化で元のバイトへ戻る。

    GL は書き込み時に `round(clamp(v, 0, 1) * 255)` を行う。`k / 255` の float32
    表現でこの往復が厳密であることを CPU 側で先に確かめておく(GPU 側の実証は
    sentinel ゲート)。
    """
    colors = encode_face_codes(_ROUND_TRIP_CODES)
    assert colors.dtype == np.float32
    assert colors.shape == (_ROUND_TRIP_CODES.size, 3)
    assert colors.min() >= 0.0 and colors.max() <= 1.0

    requantized = np.round(colors.astype(np.float64) * 255.0).astype(np.uint8)
    assert np.array_equal(requantized, encode_face_codes_rgba(_ROUND_TRIP_CODES)[:, :3])


def test_background_decodes_to_minus_one() -> None:
    """クリア値 `(0,0,0,0)` は `face_id == -1` / `coverage == False`(ゲート1)。

    `e == 0` を背景専用にしてあるので、面 0 と背景が区別できる。
    """
    buf = np.array([[[0, 0, 0, 0], [1, 0, 0, FOREGROUND_ALPHA]]], dtype=np.uint8)
    face_id, coverage = decode_face_id(buf)

    assert face_id[0, 0] == BACKGROUND_FACE_ID
    assert not bool(coverage[0, 0])
    assert face_id[0, 1] == 0  # e = 1 -> face 0
    assert bool(coverage[0, 1])


def test_coverage_is_independent_of_colour() -> None:
    """被覆判定は alpha だけで決まる(色が 0 でも前景、色があっても alpha 0 は背景)。

    どう壊れたら落ちるか: 被覆を「RGB が非ゼロ」で判定する実装に戻すと、
    1 行目(alpha=255・RGB=0)が背景に化けて落ちる。
    """
    buf = np.array(
        [[[0, 0, 0, FOREGROUND_ALPHA], [7, 0, 0, 0]]],
        dtype=np.uint8,
    )
    face_id, coverage = decode_face_id(buf)

    assert bool(coverage[0, 0]) and face_id[0, 0] == BACKGROUND_FACE_ID
    assert not bool(coverage[0, 1])


def test_decode_promotes_to_int64_before_combining() -> None:
    """デコードが int64 昇格式であることを、最大値の往復で実挙動として示す。

    負の対照として「uint8 のまま 256 倍する」式が NEP 50(numpy 2.x)で
    `OverflowError` になることも示す — 昇格を落とした実装は静かに間違うのでは
    なく、この例外で落ちる。
    """
    buf = np.array([[[255, 255, 255, FOREGROUND_ALPHA]]], dtype=np.uint8)
    face_id, _coverage = decode_face_id(buf)
    assert int(face_id[0, 0]) == MAX_FACE_CODE

    with pytest.raises(OverflowError):
        _ = np.uint8(255) * 256


def test_decode_rejects_malformed_buffers() -> None:
    """`(H, W, 4) uint8` 以外の入力は `ValueError`。"""
    with pytest.raises(ValueError, match=r"\(H, W, 4\) uint8"):
        decode_face_id(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(H, W, 4\) uint8"):
        decode_face_id(np.zeros((4, 4, 4), dtype=np.float32))


def test_validate_face_count_rejects_over_the_24bit_limit() -> None:
    """`M > 16_777_215` は `ValueError`(ゲート1 / ゲート10)。

    メッセージには実 M・上限・代替経路が入る(黙って上位ビットを捨てない)。
    """
    validate_face_count(MAX_FACE_COUNT)  # 上限ちょうどは通る。
    with pytest.raises(ValueError) as excinfo:
        validate_face_count(MAX_FACE_COUNT + 1)
    message = str(excinfo.value)
    assert str(MAX_FACE_COUNT + 1) in message
    assert str(MAX_FACE_COUNT) in message
    assert "--segmenter geometric" in message
    assert "R32UI" in message


def test_encode_rejects_out_of_contract_codes() -> None:
    """面コードの shape / dtype / 値域の違反は `ValueError`。"""
    with pytest.raises(ValueError, match=r"shape \(M,\)"):
        encode_face_codes_rgba(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="integer dtype"):
        encode_face_codes_rgba(np.array([0.0, 1.0], dtype=np.float64))
    with pytest.raises(ValueError, match=r"within \[0, 16777214\]"):
        encode_face_codes_rgba(np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError, match=r"within \[0, 16777214\]"):
        encode_face_codes_rgba(np.array([MAX_FACE_CODE + 1], dtype=np.int64))


def test_validate_coverage_consistency_detects_violations() -> None:
    """`coverage <=> (face_id >= 0)` の破れを件数と座標付きで報告する。

    これは production の fail-loud 不変条件(ディザ/ブレンド事故の検出網)。
    GL 経路のゲートは `tests/test_multiview_render.py` 側にあり、ここでは
    検出器そのものが空虚でないことを見る。
    """
    face_id = np.array([[0, -1], [-1, 3]], dtype=np.int32)
    coverage = np.array([[True, False], [False, True]], dtype=bool)
    validate_coverage_consistency(face_id, coverage, context="view 0")

    broken = coverage.copy()
    broken[1, 1] = False
    with pytest.raises(ValueError) as excinfo:
        validate_coverage_consistency(face_id, broken, context="view 7")
    message = str(excinfo.value)
    assert "view 7" in message
    assert "row=1, col=1" in message
