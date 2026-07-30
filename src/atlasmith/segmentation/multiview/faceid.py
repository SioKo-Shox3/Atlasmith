"""面IDバッファの 24bit エンコード / デコード規約(計画v4 §2.4.2)。

レンダラ(`render.py`)が RGBA8 のカラーアタッチメントへ面IDを書き、CPU 側が
それを厳密に読み戻すための**数値契約**をここに一本化する。production と
テストが同じ関数を使うのは意図的(契約が 2 箇所に書かれると必ずずれる)。

規約:

  - 送出値 `e = face_code + 1`。**`e == 0` は背景専用**なので、面IDそのものを
    書くのではなく 1 を足す。
  - `R = e & 0xFF` / `G = (e >> 8) & 0xFF` / `B = (e >> 16) & 0xFF` /
    **`A = 255`(前景マーカー)**。背景はクリア値 `(0, 0, 0, 0)`。
  - 被覆判定は **alpha** で行う(色値から独立 — 色が真っ黒な前景画素と背景を
    取り違えない)。
  - 上限は `M <= 16_777_215`(= 2^24 - 1)。超過は `ValueError` で、**黙って
    上位ビットを捨てない**。

依存方向(計画v4 §2.1): numpy のみ。GL も torch も知らない。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BACKGROUND_FACE_ID",
    "FOREGROUND_ALPHA",
    "MAX_FACE_CODE",
    "MAX_FACE_COUNT",
    "decode_face_id",
    "encode_face_codes",
    "encode_face_codes_rgba",
    "validate_coverage_consistency",
    "validate_face_count",
]

# 面数の上限(= 2^24 - 1)。面コードは 0..M-1 なので、送出値 `e = code + 1` の
# 最大は 0xFFFFFF に収まる。
MAX_FACE_COUNT = 16_777_215
# 面コードの上限(= MAX_FACE_COUNT - 1)。`e = MAX_FACE_CODE + 1 = 0xFFFFFF`。
MAX_FACE_CODE = MAX_FACE_COUNT - 1
# 背景画素のデコード結果(`e == 0` → `face_id == -1`)。
BACKGROUND_FACE_ID = -1
# 前景マーカーの alpha。シェーダは常に 1.0 を書き、クリア値は 0。
FOREGROUND_ALPHA = 255

# 8bit 固定小数点の分母。`k / 255.0` は float32 でも往復が厳密
# (相対誤差 < 2^-24 なので `round(255 * float32(k/255)) == k`)。
_UNORM8_MAX = 255.0


def validate_face_count(n_faces: int) -> None:
    """24bit 面IDに収まる面数かを検査する。

    Args:
        n_faces: メッシュの面数。

    Raises:
        ValueError: `n_faces` が `MAX_FACE_COUNT` を超えるとき。メッセージには
            実 M・上限・回避策(幾何バックエンド / 将来の R32UI 経路)を含める。
    """
    if int(n_faces) > MAX_FACE_COUNT:
        raise ValueError(
            f"mesh has {int(n_faces)} faces, which exceeds the 24-bit face-id "
            f"limit of {MAX_FACE_COUNT}. Use `--segmenter geometric`, or wait for "
            "the R32UI face-id path (deferred to Phase 3). Face ids are never "
            "silently truncated."
        )


def encode_face_codes_rgba(face_codes: np.ndarray) -> np.ndarray:
    """面コード `(M,)` を、GPU が書き込むはずの RGBA バイト `(M, 4) uint8` へ。

    **この関数が 24bit 規約の一次定義**であり、`encode_face_codes`(頂点属性用の
    float 色)も `decode_face_id` の期待値もここから導かれる。テストの往復ゲートが
    見るのもこのバイト列。

    Args:
        face_codes: `(M,)` の整数配列。値域は `[0, MAX_FACE_CODE]`。

    Returns:
        `(M, 4) uint8`。alpha は全行 `FOREGROUND_ALPHA`。

    Raises:
        ValueError: shape が `(M,)` でない、整数でない、値域外のとき。
    """
    codes = _as_face_codes(face_codes)
    e = codes + 1
    rgb = np.stack([e & 0xFF, (e >> 8) & 0xFF, (e >> 16) & 0xFF], axis=1)
    alpha = np.full((codes.shape[0], 1), FOREGROUND_ALPHA, dtype=np.int64)
    return np.concatenate([rgb, alpha], axis=1).astype(np.uint8)


def encode_face_codes(face_codes: np.ndarray) -> np.ndarray:
    """面コード `(M,)` を頂点属性用の色 `(M, 3) float32 [0, 1]` へ。

    **WHY float 属性か**(計画v4 §5 Step 2-3 probe (c)): 整数頂点属性
    (`glVertexAttribIPointer`)の可搬性に依存せず、0..1 正規化色を CPU 側で
    作って渡す。GL は書き込み時に `round(clamp(v, 0, 1) * 255)` で 8bit へ戻すので、
    `encode_face_codes_rgba` のバイト列がそのまま復元される。

    Args:
        face_codes: `(M,)` の整数配列。値域は `[0, MAX_FACE_CODE]`。

    Returns:
        `(M, 3) float32`。alpha はシェーダが定数 1.0 を書くので含めない。

    Raises:
        ValueError: `encode_face_codes_rgba` と同じ条件。
    """
    rgba = encode_face_codes_rgba(face_codes)
    return rgba[:, :3].astype(np.float32) / np.float32(_UNORM8_MAX)


def decode_face_id(buf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """面IDバッファ `(H, W, 4) uint8` を `(face_id int32, coverage bool)` へ。

    **int64 への明示昇格は数値契約の一部**(計画v4 §2.4.2): uint8 のまま
    `* 256` すると NEP 50(numpy 2.x)で
    `OverflowError: Python integer 256 out of bounds for uint8` になる。
    昇格を落とすと最大値付近の面IDが黙って壊れるのではなく即座に落ちる —
    そのこと自体をテストで固定してある。

    Args:
        buf: `(H, W, 4) uint8` の読み戻しバッファ(行方向は呼び出し側で
            画面上端が row 0 になるよう整えてから渡す)。

    Returns:
        `(face_id (H, W) int32, coverage (H, W) bool)`。背景の `face_id` は
        `BACKGROUND_FACE_ID`(= -1)。`coverage` は **alpha だけ**から決まる。

    Raises:
        ValueError: shape/dtype が契約と違うとき。
    """
    array = np.asarray(buf)
    if array.ndim != 3 or array.shape[2] != 4 or array.dtype != np.uint8:
        raise ValueError(
            f"buf must be (H, W, 4) uint8, got shape {array.shape} dtype {array.dtype}"
        )
    rgb = array[..., :3].astype(np.int64)
    e = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    face_id = (e - 1).astype(np.int32)
    coverage = array[..., 3] == FOREGROUND_ALPHA
    return face_id, coverage


def validate_coverage_consistency(
    face_id: np.ndarray, coverage: np.ndarray, *, context: str
) -> None:
    """`coverage <=> (face_id >= 0)` が全画素で成立することを確認する。

    計画v4 §2.4.2 の production 不変条件。alpha 由来の被覆と RGB 由来の面IDが
    食い違うのは、ディザ / ブレンド / sRGB 変換のいずれかがカラーバッファに
    介入した証拠なので、**壊れた面IDを下流(視点間融合)へ流さずに止める**。

    Args:
        face_id: `(H, W)` の面ID(背景は負)。
        coverage: `(H, W)` の被覆 bool。
        context: エラーメッセージへ入れる文脈(例: `"view 3"`)。

    Raises:
        ValueError: 1 画素でも食い違うとき(件数と最初の座標を報告する)。
    """
    mismatch = coverage != (face_id >= 0)
    n_bad = int(mismatch.sum())
    if n_bad:
        rows, cols = np.nonzero(mismatch)
        row, col = int(rows[0]), int(cols[0])
        raise ValueError(
            f"{context}: coverage <=> (face_id >= 0) is violated at {n_bad} "
            f"pixel(s); first at (row={row}, col={col}) with "
            f"coverage={bool(coverage[row, col])} face_id={int(face_id[row, col])}. "
            "Dithering, blending or sRGB conversion is corrupting the face-id "
            "attachment."
        )


def _as_face_codes(face_codes: np.ndarray) -> np.ndarray:
    """`(M,) int64` として検証した面コード配列を返す。"""
    codes = np.asarray(face_codes)
    if codes.ndim != 1:
        raise ValueError(f"face_codes must have shape (M,), got {codes.shape}")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(
            f"face_codes must have an integer dtype, got {codes.dtype} "
            "(float codes cannot round-trip through the 24-bit encoding)"
        )
    codes = codes.astype(np.int64)
    if codes.size:
        lo, hi = int(codes.min()), int(codes.max())
        if lo < 0 or hi > MAX_FACE_CODE:
            raise ValueError(
                f"face_codes must be within [0, {MAX_FACE_CODE}], got [{lo}, {hi}]"
            )
    return codes
