"""マスク付き PSNR。評価マスク内テクセルの全チャンネル一括 MSE から PSNR を返す。

品質契約(横断規約): 画素は float32 [0, 1]・channels last・data_range = 1.0。
MSE は全チャンネル・全評価点を一括平均する(チャンネル別に分けない)。sRGB 変換は
行わない(線形配列として扱う)。空マスクは ValueError にする — 評価対象ゼロを黙って
成功(inf)扱いにしないため(B3)。
"""

from __future__ import annotations

import numpy as np

# 画素値域 [0, 1] の最大振幅。PSNR = 10 * log10(range^2 / MSE)。
_DATA_RANGE = 1.0


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """`mask` が True のテクセルだけで `a` と `b` の PSNR(dB)を返す。

    引数:
        a, b: 比較する画像 `(H, W, C)` または `(H, W)`。dtype は任意(内部で
            float64 に昇格して計算する)。同 shape であること。
        mask: 評価テクセル `(H, W) bool`。空間位置を選び、選ばれたテクセルの
            全チャンネルが MSE に寄与する(チャンネル一括平均 — 品質契約)。

    戻り値:
        PSNR(dB)。選択領域で `a == b`(MSE = 0)のとき `inf`。

    例外:
        ValueError: `mask` が1つも True を含まないとき(評価対象が空)。
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    # 空マスクは 0 サンプルの平均(= nan / 発散)を招く。黙って inf を返さず、
    # 呼び出し側の設計ミス(評価領域が空)として顕在化させる(B3)。
    if not mask_bool.any():
        raise ValueError("masked_psnr: mask selects no texels (empty evaluation set).")
    diff = a64[mask_bool] - b64[mask_bool]
    mse = float(np.mean(diff * diff))
    # MSE = 0(完全一致)は log で発散するため明示的に inf を返す。
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(_DATA_RANGE * _DATA_RANGE / mse))
