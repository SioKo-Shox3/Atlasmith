"""masked_psnr のテスト(test-design-phase1.md ME1-ME5)。

品質契約に沿った定数の実測: 同一→inf / δ0.1 一様→20.00 dB / 1ch のみ δ0.1→24.77 dB /
空マスク→ValueError / マスク外改変→結果不変。
"""

from __future__ import annotations

import numpy as np
import pytest

from atlasmith.metrics import masked_psnr


def _full_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def test_me1_identical_is_inf() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((4, 4, 3)).astype(np.float32)
    assert masked_psnr(a, a.copy(), _full_mask(4, 4)) == float("inf")


def test_me2_uniform_delta_is_20db() -> None:
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = a + 0.1  # 全チャンネル一様に 0.1 ずれ → MSE = 0.01 → 20.00 dB。
    psnr = masked_psnr(a, b, _full_mask(4, 4))
    assert abs(psnr - 20.0) < 1e-6


def test_me3_single_channel_delta_is_24_77db() -> None:
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = a.copy()
    b[..., 0] += 0.1  # 3ch 中 1ch のみ 0.1 → MSE = 0.01/3 → 10*log10(300) ≈ 24.77 dB。
    psnr = masked_psnr(a, b, _full_mask(4, 4))
    assert abs(psnr - 24.7712) < 0.01


def test_me4_empty_mask_raises() -> None:
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = a + 0.1
    empty = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        masked_psnr(a, b, empty)


def test_me5_masked_out_change_is_ignored() -> None:
    rng = np.random.default_rng(1)
    a = rng.random((4, 4, 3)).astype(np.float32)
    b = a.copy()
    b[..., 0] += 0.1
    mask = _full_mask(4, 4)
    mask[0, 0] = False  # [0,0] を評価から除外。
    psnr_ref = masked_psnr(a, b, mask)
    # マスク外([0,0])だけを大きく壊す → 結果は変わらないはず。
    b_perturbed = b.copy()
    b_perturbed[0, 0] += 999.0
    psnr_perturbed = masked_psnr(a, b_perturbed, mask)
    assert psnr_ref == psnr_perturbed
