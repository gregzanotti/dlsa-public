import numpy as np
import pytest
import torch

from preprocess import preprocess_cumsum, preprocess_fourier, preprocess_ou

T, N, LOOKBACK = 80, 12, 30


@pytest.fixture
def residuals():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 0.01, size=(T, N))
    data[:40, 0] = 0  # zeros encode missing observations
    return data


@pytest.mark.parametrize(
    "func,signal_length",
    [(preprocess_cumsum, LOOKBACK), (preprocess_fourier, LOOKBACK), (preprocess_ou, 4)],
)
def test_shapes_and_missing(residuals, func, signal_length):
    windows, idxs = func(residuals, LOOKBACK)
    assert windows.shape == (T - LOOKBACK, N, signal_length)
    assert windows.dtype == np.float32
    assert isinstance(idxs, torch.Tensor) and idxs.dtype == torch.bool
    assert idxs.shape == (T - LOOKBACK, N)
    # asset 0 is missing during the first window, so it must not be selected
    assert not idxs[0, 0]
    assert np.isfinite(windows).all()


def test_cumsum_window_starts_at_zero(residuals):
    windows, idxs = preprocess_cumsum(residuals, LOOKBACK)
    t, n = 5, 3
    expected = np.cumsum(residuals[t : t + LOOKBACK, n])
    np.testing.assert_allclose(windows[t, n], expected, rtol=1e-4, atol=1e-6)
