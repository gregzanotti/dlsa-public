import os

import numpy as np
import pandas as pd
import pytest

from models.OUFFN import OUFFN
from models.RawFFN import RawFFN
from preprocess import preprocess_cumsum, preprocess_ou
from train_test import test as run_test
from train_test import train

# (T - length_training) must not be a multiple of retrain_freq, otherwise
# train_test.test() builds an empty last subperiod and fails
T, N, LOOKBACK = 150, 20, 10


@pytest.fixture
def residuals():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.01, size=(T, N))


@pytest.mark.parametrize(
    "model,preprocess,model_cfg",
    [
        (OUFFN, preprocess_ou, dict(lookback=LOOKBACK, hidden_units=[4, 4], dropout=0.0)),
        (RawFFN, preprocess_cumsum, dict(lookback=LOOKBACK, hidden_units=[LOOKBACK, 4], dropout=0.0)),
    ],
    ids=["OUFFN", "RawFFN"],
)
def test_rolling_train_test_cpu(tmp_path, residuals, model, preprocess, model_cfg):
    config = {"model": model_cfg, "optimizer_name": "Adam", "optimizer_opts": {"lr": 1e-3}}
    dates = pd.bdate_range("2000-01-03", periods=T).to_pydatetime()
    returns, sharpe, ret, std, turnovers, shorts = run_test(
        residuals, dates, model, preprocess, config,
        num_epochs=2, batchsize=50, device="cpu", parallelize=False,
        output_path=str(tmp_path), model_tag="smoke",
        lookback=LOOKBACK, retrain_freq=30, length_training=80,
        log_dev_progress_freq=1, log_plot_freq=1,
    )
    assert returns.shape == (T - 80,)
    assert np.isfinite(returns).all() and np.isfinite(sharpe)
    assert os.path.isfile(tmp_path / "smoke_cumulative-returns.png")
    assert os.path.isfile(tmp_path / "WeightsComplete_smoke.npy")


def test_checkpoint_reload(tmp_path, residuals):
    # torch>=2.6 defaults to torch.load(weights_only=True); checkpoints must still load
    kwargs = dict(
        preprocess=preprocess_ou, data_train=residuals[:80], num_epochs=1, batchsize=50,
        device="cpu", parallelize=False, output_path=str(tmp_path), model_tag="ckpt",
        lookback=LOOKBACK,
    )
    train(OUFFN(logdir=str(tmp_path), lookback=LOOKBACK, hidden_units=[4, 4]), force_retrain=True, **kwargs)
    assert os.path.isfile(tmp_path / "Checkpoint-0_seed_ckpt.tar")
    train(OUFFN(logdir=str(tmp_path), lookback=LOOKBACK, hidden_units=[4, 4]), force_retrain=False, **kwargs)
