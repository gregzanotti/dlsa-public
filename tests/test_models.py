import pytest
import torch

from models.CNNTransformer import CNNTransformer
from models.FourierFFN import FourierFFN
from models.OUFFN import OUFFN
from models.RawFFN import RawFFN

LOOKBACK = 30
N = 50

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

# (model class, kwargs, input feature length)
CASES = [
    (OUFFN, dict(hidden_units=[4, 4, 4, 4]), 4),
    (FourierFFN, dict(hidden_units=[LOOKBACK, 16, 8, 4]), LOOKBACK),
    (RawFFN, dict(hidden_units=[LOOKBACK, 16, 8, 4]), LOOKBACK),
    (CNNTransformer, dict(filter_numbers=[1, 8], attention_heads=4, hidden_units_factor=2, hidden_units=16), LOOKBACK),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls,kwargs,n_features", CASES, ids=[c[0].__name__ for c in CASES])
def test_forward_backward(tmp_path, device, cls, kwargs, n_features):
    model = cls(logdir=str(tmp_path), lookback=LOOKBACK, device=device, **kwargs).to(device)
    x = torch.randn(N, n_features, device=device)
    out = model(x)
    assert out.shape == (N,)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
