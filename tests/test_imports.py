import importlib

import pytest

MODULES = [
    "utils",
    "data",
    "preprocess",
    "train_test",
    "run_train_test",
    "run_factor_model",
    "models.OUFFN",
    "models.FourierFFN",
    "models.RawFFN",
    "models.CNNTransformer",
    "factor_models.pca",
    "factor_models.ipca",
    "factor_models.famafrench",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_numpy_bool_alias():
    # train_test.py uses np.bool, removed in numpy 1.24 and restored in numpy 2.0
    import numpy as np

    assert np.ones(2, dtype=np.bool).dtype == bool


def test_matplotlib_plot_date():
    # train_test.test() uses plt.plot_date, removed in matplotlib 3.11
    import matplotlib.pyplot as plt

    assert hasattr(plt, "plot_date")
