import glob
import gzip
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(ROOT, "residuals", "*", "*.npy.gz")))


def test_residual_files_present():
    assert len(FILES) == 21


@pytest.mark.parametrize("path", FILES, ids=[os.path.basename(f)[:40] for f in FILES])
def test_residual_header(path):
    # Reads only the .npy header from the gzip stream (no full decompression)
    with gzip.open(path) as f:
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read_header(f)
    assert len(shape) == 2
    assert shape[1] == 9483  # same asset universe as superMask.npy
    assert dtype == np.float64


def test_supermask():
    mask = np.load(os.path.join(ROOT, "residuals", "superMask.npy"))
    assert mask.dtype == bool and mask.shape == (9483,)
