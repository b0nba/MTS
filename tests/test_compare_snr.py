import numpy as np
import pandas as pd
import pytest

from mts.md import compare_snr

@pytest.fixture
def oa_design():
    return pd.DataFrame({'1': [1, 1, 0, 0],
                         '2': [1, 0, 1, 0],
                         '3': [1, 0, 0, 1]})

@pytest.fixture
def snrs():
    return np.array([7.5, 2.5, 8.6, 3.4])

def test_compare_snr_smoke(oa_design, snrs):
    result = compare_snr(oa_design.to_numpy(), snrs)
    print(result)

def test_compare_snr_panads_vs_nunpy():
    ...