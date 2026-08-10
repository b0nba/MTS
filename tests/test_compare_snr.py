import numpy as np
import pandas as pd
import pytest

from mts._math import compare_snr
import deprecated.mts.src.md as dp
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

    assert result is not None

def test_compare_snr_panads_vs_numpy(oa_design, snrs):
    result_current = compare_snr(oa_design.to_numpy(), snrs)
    result_deprecated = dp.compare_snr(oa_design, snrs)

    np.testing.assert_allclose(result_current, result_deprecated)
    assert result_current == pytest.approx(result_deprecated)