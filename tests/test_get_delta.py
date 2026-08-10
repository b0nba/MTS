import numpy as np
import pandas as pd
import pytest

from mts._math import compare_snr, get_delta
from deprecated.mts.src import md as dp

@pytest.fixture
def oa_design():
    return pd.DataFrame({'1': [1, 1, 0, 0],
                         '2': [1, 0, 1, 0],
                         '3': [1, 0, 0, 1]})

@pytest.fixture
def snrs():
    return np.array([7.5, 2.5, 8.6, 3.4])

def test_get_delta_smoke(oa_design, snrs):

    snr_table = compare_snr(oa_design.to_numpy(), snrs)
    delta = get_delta(snrs)

    assert delta is not None

def test_get_delta_smoke_np_vs_ps(oa_design, snrs):
    snr_table = compare_snr(oa_design.to_numpy(), snrs)

    # Prepare our df so function from 'dp' won't produce an error.
    snr_table_dp = pd.DataFrame(snr_table)
    snr_table_dp.index = ["included", "excluded"]

    delta_np = get_delta(snr_table)
    delta_pd = dp.get_delta(snr_table_dp)
    
    # Convert df to numpy and compare
    assert delta_np == pytest.approx(delta_pd.to_numpy())