import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import mts.md as md
from statsmodels.datasets import get_rdataset
from scipy.spatial.distance import mahalanobis

'''
To Do:

1. Come back here when 'get_normal_space' is turned to be compatible with scikit-learn (change pandas to numpy)
'''
@pytest.fixture
def oa_design():
    return pd.DataFrame({'1': [1, 1, 0, 0], '2': [1, 0, 1, 0], '3': [1, 0, 0, 1]})

@pytest.fixture
def test_n_data():
    return pd.DataFrame({'A': (1, 6, 3, 3),
                          'B': (4, 1, 2, 7),
                          'C': (5, 9, 2, 1)})
@pytest.fixture
def test_ab_data():
    return pd.DataFrame({'A': (5,6,3,5),
                         'B' : (2,4,5,8),
                         'C': (1,3,2,9)})

# Test to be removed later for now it only checks if the function is running
def test_get_snrs_smoke(oa_design, test_n_data, test_ab_data):

    snrs = md.get_snrs(oa_design, test_n_data, test_ab_data)

    assert snrs is not None
    assert len(snrs) == len(oa_design)
    assert np.all(np.isfinite(snrs))
