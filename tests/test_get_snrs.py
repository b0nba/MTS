import numpy as np
import pandas as pd
import pytest

import mts.md as md

'''
To Do:
1. Consider what else to test (edge-cases) and prototype it.
2. Repeat this to do for every new .py file created in this folder
3. In the last file add "consider what do do next" in the last step of 'To Do' 
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

    snrs = md.get_snrs(oa_design.to_numpy(), test_n_data.to_numpy(), test_ab_data.to_numpy())

    assert snrs is not None
    assert len(snrs) == len(oa_design)
    assert np.all(np.isfinite(snrs))


