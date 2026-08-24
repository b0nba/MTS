import numpy as np
import pandas as pd
import pytest
import mts.MTS as MTS
from sklearn.utils.estimator_checks import check_estimator


'''
To Do: 
1. Test, when OA size and data size differ.
'''


@pytest.fixture
def oa_design():
    return pd.DataFrame({'1': [1, 1, 0, 0], '2': [1, 0, 1, 0], '3': [1, 0, 0, 1]})

@pytest.fixture
def test_n_data():
    return pd.DataFrame({'A': (1, 6, 3, 3, 4, 2),
                          'B': (4, 1, 2, 7, 5, 9),
                          'C': (5, 9, 2, 1, 1, 8)})

@pytest.fixture
def y():
    return np.array([1, 0, 1, 1,  0,0])

def test_MTS_smoke(test_n_data, y, oa_design):
    mts = MTS.MTS(opt = oa_design.to_numpy(), threshold=1.0)

    xx = test_n_data.to_numpy()
    yy = y
    df = mts.fit(X=xx, y = yy)

    # check_estimator(MTS.MTS(opt=oa_design.to_numpy(),threshold=1.0)) # This is for sklean guidelines check.
    assert mts is not None
    assert df is not None