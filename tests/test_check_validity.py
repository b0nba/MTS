import numpy as np
import pandas as pd
import pytest

from mts.md import get_md, get_normal_space, get_abnormal_space, check_validity

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

def test_check_validity_smoke(test_n_data, test_ab_data):

    m_space = get_normal_space(test_n_data.to_numpy())
    ab_space = get_abnormal_space(m_space,test_ab_data.to_numpy())

    md_n = get_md(m_space)
    md_ab = get_md(ab_space)
    validity = check_validity(md_n,md_ab)

    assert validity is None
