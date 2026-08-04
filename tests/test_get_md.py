import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from mts.md import get_md, get_normal_space, get_abnormal_space
from statsmodels.datasets import get_rdataset
from scipy.spatial.distance import mahalanobis

'''
To Do:

1. Consider what else to test (edge-cases) and prototype it 
2. Resolve current comments

'''
@pytest.fixture
def test_data():
    return pd.DataFrame({'A': (1, 6, 3, 3),
                          'B': (4, 1, 2, 7),
                          'C': (5, 9, 2, 1)})

DATA_DIR = Path(__file__).parent / "test_data"

def test_check_negative_input(test_data):
    negative_input = test_data * -1

    m_space = get_normal_space(negative_input)
    md = get_md(m_space)

    assert np.all(md > 0)

def test_mixed_input(test_data):
    mixed_design_vec = np.array([1,-1,-1])
    mixed_input = test_data *  mixed_design_vec

    m_space = get_normal_space(mixed_input)
    md = get_md(m_space)

    assert np.all(md > 0)

# Should this be tested?
# Do I want my code to produce an error when Nan is provided or do I want to relay on dependent libraries to do so?
def test_produces_nan(test_data):
    nan_design_vec = np.array([1, np.nan, -1])
    nan_input = test_data * nan_design_vec

    m_space = get_normal_space(nan_input)
    md = get_md(m_space)

    #assert np.all(md > 0)

# For this get_normal_space needs to be tested first
def test_empty_data():
    data = np.array([])

    #m_space = get_normal_space(test_data)
    #md = get_md(m_space)

    #assert np.all(md > 0)

# Compared with (https://metricgate.com/calculator/mahalanobis-taguchi-system) MTS calculator written in R
def test_compare_md_metric_gate():
    mt_cars = pd.read_csv(DATA_DIR / "mtcars.csv", index_col=0)
    df = mt_cars[["mpg", "disp","am"]]

    normal = df[df["am"] == 1]
    abnormal = df[df["am"] == 0]

    normal_test = normal.drop(columns="am")
    abnormal_test = abnormal.drop(columns="am")

    n_space = get_normal_space(normal_test)
    normal_test = get_md(n_space)
    normal_test = normal_test.round(4)
    
    ab_space = get_abnormal_space(n_space,abnormal_test)
    abnormal_test = get_md(ab_space)
    abnormal_test = abnormal_test.round(4)

    mg_result_normal = np.array([ # copied from calculator
    0.2720, 0.2720, 0.6740, 1.0348, 0.4767, 1.5328,
    0.3091, 0.0379, 0.5845, 3.4104, 0.9208, 1.6308, 0.8443
    ])

    mg_result_abnormal = np.array([ # copied from calculator
    1.4770, 5.2614, 0.5318, 3.3945, 0.0023, 0.1339, 0.6554,
    1.1988, 1.1518, 1.1726, 1.2334, 8.3596, 7.5404, 8.4270,
    0.8295, 2.0875, 1.6965, 2.8552, 8.6212
    ])

    assert normal_test == pytest.approx(mg_result_normal)
    assert abnormal_test == pytest.approx(mg_result_abnormal)

# Compared with scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html)
def test_compare_md_scipy(test_data):

    m_space = get_normal_space(test_data)
    md_project = get_md(m_space)

    z_scores = np.asarray(m_space.z)
    s_inv = np.asarray(m_space.corr_matrix_inv)

    p = z_scores.shape[1]
    zero_vec = np.zeros(p)  # explain why second array is zeroed-out

    md_scipy = np.array(
        [mahalanobis(z, zero_vec, s_inv) ** 2 / p for z in z_scores]
    )

    assert md_project == pytest.approx(md_scipy, rel=1e-6)
