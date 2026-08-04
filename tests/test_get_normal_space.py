import pytest
import pandas as pd

'''
To Do: 

1. Consider what else to test (edge-cases) and prototype it.
2. Head to 'test_get_snrs.py'
'''
@pytest.fixture
def test_data():
    return pd.DataFrame({'A': (3, 2, 6, 7),
                          'B': (2, 1, 1, 7),
                          'C': (3, 4, 9, 10)})

def test_normal_space(test_data):
    ...

def test_singular_matrix():
    ...