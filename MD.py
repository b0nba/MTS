import pandas as pd
import numpy as np

class MSpace:
    def __init__(self):
        self.mean = pd.DataFrame()
        self.cov = pd.DataFrame()
        self.corr_matrix = pd.DataFrame()


def GetMahalanobisSpace(data): # <- consider changing argument 'data' to sth more descriptive
    tested_set = data.iloc[:, :] # <- do I need this variable or should I work on 'data' directly

    mean = tested_set.mean()
    std = tested_set.std()

    z_scores = (tested_set - mean) / std

    corr_matrix = tested_set.corr()
    corr_inv = np.linalg.inv(corr_matrix.values)

    m_space = MSpace()
    m_space.mean = mean
    m_space.std = std
    m_space.corr_matrix = corr_inv

    # Code above and below do two different things.
    # I should consider refactoring this function into 2 separate ones.

    z = z_scores.values
    m_d = np.diag(z @ corr_inv @ z.T) / len(data.columns)

    df = pd.DataFrame(tested_set, columns=data.columns) # <- should this
    df['MD_distance'] = m_d # <- and this be inside of THIS function ?
    # also shouldn't I save MD_distance inside 'data' directly?


    return m_space, df # should I return 'df' as a separate output or maybe put this inside 'm_space' class

def CheckValidityPrototype(df):
    md_mean = df['MD'].mean()  # used for assessing reference space validity
    print("MD mean (should be ~1):", md_mean)