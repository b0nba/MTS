from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass(frozen=True, eq=False)
class UnitSpace: # normal space??
    mean: pd.Series
    std: pd.Series
    corr_matrix_inv: pd.DataFrame
    z: pd.Series # consider removing z as return value - I don't think it is necessary for the next steps


def get_unit_space(dataset): # maybe after all this should be names 'get_normal_space' as it won't be really usable to get abnormal space (as this uses mean,std and cov_inv from normal group)

    mean = dataset.mean()
    std = dataset.std()

    z_scores = (dataset - mean) / std

    corr_matrix = dataset.corr()
    corr_inv = pd.DataFrame(
        np.linalg.inv(corr_matrix.values),
        index=corr_matrix.columns,
        columns=corr_matrix.columns,
    )

    return UnitSpace(mean=mean, std=std, corr_matrix_inv=corr_inv, z=z_scores) # Do I really need z scores? I can easily reconstruct them later

def get_md(us: UnitSpace):
    return np.diag(us.z @ us.corr_matrix_inv@ us.z.T) / len(us.z)

def check_validity_prototype(df): # this is propably fine but I would add two methods inside of it (one for md~1, second for SNR (normal) <> SNR (abnormal)
    md_mean = df['MD'].mean()  # used for assessing reference space validity
    print("MD mean (should be ~1):", md_mean)

# Section below is for tests and debugging, it needs to be moved to a separate file thought
test = pd.DataFrame({'A': (1,6,3),
                     'B' : (4,1,2),
                    'C': (5,7,2)})

m_space = get_unit_space(test)
md_vec = get_md(m_space)
