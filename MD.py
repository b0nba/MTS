from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass(frozen=True, eq=False)
class UnitSpace: # normal space??
    mean: pd.Series
    std: pd.Series
    corr_matrix_inv: pd.DataFrame
    z: pd.Series # consider removing z as return value - I don't think it is necessary for the next steps

# since this gets used in get_abnormal and get_normal this stops to be DRY
def get_normal_space(normal_data): # maybe after all this should be names 'get_normal_space' as it won't be really usable to get abnormal space (as this uses mean,std and cov_inv from normal group)

    mean = normal_data.mean()
    std = normal_data.std()

    z_scores = (normal_data - mean) / std # consider moving it ot get_md as it is used only there

    corr_matrix = normal_data.corr()
    corr_inv = pd.DataFrame(
        np.linalg.inv(corr_matrix.values),
        index=corr_matrix.columns,
        columns=corr_matrix.columns,
    )

    return UnitSpace(mean=mean, std=std, corr_matrix_inv=corr_inv, z=z_scores) # Do I really need z scores? I can easily reconstruct them later

def get_abnormal_space(normal_space: UnitSpace, abnormal_data):

    z_scores = (abnormal_data - normal_space.mean) / normal_space.std

    corr_matrix = abnormal_data.corr()
    corr_inv = pd.DataFrame( # since this gets used in get_abnormal and get_normal this stops to be DRY
        np.linalg.inv(corr_matrix.values),
        index=corr_matrix.columns,
        columns=corr_matrix.columns,
    )

    return UnitSpace(mean=normal_space.mean, std=normal_space.std,corr_matrix_inv=corr_inv, z=z_scores)

def get_md(us: UnitSpace):
    return np.diag(us.z @ us.corr_matrix_inv@ us.z.T) / len(us.z.columns)

# larger the better by default I should implement other thought
def get_snr_prot(md):
    return -10 * np.log10(np.mean(1 / md ** 2))


def check_validity(df): # this is propably fine but I would add two methods inside of it (one for md~1, second for SNR (normal) <> SNR (abnormal)
    md_mean = df['MD'].mean()  # used for assessing reference space validity
    print("MD mean (should be ~1):", md_mean)

# Section below is for tests and debugging, it needs to be moved to a separate file thought
normal_test = pd.DataFrame({'A': (1,6,3,3),
                     'B' : (4,1,2,7),
                    'C': (5,7,2,1)})

abnormal_test = pd.DataFrame({'A': (5,6,3,5),
                              'B' : (2,4,5,8),
                              'C': (1,3,2,9)})


m_space = get_normal_space(normal_test)
ab_space = get_abnormal_space(m_space, abnormal_test)
md = get_md(ab_space)
snr = get_snr_prot(md)

print(snr)