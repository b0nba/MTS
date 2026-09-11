import numpy as np
from dataclasses import replace, dataclass

@dataclass(frozen=True, eq=False)
class UnitSpace:
    mean: np.ndarray
    std: np.ndarray
    corr_matrix_inv: np.ndarray
    z: np.ndarray


def get_normal_space(normal_data: np.ndarray):

    mean = np.mean(normal_data, axis=0)
    std = np.std(normal_data,axis=0,ddof=1) # ddof set to '1' to resemble pandas "df.std()". Conceptually this sets std() as sample standard deviation.

    z_scores = (normal_data - mean) / std # consider moving it ot get_md as it is used only there

    corr_matrix = np.atleast_2d(np.corrcoef(normal_data,rowvar=False))
    corr_inv = np.linalg.inv(corr_matrix)

    return UnitSpace(mean=mean, std=std, corr_matrix_inv=corr_inv, z=z_scores)

def get_abnormal_space(normal_space: UnitSpace, abnormal_data: np.ndarray):
    z_scores = (abnormal_data - normal_space.mean) / normal_space.std
    return replace(normal_space, z=z_scores)

def get_md(us: UnitSpace):
    col_amount = us.z.shape[1]
    return np.diag(us.z @ us.corr_matrix_inv @ us.z.T) / col_amount

# Has to be here until space validation in _validation.py() is changed
def get_snr_ltb(md):
    return -10 * np.log10(np.mean(1 / md ** 2))

def get_snr_stb(md):
    return -10 * np.log10(np.mean(md ** 2))

def get_snr_ntb_target(md: np.ndarray, target=1.0):
    variance = np.var(md, ddof=1)
    return 10 * np.log10(target**2 / variance)
########################  Remove up to this point after change  #############################################

def compare_snr(oa: np.ndarray, snr: np.ndarray):

    result = []

    for variable in oa.T: # .T makes it to iterate columns instead of rows

        mask = variable.astype(bool)

        mean_on = np.mean(snr[mask])
        mean_off = np.mean(snr[~mask])

        result.append([mean_off,mean_on]) # Excluded == 'index = 0' and included == 'index = 1' to make it more readable.

    return np.array(result).T

def get_delta(to_compare: np.ndarray):

    id_excluded = 0
    id_included = 1

    return to_compare[id_included] - to_compare[id_excluded]
