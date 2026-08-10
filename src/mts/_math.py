from dataclasses import dataclass
import numpy as np
from dataclasses import replace

from pandas.core.dtypes import astype


@dataclass(frozen=True, eq=False)
class UnitSpace:
    mean: np.ndarray
    std: np.ndarray
    corr_matrix_inv: np.ndarray
    z: np.ndarray

'''
To Do: 

'''

def get_normal_space(normal_data: np.ndarray):

    mean = np.mean(normal_data, axis=0)
    std = np.std(normal_data,axis=0,ddof=1) # ddof set to '1' to resemble pandas "df.std()". Conceptually this sets std() as sample standard deviation.

    z_scores = (normal_data - mean) / std # consider moving it ot get_md as it is used only there

    corr_matrix = np.atleast_2d(np.corrcoef(normal_data,rowvar=False))
    corr_inv = np.linalg.inv(corr_matrix)

    return UnitSpace(mean=mean, std=std, corr_matrix_inv=corr_inv, z=z_scores)

def get_abnormal_space(normal_space: np.ndarray, abnormal_data: np.ndarray):
    z_scores = (abnormal_data - normal_space.mean) / normal_space.std
    return replace(normal_space, z=z_scores)

def get_md(us: UnitSpace):
    col_amount = us.z.shape[1]
    return np.diag(us.z @ us.corr_matrix_inv @ us.z.T) / col_amount

def get_snr_ltb(md):
    return -10 * np.log10(np.mean(1 / md ** 2))

def get_snr_stb(md):
    return -10 * np.log10(np.mean(md ** 2))

def get_snr_ntb_target(md: np.ndarray, target=1.0):
    variance = np.var(md, ddof=1)
    return 10 * np.log10(target**2 / variance)

def check_validity(md_normal: np.ndarray, md_abnormal: np.ndarray):
    md_mean_n = np.mean(md_normal)
    snr = get_snr_ntb_target(md_normal)
    print("\nMD mean (should be ~ 1):", md_mean_n)
    print("SNR:", snr)

    md_mean_ab = np.mean(md_abnormal)

    print(f"MD mean for normal space is:   {md_mean_n} \nMD mean for abnormal space is: {md_mean_ab}")

    separation = md_mean_ab/md_mean_n

    print(f"Separation: {separation}")

def get_snrs(oa_design: np.ndarray, normal_data: np.ndarray, ab_data: np.ndarray):

    snrs = []
    for row in oa_design:

        mask = row.astype(bool)

        sub_m_data = normal_data[:,mask]
        sub_ab_data = ab_data[:,mask]

        m_space = get_normal_space(sub_m_data) # sub-normal space

        ab_space = get_abnormal_space(m_space, sub_ab_data) # sub-abnormal space

        md = get_md(ab_space)

        snr = get_snr_ltb(md) # Using larger the better as default SNR.

        snrs.append(snr)

    snrs = np.array(snrs)
    return snrs

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

    return abs(to_compare[id_included] - to_compare[id_excluded])

def optimize_space(normal_data: np.ndarray, abnormal_data: np.ndarray, oa_design: np.ndarray):
    snrs = get_snrs(oa_design, normal_data, abnormal_data)
    compared = compare_snr(oa_design,snrs)
    deltas = get_delta(compared)

    # Make a decision based on deltas, and return optimized dataset?
    return 0

