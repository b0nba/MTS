from dataclasses import dataclass
import pandas as pd
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

0. Convert all functions to use numpy instead of pandas
1. Implement nominal the better and lower the better inside of 'get_snr_prot'.
    a. Rename 'get_snr_prot' after implementation    
2. Implement additional space validation methods inside of 'check_validity'.
3. Check comments inside of 'get_snrs' function and remove if they are no needed
4. After that head to main.py for further instructions.

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

# larger the better by default I should implement other thought
def get_snr_prot(md):
    return -10 * np.log10(np.mean(1 / md ** 2))

def check_validity(df):
    md_mean = df['MD'].mean()
    print("MD mean (should be ~1):", md_mean)

def get_snrs(oa_design: np.ndarray, data: np.ndarray, ab_data: np.ndarray):

    snrs = []
    for row in oa_design:
    # 0. Mask
        mask = row.astype(bool)
    # 1. Get test_data to calculate snr for
        sub_m_data = data[:,mask]
        sub_ab_data = ab_data[:,mask]
    # 2. Get sub_normal space
        m_space = get_normal_space(sub_m_data)
    # 3. Get sub_abnormal space
        ab_space = get_abnormal_space(m_space, sub_ab_data)
     # 4. Get MD's for abnormal space
        md = get_md(ab_space)
    # 5. Get SNR for abnormal space
        snr = get_snr_prot(md)
    # 6. Save SNR for CURRENT row in oa_design_row
        snrs.append(snr)

    # Return np.array of SNR for each of OA design rows
    snrs = np.array(snrs)
    return snrs

def compare_snr(oa: np.ndarray, snr: np.ndarray):

    result = []

    for variable in oa.T: # .T makes it to iterate columns instead of rows

        mask = variable.astype(bool)

        mean_on = np.mean(snr[mask])
        mean_off = np.mean(snr[~mask])

        result.append([mean_on, mean_off])

    print(result)
    return np.array(result).T

def get_delta(to_compare: pd.DataFrame):
    return abs(to_compare.loc['included'] - to_compare.loc['excluded'])
    # this one does simple subtraction to get change in snr in included vs excluded case

def optimize_space():
    ...
# Section below is for tests and debugging, it needs to be moved to a separate file thought
