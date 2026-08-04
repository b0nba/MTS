from dataclasses import dataclass
import pandas as pd
import numpy as np
from dataclasses import replace

@dataclass(frozen=True, eq=False)
class UnitSpace:
    mean: pd.DataFrame
    std: pd.DataFrame
    corr_matrix_inv: pd.DataFrame
    z: pd.DataFrame

'''
To Do: 
1. Implement nominal the better and lower the better inside of 'get_snr_prot'.
    a. Rename 'get_snr_prot' after implementation
    
2. Implement additional space validation methods inside of 'check_validity'.
3. Check comments inside of 'get_snrs' function and remove if they are no needed

'''

def get_normal_space(normal_data):
    mean = normal_data.mean()
    std = normal_data.std()

    z_scores = (normal_data - mean) / std # consider moving it ot get_md as it is used only there

    corr_matrix = normal_data.corr()
    corr_inv = pd.DataFrame(
        np.linalg.inv(corr_matrix.values),
        index=corr_matrix.columns,
        columns=corr_matrix.columns,
    )
    return UnitSpace(mean=mean, std=std, corr_matrix_inv=corr_inv, z=z_scores)

def get_abnormal_space(normal_space: UnitSpace, abnormal_data):
    z_scores = (abnormal_data - normal_space.mean) / normal_space.std
    return replace(normal_space, z=z_scores)

def get_md(us: UnitSpace):
    return np.diag(us.z @ us.corr_matrix_inv @ us.z.T) / len(us.z.columns)

# larger the better by default I should implement other thought
def get_snr_prot(md):
    return -10 * np.log10(np.mean(1 / md ** 2))

def check_validity(df):
    md_mean = df['MD'].mean()
    print("MD mean (should be ~1):", md_mean)

def get_snrs(oa_design: pd.DataFrame, data: pd.DataFrame, ab_data: pd.DataFrame):

    snrs = []
    for row in oa_design.to_numpy(): # row in oa_design.row() # why switch to_numpy(), ask user to ensure format and make it default
    # 1. Get test_data to calculate snr for
        sub_m_data = data.loc[:,row.astype(bool)]
        sub_ab_data = ab_data.loc[:,row.astype(bool)]
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

    # Return list of SNR for each of OA design rows
    snrs = np.array(snrs)
    return snrs

''' This converts pandas to numpy thus is not compatibile with sklearn standard
def get_snrs(oa_design: pd.DataFrame, data: pd.DataFrame, ab_data: pd.DataFrame):

    snrs = []
    for row in oa_design.to_numpy(): # row in oa_design.row() # why switch to_numpy(), ask user to ensure format and make it default
    # 1. Get test_data to calculate snr for
        sub_m_data = data.loc[:,row.astype(bool)]
        sub_ab_data = ab_data.loc[:,row.astype(bool)]
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

    # Return list of SNR for each of OA design rows
    snrs = np.array(snrs)
    return snrs
'''
def compare_snr(OA: pd.DataFrame, snr:pd.Series ):

    result = pd.DataFrame()
    for variable in OA.columns:

        mean_on = snr[OA[variable] == 1].mean()
        mean_off = snr[OA[variable] == 0].mean()

        result[variable] = {'included': mean_on,
                            'excluded': mean_off}

    return result

def get_delta(to_compare: pd.DataFrame):
    return abs(to_compare.loc['included'] - to_compare.loc['excluded'])
    # this one does simple subtraction to get change in snr in included vs excluded case

def optimize_space():
    ...
# Section below is for tests and debugging, it needs to be moved to a separate file thought
