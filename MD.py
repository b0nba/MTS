from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass(frozen=True, eq=False)
class UnitSpace: # normal space??
    mean: pd.Series
    std: pd.Series
    corr_matrix_inv: pd.DataFrame
    z: pd.Series # consider removing z as return value - I don't think it is necessary for the next steps

# since this gets used in get_abnormal and get_normal this stops to be DRY??
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

def check_validity(df): # this is probably fine, but I would add two methods inside of it (one for md~1, second for SNR (normal) <> SNR (abnormal)
    md_mean = df['MD'].mean()  # used for assessing reference space validity
    print("MD mean (should be ~1):", md_mean)

# This takes OA design and gets SNR for each row so name it more appropriately
# Do I want to have current output format? It resembles classical orthogonal array so output might be easier to read, but I should
# make myself sure it won't be pain in the 'a' when it comes to readable and efficient code
def get_snrs_prot(oa_design: pd.DataFrame, data: pd.DataFrame, ab_data: pd.DataFrame):

    snrs = []
    for row in oa_design.to_numpy(): # row in oa_design.row() # why switch to_numpy(), ask user to ensure format and make it default
        # 1. Get data to calculate snr for
        sub_m_data = data.loc[:,row.astype(bool)]
        sub_ab_data = ab_data.loc[:,row.astype(bool)]
        #print(row)
        #print(sub_m_data) # for testing only
        # 2. Get sub_normal space
        m_space = get_normal_space(sub_m_data)
        # 3. Get sub_abnormal space
        ab_space = get_abnormal_space(m_space, sub_ab_data)
        # 4. Get MD's for abnormal space
        md = get_md(ab_space)
        # 5. Get SNR for abnormal space
        snr = get_snr_prot(md)
        # 6. save SNR for CURRENT row in oa_design_row
        snrs.append(snr)
        #print("SNR:", snr)
        # 8. Return list of SNR for OA design rows
    snrs = np.array(snrs)
    return snrs

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
normal_test = pd.DataFrame({'A': (1,6,3,3),
                     'B' : (4,1,2,7),
                    'C': (5,9,2,1)})

abnormal_test = pd.DataFrame({'A': (5,6,3,5),
                              'B' : (2,4,5,8),
                              'C': (1,3,2,9)})

m_space = get_normal_space(normal_test)
ab_space = get_abnormal_space(m_space, abnormal_test)
md = get_md(ab_space)
snr = get_snr_prot(md)

oa_design = pd.DataFrame({'1': [1,1,0,0], '2':[1,0,1,0], '3':[1,0,0,1]})

snrs = get_snrs_prot(oa_design, normal_test,abnormal_test)
compare_snr = compare_snr(oa_design,snrs)
delta = get_delta(compare_snr)
print(delta)