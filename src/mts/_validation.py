import numpy as np

import mts._math as m

def check_validity(
    md_normal: np.ndarray,
    md_abnormal: np.ndarray
):
    md_mean_n = np.mean(md_normal)

    snr = m.get_snr_ntb_target(
        md_normal
    )

    print(
        "\nMean MD of normal space (expected to be close to 1):",
        md_mean_n
    )

    print(
        "SNR:",
        snr
    )

    md_mean_ab = np.mean(
        md_abnormal
    )

    print(
        f"MD mean for normal space is:   {md_mean_n}\n"
        f"MD mean for abnormal space is: {md_mean_ab}"
    )

    separation = (
        md_mean_ab /
        md_mean_n
    )

    print(
        f"Separation: {separation}"
    )

def check_space(x, y):

    normal_data = x[y == 1]
    abnormal_data = x[y == 0]

    m_space = m.get_normal_space(normal_data)

    ab_space = m.get_abnormal_space(
        m_space,
        abnormal_data
    )

    md_normal = m.get_md(m_space)
    md_abnormal = m.get_md(ab_space)

    check_validity(
        md_normal,
        md_abnormal
    )


