"""Feature optimization strategies for MTS.

Additional optimizers are planned for future releases.
"""
import mts._math as math
from sklearn.base import BaseEstimator
from tests.test_get_snrs import oa_design
import numpy as np


class TaguchiOptimizer(BaseEstimator):
    def __init__(self,oa_design,snr_method = 'ltb',target=1.0):
        self.snr_method = snr_method
        self.oa_design = oa_design
        self.target = target

    def fit(self, X, y):

        normal_data = X[y == 1]
        abnormal_data = X[y == 0]

        snrs = self._get_snrs(normal_data, abnormal_data)
        compared = math.compare_snr(self.oa_design, snrs)
        deltas = math.get_delta(compared)

        self.selected_features = deltas > 0

        selected_normal_data = normal_data[:, self.selected_features]
        optimized_space = math.get_normal_space(selected_normal_data)

        if not np.any(self.selected_features):
            raise ValueError("MTS optimization did not select any features.")

        return self


    def get_support(self):
        return self.selected_features

    def _get_snrs(self,X, y):
        normal_data = X
        abnormal_data = y

        snrs = []
        for row in self.oa_design:
            mask = row.astype(bool)

            sub_m_data = normal_data[:, mask]
            sub_ab_data = abnormal_data[:, mask]

            m_space = math.get_normal_space(sub_m_data)  # sub-normal space

            ab_space = math.get_abnormal_space(m_space, sub_ab_data)  # sub-abnormal space

            md = math.get_md(ab_space)

            snr = math.get_snr_ltb(md)  # Using larger the better as default SNR.

            snrs.append(snr)

        snrs = np.array(snrs)
        return snrs

    def _calculate_snr(self,md):
        if self.snr_method == 'ltb':
            return -10 * np.log10(np.mean(1 / md**2))

        if self.snr_method == 'stb':
            return -10 * np.log10(np.mean(md**2))

        if self.snr_method == 'ntb':
            variance = np.var(md, ddof=1)
            return 10 * np.log10(self.target**2 / variance)