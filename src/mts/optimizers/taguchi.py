"""Feature optimization strategies for MTS.

Additional optimizers are planned for future releases.
"""
import mts._math as math
from sklearn.base import BaseEstimator
from tests.test_get_snrs import oa_design
import numpy as np


class TaguchiOptimizer(BaseEstimator):
    def __init__(
        self,
        oa_design,
        snr_method = 'ltb'):
        self.snr_method = snr_method
        self.oa_design = oa_design
    
    def fit(self, x, y):


        """
        snrs = math.get_snrs(self.oa_design, normal_data, abnormal_data)
        compared = math.compare_snr(self.oa_design, snrs)
        deltas = math.get_delta(compared)

        selected_features = deltas > 0

        if not np.any(selected_features):
            raise ValueError("MTS optimization did not select any features.")
        """

    def _snr_method(self):
        ...