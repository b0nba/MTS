import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import mts._math as m
import mts._validation as validate
import mts._threshold as threshold

class MTS(ClassifierMixin, BaseEstimator):

    def __init__(self, opt, alpha = 0.05):
        self.opt = opt # for now a placeholder
        self.alpha = alpha # for now a placeholder

    def fit(self, X, y):

        normal_data = X[y == 1]
        abnormal_data = X[y == 0]

        validate.check_space(normal_data, abnormal_data)

        self.optimized_space_, self.selected_features_ = m.optimize_space(
            normal_data=normal_data,
            abnormal_data=abnormal_data,
            oa_design=self.opt,
        )

        feature_count = np.sum(self.selected_features_)
        self.md_threshold_ = threshold.get_chi_square_threshold(feature_count, alpha=self.alpha)


        return self

    def predict(self, X):
        selected_x = X[:, self.selected_features_]

        sample_space = m.get_abnormal_space(self.optimized_space_, selected_x)
        md = m.get_md(sample_space)

        return np.where(md > self.md_threshold_, 0, 1)
