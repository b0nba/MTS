import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import mts._math as m
import mts._validation as validate
import mts._threshold as threshold

class MTS(ClassifierMixin, BaseEstimator):

    def __init__(self, optimizer, threshold):
        self.optimizer = optimizer # for now a placeholder
        self.threshold = threshold

    def fit(self, x, y):

        normal_data = x[y == 1]
        abnormal_data = x[y == 0]

        validate.check_space(normal_data, abnormal_data)

        #self.optimizer_ = clone(self.optimizer)
        #self.optimizer_.fit(x,y)
        #self.selected_features_ = (self.optimizer_.get_support())
        #selected_normal_data = normal_data[:,self.selected_features_]
        #self.optimized_space_ = m.get_normal_space(selected_normal_data)

        self.optimized_space_, self.selected_features_ = m.optimize_space(
            normal_data=normal_data,
            abnormal_data=abnormal_data,
            oa_design=self.optimizer.oa_design,
        )
        #self.threshold_ = clone(self.threshold)
        #self.threshold_.fit()

        feature_count = np.sum(self.selected_features_)
        self.md_threshold_ = self.threshold.get_chi_square_threshold(feature_count, alpha=self.alpha)


        return self

    def predict(self, X):
        selected_x = X[:, self.selected_features_]

        sample_space = m.get_abnormal_space(self.optimized_space_, selected_x)
        md = m.get_md(sample_space)

        return np.where(md > self.threshold_.get_support(), 0, 1)
