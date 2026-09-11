from scipy.stats import chi2
from sklearn.base import BaseEstimator


class ChiSquareThreshold(BaseEstimator):
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, md, y=None, *, feature_count=None):
        self.threshold_ =  (chi2.ppf(1 - self.alpha, df=feature_count) / feature_count )

        return self
