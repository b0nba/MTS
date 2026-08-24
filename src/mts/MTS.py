import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import mts._math as m
from tests.test_get_snrs import oa_design

'''
To Do: 

- In a meantime read: https://scikit-learn.org/stable/developers/develop.html#api-overview
    to learn how to make this package integration-ready. 
    
-1. First finish 'optimize_space' function in _math.py. Decide how space will be optimized.
0. Think about how to prepare MTS class for different optimization methods (eg. Bee Hive Algorithm)   
1. Design MTS class, prototype all the methods.
2. Name modules inside of 'src' more appropriately.
3. Incorporate functions from src into MTS class.
4. After that head to test_get_md.py for further instructions.

'''

class MTS(ClassifierMixin, BaseEstimator):

    '''

    At current state this class is in MWE (Minimal Working Example). From the user's perspective,
    method names follow sklearn conventions (fit, predict, transform). Full sklearn compatibility (BaseEstimator,
    Pipeline support) is planned for a future release.

    To do:
    1. Make this library integration ready with scikit-learn.
        a. Read: https://scikit-learn.org/dev/developers/develop.html#api-overview

        Note:
            Q: Shouldn't MTS use fit_transform(), since it optimizes and returns an
            optimized Mahalanobis space?

            A: MTS is not an estimator in the statistical sense. However, in
            scikit-learn an estimator is defined by its API rather than by statistical
            terminology. Therefore, the most natural integration follows the scikit-learn
            conventions:

                1. fit() builds the Mahalanobis Normal Space, evaluates orthogonal array
                   runs, and determines the optimal feature subset and threshold.
                2. transform() applies the learned feature selection to any dataset with
                    the same feature schema as the dataset provided to fit().
                3. fit_transform() naturally combines fit() and transform(), returning
                   the optimized feature space.
                4. predict() computes Mahalanobis distances in the optimized feature space and
                   classifies samples accordingly.

        fit():
            https://scikit-learn.org/dev/glossary.html#term-fit

        transform():
            https://scikit-learn.org/dev/glossary.html#term-transform

        fit_transform(): <- Do I need this?
            https://scikit-learn.org/dev/glossary.html#term-fit_transform

        predict():
            https://scikit-learn.org/dev/glossary.html#term-predict

    2. Fix broken paths in tests '.py' modules.
    '''

    def __init__(self, opt, threshold):
        self.opt = opt # opt has to be a class that is storing optimization specifics # check if it is correct with sklearn
        self.threshold = threshold # parameter how to calculate threshold

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X # AI says that this notation "X_" is used for storing 'fitted' variables. If that's true
        # check it with sklearn docs and change code below so it uses X instead of X_.
        self.y_ = y

        # This is for test only move/remove later. This prepares data for initial validation step, it should be refactored and moved up or left in this place.
        # Make sure to make it according to sklearn.
        print(self.X_[self.y_ == 1])
        m_space = m.get_normal_space(self.X_[self.y_ == 1]) # This is probably wrong check it with sklearn or do propper mapping.
        ab_space = m.get_abnormal_space(m_space, self.X_[self.y_ == 0]) # and this ofc too
        md_n = m.get_md(m_space)
        md_ab = m.get_md(ab_space)
        m.check_validity(md_n, md_ab)

        oa_d = self.opt
        # Here actual MTS is going on
        self.X_ = m.optimize_space(normal_data = self.X_[self.y_ == 1], abnormal_data = self.X_[self.y_ == 0], oa_design = oa_d) # make it to save to self instead
        # end

        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]





