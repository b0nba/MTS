import pandas as pd
import numpy as np
import sklearn as sk

'''
To Do: 

- In a meantime read: https://scikit-learn.org/dev/developers/contributing.html#new-contributors 
    to learn how to make this package integration-ready. 
    
1. Design MTS class, prototype all the methods.
2. Name functions inside of 'src' more appropriately.
3. Incorporate functions from src into MTS class.
4. After that head to test_get_md.py for further instructions.

'''

class MTS:

    '''

    At current state this class is in MWE (Minimal Working Example). From the user's perspective,
    method names follow sklearn conventions (fit, predict, transform). Full sklearn compatibility (BaseEstimator,
    Pipeline support) is planned for a future release.

    To do:
    1. Make this library integration ready with scikit-learn.
        a. Learn how api objects work: https://scikit-learn.org/dev/developers/develop.html#api-overview

        Note:
            Q: Shouldn't MTS use fit_transform(), since it optimizes and returns an
            optimized Mahalanobis space?

            A: MTS is not an estimator in the statistical sense. However, in
            scikit-learn an estimator is defined by its API rather than by statistical
            terminology. Therefore, the most natural integration follows the scikit-learn
            conventions:

                1. fit() builds the Mahalanobis Normal Space, evaluates orthogonal array
                   runs, and determines the optimal feature subset.
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

    '''

    #
    # Separate normal and abnormal cases
    #

    #
    # Check if given arguments are proper
    #

    #
    # Construct m_space (mahalanobis_space) for a normal group and check its validity
    #

    #
    # Optimize a normal space using OA.
    #


    def __init__(self):
        ...

    def split_dataset(self):
        ...

    def fit(self):
        ...



