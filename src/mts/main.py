import pandas as pd
import numpy as np
import sklearn as sk

'''
To Do: 

1. Read LogisticRegression() class from sklearn
    a. Resolve comments starting with "-".
    b. Maybe there is no don't need to delve into sklearn so much, I should read guidelines for contributors first! 
    
2. Design MTS class, prototype all the methods.
3. Name functions inside of 'src' more appropriately.
4. Incorporate functions from src into MTS class.
3. After that head to test_get_md.py for further instructions.
'''

class MTS:

    '''
    Example class construction from sklearn (LogisticRegression):

        fit():
        Does the following (step by step):
            1. Checks given inputs.
                a. Produces a warnings when given input is deprecated.
                b. Produces a warning when given input is invalid or not supported.
                d. Produces a warning when given inputs contradict each other.
                c. Sets local (not self) variables values.
                - Check if that's all ???
            2. Validates the data.
                a. Calling sklearn validate_data()* function
                    <- This function essentially
                b. Calling check_classification_targets()
                    "Ensure that target y is of a non-regression type." <- In other words ensures there isn't 'too many' classes in 'y'.
                c.
            3.
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
    



