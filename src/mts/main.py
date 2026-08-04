import pandas as pd
import numpy as np

'''
1. Check comments, and remove if not needed.
2. Design MTS class, prototype all the methods.
3. Name modules inside of 'src' more appropriately.
3. After that head to test_get_md.py for further instructions.
'''
# Best approach will be to keep it in scikit-learn convention so before improving
# I have to make sure code is similar enough to scikit-learn packages.
def MTS(oa_design, data):
    '''
    I used this code as testing example:

    oa_design = pd.DataFrame({'1': [1,1,0,0], '2':[1,0,1,0], '3':[1,0,0,1]})

    snrs = get_snrs_prot(oa_design, normal_test,abnormal_test)
    compare_snr = compare_snr(oa_design,snrs)
    delta = get_delta(compare_snr)
    print(delta)

    '''
    #
    # Check if given arguments are proper
    #

    #
    # Separate normal and abnormal cases
    #
    #

    # Construct m_space (mahalanobis_space) for a normal group and check its validity
    #

    #
    # Optimize a normal space using OA.
    #

    return 0 # what should it return???

    '''
    In my opinion this function should take:
    
     oa_design, and test_data (arguments)
     
    And should produce:
     
    reduced normal space, SNR with and without a variable, delta. (returns)
    
    I should consider making it a class over a function as the final functionality my consist of many functions.
    '''


