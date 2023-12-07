'''
This file contains different helper functions that are useful
'''

# Import
import numpy as np

def hardlims(n):

    if n >= 0:
        return 1
    else:
        return -1

def fdtansig(n):
    da_dn = 1 - (np.tanh(n) ** 2)
    FDMNM = np.diag(da_dn)
    return FDMNM