'''
Function to train one epoch of a 2 layer MLP with basic backpropogation (BP) and BP with momentum
'''

# Imports
import numpy as np
import bptans1pat

def bptans1e(W1, W2, b1, b2, alpha, P, T):

    cols_p = np.shape(P)[1]
    Te2 = 0
    
    for i in range(cols_p):

        W1_new, W2_new, b1_new, b2_new, avg2 = bptans1pat.bptans1pat(W1, W2, b1, b2, alpha, P[:, i], T[:, i])
        Te2 += avg2
    
        # Recirculate Values
        W1 = W1_new
        W2 = W2_new
        b1 = b1_new
        b2 = b2_new
    
    AE2 = Te2 / cols_p

    return W1_new, W2_new, b1_new, b2_new, AE2


def mbp1e(W1, W2, b1, b2, delta_W1, delta_W2, delta_b1, delta_b2, ga, alpha, P, T):

    cols_p = np.shape(P)[1]
    Te2 = 0
    
    for i in range(cols_p):

        W1_new, W2_new, b1_new, b2_new, avg2, D1, D2, c1, c2 = bptans1pat.mbp1p(W1, W2, b1, b2, alpha, P[:, i], T[:, i], delta_W1, delta_W2, delta_b1, delta_b2, ga)
        Te2 += avg2
    
        # Recirculate Values
        W1 = W1_new
        W2 = W2_new
        b1 = b1_new
        b2 = b2_new
        F1 = D1
        F2 = D2
        g1 = c1
        g2 = c2
    
    AE2 = Te2 / cols_p

    return W1_new, W2_new, b1_new, b2_new, AE2