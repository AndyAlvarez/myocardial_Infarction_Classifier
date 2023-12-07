'''
Functions to train a 2 layer MLP with basic backpropogation (BP) and BP with momentum
'''

# Imports
import numpy as np
import toolbox as tb

def bptans1pat(W1, W2, b1, b2, alpha, p, t):

    # Step 1: Forward Propogation and calculation of avg2

    # Input Pattern to Activation of first layer a1
    n1 = np.dot(W1, p) + b1.flatten()
    a1 = np.tanh(n1)

    # a1 to activation of layer 2 (which is the output)
    n2 = np.dot(W2, a1) + b2.flatten()
    a2 = np.tanh(n2)

    numouts = len(t)
    tminusa2 = t - a2 
    tminusa2_reshaped = tminusa2.reshape(np.shape(tminusa2)[0], 1)
    avg2 = np.dot(np.transpose(tminusa2_reshaped), tminusa2_reshaped) / numouts 
    
    # Step 2: Backpropogation of sensitivities

    s2 = -2 * np.dot(tb.fdtansig(n2), tminusa2)

    # Backpropogate (s1 is calculated using s2)
    s1 = np.dot(tb.fdtansig(n1), np.dot(W2.T, s2))


    # Step 3: Update Weights and Biases

    a1_reshaped = a1.reshape(np.shape(a1)[0], 1)
    s2_reshaped = s2.reshape(np.shape(s2)[0], 1)

    s1_reshaped = s1.reshape(np.shape(s1)[0], 1)
    p_reshaped = p.reshape(np.shape(p)[0], 1)

    W2_new = W2 - alpha * np.dot(s2_reshaped, a1_reshaped.T)
    b2_new = b2 - alpha * s2_reshaped

    W1_new = W1 - alpha * np.dot(s1_reshaped, p_reshaped.T)
    b1_new = b1 - alpha * s1_reshaped

    return W1_new, W2_new, b1_new, b2_new, avg2



def mbp1p(W1, W2, b1, b2, alpha, p, t, F1, F2, g1, g2, ga):

    # Step 1: Forward Propogation and calculation of avg2

    # Input Pattern to Activation of first layer a1
    n1 = np.dot(W1, p) + b1.flatten()
    a1 = np.tanh(n1)

    # a1 to activation of layer 2 (which is the output)
    n2 = np.dot(W2, a1) + b2.flatten()
    a2 = np.tanh(n2)

    numouts = len(t)
    tminusa2 = t - a2 
    tminusa2_reshaped = tminusa2.reshape(np.shape(tminusa2)[0], 1)
    avg2 = np.dot(np.transpose(tminusa2_reshaped), tminusa2_reshaped) / numouts 
    

    # Step 2: Backpropogation of sensitivities

    s2 = -2 * np.dot(tb.fdtansig(n2), tminusa2)

    # Backpropogate (s1 is calculated using s2)
    s1 = np.dot(tb.fdtansig(n1), np.dot(W2.T, s2))


    # Step 4: Update Weights and Biases

    a1_reshaped = a1.reshape(np.shape(a1)[0], 1)
    s2_reshaped = s2.reshape(np.shape(s2)[0], 1)

    s1_reshaped = s1.reshape(np.shape(s1)[0], 1)
    p_reshaped = p.reshape(np.shape(p)[0], 1)

    # Update for Layer 2
    D2 = (ga * F2) - ((1-ga) * (alpha * np.dot(s2_reshaped, a1_reshaped.T)))
    c2 = (ga * g2) - ((1-ga) * alpha * s2_reshaped)
    W2_new = W2 + D2
    b2_new = b2 + c2

    # Update for Layer 1
    D1 = (ga * F1) - ((1-ga) * (alpha * np.dot(s1_reshaped, p_reshaped.T)))
    c1 = (ga * g1) - ((1-ga) * alpha * s1_reshaped)
    W1_new = W1 + D1
    b1_new = b1 + c1

    return W1_new, W2_new, b1_new, b2_new, avg2, D1, D2, c1, c2