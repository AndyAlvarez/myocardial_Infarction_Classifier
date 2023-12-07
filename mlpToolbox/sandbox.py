'''
This file is used to test code
'''
# Imports 
import numpy as np
import pandas as pd
import toolbox as tb
import matplotlib.pyplot as plt

# Variables
maxepochs = 60
alpha = 0.0001
ga = 0.8
H = 50

# Data 
P_TR = pd.read_csv(r'datasets/P_TR.csv', header=None)
P_TS = pd.read_csv(r'datasets/P_TS.csv', header=None)
P_TT = pd.read_csv(r'datasets/P_TT.csv', header=None)

T_TR = pd.read_csv(r'datasets/T_TR.csv', header=None)
T_TS = pd.read_csv(r'datasets/T_TS.csv', header=None)
T_TT = pd.read_csv(r'datasets/T_TT.csv', header=None)

P_TR = P_TR.to_numpy()
P_TS = P_TS.to_numpy()
P_TT = P_TT.to_numpy()

T_TR = T_TR.to_numpy()
T_TS = T_TS.to_numpy()
T_TT = T_TT.to_numpy()


# MLP Script
def mlpMomentumBP(P, T, hidden_layers, alpha, ga, max_epoch, output_layers=1, VALXX=None, VALY=None):

    LCRV = np.zeros((max_epoch))
    validation_LCRV = np.zeros((max_epoch))
    val = False

    rows_p = np.shape(P)[0]
    
    # Initialize random weights and biases between -0.5 and 0.5
    W1 = np.random.randn(hidden_layers, rows_p) / 6
    W2 = np.random.randn(output_layers, hidden_layers) / 6
    b1 = np.random.randn(hidden_layers, 1) / 6
    b2 = np.random.randn(output_layers, 1) / 6

    # Initialize delta weights and biases
    delta_W1 = np.zeros(np.shape(W1))
    delta_W2 = np.zeros(np.shape(W2))
    delta_b1 = np.zeros(np.shape(b1))
    delta_b2 = np.zeros(np.shape(b2))

    w1_epoch_i = []
    w2_epoch_i = [] 
    w3_epoch_i = [] 
    w4_epoch_i = [] 
    w5_epoch_i = [] 
    
    if VALXX is not None and VALY is not None:
        val = True

    for ep in range(max_epoch):
        ep = ep
        W1_new, W2_new, b1_new, b2_new, AE2 = mbp1e(W1, W2, b1, b2, delta_W1, delta_W2, delta_b1, delta_b2, ga, alpha, P, T)

        # Recirculate values
        W1 = W1_new
        W2 = W2_new
        b1 = b1_new
        b2 = b2_new

        LCRV[ep] = AE2

        w1_epoch_i.append(W1_new[0][1]) 
        w2_epoch_i.append(W1_new[0][2]) 
        w3_epoch_i.append(W1_new[0][7]) 
        w4_epoch_i.append(W1_new[0][4]) 
        w5_epoch_i.append(W2_new[0][1]) 

        # After each epoch of training, use the weights and biases obtained to classify the validation set
        if val == True:
            cols_p = np.shape(VALXX)[1]
            hits = 0
            TE2 = 0

            for i in range(cols_p):

                avg2, hit = predbp1p(W1, W2, b1, b2, VALXX[:,i], VALY[:,i])
                TE2 += avg2
                hits += hit

            val_AE2 = TE2 / cols_p
            validation_LCRV[ep] = val_AE2
        else:
            pass

    # Plotting 

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    if val == True :

        # Plotting Training and Validation MSE
        axs[0].plot(LCRV, label='Training')
        markerline, stemlines, baseline = axs[0].stem(validation_LCRV, linefmt='red', markerfmt='o', label='Validation LCRV')

        markerline.set_markerfacecolor('none')  # Hollow circles
        markerline.set_markeredgecolor('red')  # Marker edge color matches the base color
        markerline.set_markersize(4)  # Adjust marker size

        axs[0].set_title(f'MSE v. Epochs (Alpha = {alpha} ; H = {hidden_layers})')
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("MSE")
        axs[0].set_xlim(0, max_epoch)
        axs[0].legend(['Training', 'Validation'])

    else:

        # Plotting the MSE per Epoch (Learning Curve)
        axs[0].plot(LCRV)
        axs[0].set_title(f'MSE v. Epoch (Alpha = {alpha} ; H = {hidden_layers})')
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("MSE")
        axs[0].set_xlim(0, max_epoch)
        axs[0].set_ylim(0, np.max(LCRV))

        axs[1].set_xlabel("Epochs")

    # Plotting w1
    (markers, stemlines, baseline) = axs[1].stem(w1_epoch_i)
    plt.setp(stemlines, linestyle='-', color='cyan', linewidth=0 )
    plt.setp(markers, markersize=5, color='cyan', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w2
    (markers, stemlines, baseline) = axs[1].stem(w2_epoch_i)
    plt.setp(stemlines, linestyle='-', color='olive', linewidth=0 )
    plt.setp(markers, markersize=5, color='olive', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w3
    (markers, stemlines, baseline) = axs[1].stem(w3_epoch_i)
    plt.setp(stemlines, linestyle='-', color='green', linewidth=0 )
    plt.setp(markers, markersize=5, color='green', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w4
    (markers, stemlines, baseline) = axs[1].stem(w4_epoch_i)
    plt.setp(stemlines, linestyle='-', color='red', linewidth=0 )
    plt.setp(markers, markersize=5, color='red', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w5
    (markers, stemlines, baseline) = axs[1].stem(w5_epoch_i)
    plt.setp(stemlines, linestyle='-', color='brown', linewidth=0 )
    plt.setp(markers, markersize=5, color='brown', linestyle='-')
    plt.setp(baseline, visible=False)

    axs[1].set_title("5 Weights v Epoch")
    axs[1].legend(['w1', 'w2', 'w3', 'w4', 'w5'])

    plt.show()

    return W1, W2, b1, b2, AE2, validation_LCRV[-1]


# 1 Epoch
def mbp1e(W1, W2, b1, b2, delta_W1, delta_W2, delta_b1, delta_b2, ga, alpha, P, T):

    cols_p = np.shape(P)[1]
    Te2 = 0
    
    for i in range(cols_p):

        W1_new, W2_new, b1_new, b2_new, avg2, D1, D2, c1, c2 = mbp1p(W1, W2, b1, b2, alpha, P[:, i], T[:, i], delta_W1, delta_W2, delta_b1, delta_b2, ga)
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



#  mbp1p 1 - pattern calculations
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

    # W2_new = W2 - alpha * np.dot(s2_reshaped, a1_reshaped.T)
    # b2_new = b2 - alpha * s2_reshaped

    # W1_new = W1 - alpha * np.dot(s1_reshaped, p_reshaped.T)
    # b1_new = b1 - alpha * s1_reshaped

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




def predbp1p(W1, W2, b1, b2, p, t):

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

    hit = 0
    # Predict based on size of a2
    if np.shape(a2)[0] == 10:

        maxa2 = max(a2)
        newa2 = [float(x-maxa2) + 0.001 for x in a2]
        snewa2 = np.sign(newa2)

        if np.array_equal(t, snewa2):
            hit = 1
    else: 
        
        a_thr = tb.hardlims(a2)

        if t.item() == a_thr: 
            hit = 1

    return avg2, hit


W1_1, W2_1, b1_1, b2_1, AE2_1, VE_1 = mlpMomentumBP(P_TR, T_TR, H, alpha, ga, maxepochs, output_layers=1, VALXX=None, VALY=None)