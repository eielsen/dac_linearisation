#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative learning control.

@author: Bikash Adhikari
@date: 25.03.2024
@license: BSD 3-Clause
"""

# %%
import numpy as np
from scipy import linalg, signal
import sys
import random
# from configurations import quantiser_configurations

def get_control(N, N_padding, Xcs, itr, QF_M, L_M, OUT_M, Qstep, Q_levels, Qtype, lvl_dict):
    """
    INPUT:
        N         - Total length of reference/test signal (in sample numbers),
        N_padding - Padding length on each end, 
        Xcs       - Reference/test signal	
        QF_M      - Q-filtering matrix
        L_M       - Learning matrix
        OUT_M     - Output matrix/Markov parameters/Impulse responses
        Qstep     - Quantizer step size
        Q_levels  - Quantizer levels
        Qtype     - Quantizer type
        lvl-dict  - Dictionary, where, keys represent the codes and the values represent the 
                    quantized measured(or ideal) levels depending on the Qstep and Range of the quantizer.
    
    OUTPUT:
        US		- Control (Stacked)
        YS		- Output
        ES		- Error
        rmsErr	- RMS Error
    """

    pMatrix = get_periodMatrix(N, N_padding, Xcs)
    N_period, T_period = pMatrix.shape  # matrix dimensions

    US = np.empty([1, itr+1])
    YS = np.empty([1, itr+1])
    ES = np.empty([1, itr+1])

    u_init = np.ones(N_period)

    for i in range(T_period):
        ref_signal = pMatrix[:,i]
        U, Y, E, rE = get_ILC_control(ref_signal, u_init, itr, QF_M, L_M, OUT_M, Qstep, Q_levels, Qtype, lvl_dict) 

        u_init = U[:,-1]  # set initial control for the next period as the optimal control of the last period

        # u_init = np.ones(N_period)

        # Trim overlaps
        U_trim = remove_Overlap(U, N, N_padding)
        Y_trim = remove_Overlap(Y, N, N_padding)
        E_trim = remove_Overlap(E, N, N_padding)

        # Store values
        US = np.vstack((US, U_trim))
        YS = np.vstack((YS, Y_trim))
        ES = np.vstack((ES, E_trim))

    # Remove first row
    US = np.delete(US, 0,0)
    YS = np.delete(YS, 0,0)
    ES = np.delete(ES, 0,0)

    # RMS Error
    rmsErr = [] 
    for i in range(ES.shape[1]):
        rmsErr_i = np.sqrt(np.square(ES[:,i]).mean())
        rmsErr = np.append(rmsErr, rmsErr_i)
    
    return US


def get_ILC_control(Xcs, u_init, itr, QF_M, L_M, OUT_M, Qstep, Q_levels, Qtype, Mlvl_dict):
    """
    INPUTS:
        Xcs         - Reference / test signal
        u_init      - Initial control signal 
        iter        - Number of iterations
        QF_M        - Q-filtering Matrix
        L_M         - Learning Matrix 
        OUT_M       - Output Matrix
        Qstep       - Quantization step size
        Q_leves     - Quantizer leves
        Qtype       - Quantizer type; Ideal or Nonideal (with INL)
        Mlvl_dict   - Measured level dictionary
        
    OUTPUTS:
        U       - Control matrix with values from every iteration
        Y       - Ouput matrix with values from every iteration
        E       - Error matrix with values from every iteration
        rmsErr  - RMSError from every iteration

    Note: Only use the value from last iteration for simulation 
    U, Y, E [:,-1]: Each column represent each iteration.
    """

    # Make test signal column vector
    Xcs = Xcs.reshape(-1,1)

    # Initial Control 
    # u = np.ones_like(Xcs)
    u = u_init.reshape(-1,1)
    U = u

    # Intial Output
    y = OUT_M @ u
    Y = y

    # Initial Error 
    e = Xcs - y  # reference/test signal - output signal
    E = e

    # RMS error 
    rmse = np.sqrt((e**2).mean())
    rmsErr = rmse

    for i in range(itr):
        # Update control using ILC algorithm, Q filter matrix and Learning matrix
        u_new = QF_M @(u + L_M @ e)       

        # Quantize control
        q_u_new = direct_quant(u_new, Qstep, Q_levels, Qtype)               

        # Convert quantized signal to code
        Vmin = np.min(Q_levels)
        q_u_new_code = gen_code(q_u_new, Qstep, Vmin, Qtype).squeeze()

        # Parsing measured levels according to the code
        q_u_new_dac = gen_dac_output(q_u_new_code, Mlvl_dict)
        q_u = np.array(q_u_new_dac).reshape(-1,1)

        # Output 
        y = OUT_M @ q_u
        y = y.reshape(-1,1)

        # Error 
        e = Xcs - y 

        # Store values 
        Y = np.hstack((Y, y))
        U = np.hstack((U, u_new))
        E = np.hstack((E, e))         

        # Rms Error 
        rmse = np.sqrt(((e**2).mean()))
        rmsErr = np.hstack((rmsErr, rmse))

        # Update control 
        u = u_new.reshape(-1,1)

    return U, Y, E, rmsErr


def learning_matrices(len_X, im):
    """  Q-filter and Learning matrix generated using results from:
    D. A. Bristow, M. Tharayil and A. G. Alleyne, "A survey of iterative learning control," 
    in IEEE Control Systems, vol. 26, no. 3, pp. 96-114, June 2006
    
    INPUT:
        len_X  - Length of reference signal. Q,L,G matrix dimension should match (len_X x len_X)
        im     - filter's impulse response
    
    OUTPUT:
        Q      - Q-filtering matrix
        L      - Learning matrix
        G      - Plant output matrix
    """

    # len_X = len(X)      # Length of test signal
    h = im[0]     # Impulse response 

    # Tuning matrices
    We = np.identity(len_X)
    Wf = np.identity(len_X)*1e-4
    Wdf = np.identity(len_X)*1e-1

    RowVec = np.zeros((1, len_X))
    ColumnVec =  h[0:len_X]
    ColumnVec = np.reshape(ColumnVec, (len(ColumnVec),1))

    # Output Matrix
    G = linalg.toeplitz(ColumnVec, RowVec)

    # Q-filter and Learning Matrices
    Subinverse11 =  G.transpose() @ We @ G 
    SubinverseQ = Subinverse11 + Wf + Wdf
    SubinverseL =  Subinverse11 + Wdf
    SubinverseQ = np.linalg.inv(SubinverseQ)
    SubinverseL = np.linalg.inv(SubinverseL)

    # Q-filter matrix   
    Q = SubinverseQ @ (G.transpose()@ We @ G + Wdf) 

    # Learning matrix 
    L = SubinverseL @ (G.transpose()@We)

    # Check if the stability and convergence condition are satisfied
    ILCloop =  Q - np.matmul(L,G)
    eig_ILC, vec_ILC = np.linalg.eig(ILCloop)       
    eig_ILC_mon, vec_ILC_mon = np.linalg.eig(ILCloop @ ILCloop.transpose())

    if max(eig_ILC) <=1:
        print('Stablity Condition Satisfied')
        if max(eig_ILC_mon) <=1:
            print('ILC Monotonic Convergent Condition also Satisfied')
    else:
        sys.exit('Stability condition not satisfied. Change tuning matrices')

    return Q, L, G


def get_periodMatrix(N, N_padding, ref_signal):

    N_period = int(N + 2*N_padding)       # Total samples in each period (signal length + padding length)

    # Reference signal period  matrix: Each row contains the reference signal with N_total samples (an arbitrary period)  

    # N-length signal storage container , first entry
    period_matrix =  ref_signal[0:N_period] 

    # Transform to column matrix to store in matrix
    period_matrix = period_matrix.reshape(len(period_matrix),1) 

    # Initial index for second period
    index_init = N_period 

    if len(ref_signal) <= N_period-1:
        raise ValueError('Length of reference signal less than the lenght of the period length')
    else:
        while True:
            # indices for each iteration 
            index1 = int(index_init - N_padding)      # inital index for ith period
            index2 = int(index1 + N_period)              # final index for ith period

            # makes the periods of reference signal of same length and store them as matrix
            if index2 <= len(ref_signal):
                vec_i = np.reshape(ref_signal[index1:index2],(N_period,1)) 
                period_matrix = np.hstack((period_matrix, vec_i))
            else:
                break
            index_init = index2        # update index
    return period_matrix 


def remove_Overlap(M, N, N_padding): 
    """ Removes the overlapping due to padding

    INPUTS:
        M         -  period Matrix (i.e. the matrix that contains the segments of signals sotred as column vector on matrix M)
        N         - Length of the segment
        N_padding - Padding length
    
    OUTPUT:
        M_trim  - Trimmed matrix with overlapping due to padding removed
    """

    nrows, ncols = M.shape
    index1 = int(N_padding/2)
    index2 = int(index1 + N + N_padding/2)
    M_trim = M[index1:index1 + index2,:]
    return M_trim


def direct_quant(Xcs, Qstep, Q_levels, Qtype):
    """ Direct quatinzer
    
    INPUT:
        Xcs      - Reference signal
        Qstep    - Qantizer step size
        Q_levels - Quantizer levels
        Qtype    - Quantizer type; midread or midrise
    
    OUTPUTS:
        q_Xcs    - Quantized signal with Qstep, step size
    """
    # Range of the quantizer
    Vmax = np.max(Q_levels)
    Vmin = np.min(Q_levels)
    # Select quantizer type
    match Qtype:
        case "midtread":
            q_Xcs = np.floor(Xcs/Qstep +1/2)*Qstep
        case "midrise":
            q_Xcs = np.floor(Xcs/Qstep )*Qstep +1/2
    # Quatizer saturation within its range
    np.place(q_Xcs, q_Xcs> Vmax, Vmax)
    np.place(q_Xcs, q_Xcs < Vmin, Vmin)

    return q_Xcs



def gen_code(q_Xcs, Qstep, Vmin, Qtype):
    """ Converter the quantized values to unsigned integers

    INPUTS:
        q_Xcs       - Quantized signal
        Qstep       - Qantizer step size
        Vmin        - Quantizer lower range
        Qtype       - Quantizer type; midread or midrise
    
    OUTPUTS:
        q_code      - code corresponsing to the quantized values and quantizer levels
    """

    match Qtype:
        case "midtread":
            q_code = q_Xcs/Qstep - np.floor(Vmin/Qstep)
        case "midrise":
            q_code = q_Xcs/Qstep - np.floor(Vmin/Qstep) - 1/2
    return q_code.astype(int)


def gen_dac_output(q_codes, ML_dict):
    """ 
    INPUTS:
        q_codes       - quantized signal in codes
        ML_dict       - measured levels corresponding to the code, LUT

    OUTPUTS:
        q_dac       -  Emulated DAC output 
    """

    q_dac = []
    for i in q_codes:
        q_dac_i = ML_dict[i]    # assing value to the code
        q_dac.append(q_dac_i)
    return  q_dac


def generate_ML(Nb, Qstep, Q_levels):
    # Generate random INL for the simulation
    
    level_codes = np.arange(0, 2**Nb,1) # Levels:  0, 1, 2, .... 2^(Nb)
    inl = []
    for _ in range(2**Nb):
        inl.append(Qstep*random.randint(-10,10))
    inl = np.array(inl)
    inl[0:2] = 0
    inl[-2:] = 0
    ml = Q_levels + inl
    ML_dict = dict(zip(level_codes, ml))
    return ML_dict
