#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative learning control.

@author: Bikash Adhikari
@date: 27.03.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import linalg , signal
import sys
import random
import gurobipy as gp
from gurobipy import GRB
import tqdm

def MPC(Nb, N_PRED, Xcs, ML, Qstep,  A1, B1, C1, D1, x0):
    Qstep = abs(Qstep)
    # Scale the input and ML to make the QStep = 1
    len_MPC = len(Xcs) - N_PRED
    X = Xcs/Qstep
    ML_Scaled = ML/Qstep
    C = []
    U_MPC_OPT = []
    for j in tqdm.tqdm(range(len_MPC)):
        x_dim =  x0.size
        m = gp.Model("MPC- INL")
        u = m.addMVar((N_PRED,1), vtype=GRB.INTEGER, name= "u", lb = -2**(Nb-1), ub =  (2**Nb-1)) # control variable
        x = m.addMVar((x_dim*(N_PRED+1),1), vtype= GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "x")  # State varible 

        # Add objective function
        Obj = 0
        for i in range(N_PRED):
            k = x_dim * i
            st = x[k:k+x_dim]
            con = u[i] - X[j+i]

            # Objective
            e_t = C1 @ x[k:k+x_dim] + D1 * con
            Obj = Obj + e_t * e_t

        # Add constraints
        m.addConstr(x[0:x_dim,:] == x0)
        for i in range(0, N_PRED):
            k = x_dim*i
            st = x[k:k+x_dim]
            con = u[i]- X[j + i]

            # Constraints
            f_value = A1 @ st + B1 * con
            st_next = x[k+x_dim:k+2*x_dim]
            m.addConstr(st_next == f_value)

        m.update
        m.setObjective(Obj, GRB.MINIMIZE)
        m.Params.LogToConsole = 0
        m.optimize()

        # Extract Values
        allvars = m.getVars()
        values = m.getAttr("X",allvars)
        values = np.array(values)

        C_MPC = values[0:N_PRED]
        C_MPC = C_MPC.astype(int)
        C.append(C_MPC[0])


        # Non-uniform 
        U_opt = ML_Scaled[C_MPC[0]]
        U_MPC_OPT.append(U_opt)
        # # State update for next horizon
        x0_new = A1 @ x0 + B1 * (U_opt - X[j])  # State Prediction
        x0 = x0_new  # State Update
    
    C = np.array(C).reshape(1,-1)
    return C


#######################################################################################################
# DIRECT QUANTIZATION
#######################################################################################################
def dq(Xcs, Qstep, Qlevels, Qtype):
    # Direct quatinzer
    """ INPUTS:
    Xcs         - Reference signal
    Qstep       - Qantizer step size
    Qlevels    - Quantizer levels
    Qtype       - Quantizer type; midread or midrise
    """

    """ OUTPUTS:
    q_Xcs       - Quantized signal with Qstep, step size
    """
    # Range of the quantizer
    Vmax = np.max(Qlevels)
    Vmin = np.min(Qlevels)
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

# Nearest neighbor quantizer. Measures the distance of each signal samples with 
# all quantizer leves and assings the level that is nearest
def DirectQuantization(Qlevels, ref):
    """ INPUTS:
    Qlevels    - Quantizer levels
    ref         - Reference signal
    """

    """ OUTPUTS:
    u_d       - Quantized signal with Qstep, step size
    """
    u_d = []
    for k in range(0, len(ref)):
        e_n = []
        for i in Q:
            err = linalg.norm(ref[k]-i)
            e_n.append(err)
        
        min_err = np.min(e_n)
        ind = e_n.index(min_err)
        u_directi = Qlevels[ind]
        u_d.append(u_directi)
    return u_d 

#######################################################################################################
# GENERATE CODE:  Converter the quantized values to unsigned integers
#######################################################################################################
def gen_C(q_Xcs, Qstep, Vmin, Qtype):

    """ INPUTS:
    q_Xcs       - Quantized signal
    Qstep       - Qantizer step size
    Vmin        - Quantizer lower range
    Qtype       - Quantizer type; midread or midrise
    """

    """ OUTPUTS:
    q_code      - code corresponsing to the quantized values and quantizer levels
    """
    match Qtype:
        case 1:
            q_code = q_Xcs/Qstep - np.floor(Vmin/Qstep)
        case 2:
            q_code = q_Xcs/Qstep - np.floor(Vmin/Qstep) - 1/2
    return q_code.astype(int)


#######################################################################################################
# GENEARTE DAC OUTPUT:  Emulated the DAC by assigning the values according to the codes
#######################################################################################################
def gen_DO(q_codes, lvl_dict):
    """ INPUTS:
    q_codes       - quantized signal in codes
    ML_dict       - measured levels corresponding to the code, LUT
    """

    """ OUTPUTS:
    q_dac       -  Emulated DAC output 
    """ 
    q_dac = []
    for i in q_codes:
        q_dac_i = lvl_dict[i]    # assing value to the code
        q_dac.append(q_dac_i)
    return  q_dac



#######################################################################################################
# GENERATE NONLINEAR DAC:   Generate the quantizer leves with INL
#######################################################################################################
def gen_ML(Nb, Qstep, Qlevels):
    """ INPUTS:
    Nb            - Quantizer bits
    Qstep         - Quantizer step size
    Qlevels       - Ideal quantizer levels
    """

    """ OUTPUTS: Emulated DAC output
    ML_dict       -  Dictionary where keys represent the codes, and values represent the measured level
    """ 
    levels = np.arange(0,2**Nb,1)

    INL = Qstep*np.random.rand(2**Nb, 1)*100
    # Set initial and final value to zero, to match the ideal transfer function
    INL[0] = 0 
    INL[-1] = 0

    ML = Qlevels + INL.squeeze()
    ML_dict = dict(zip(levels, ML.squeeze()))
    return ML_dict
