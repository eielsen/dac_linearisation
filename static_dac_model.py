#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate DAC output assuming a simple static non-linear model

@author: Arnfinn Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from configurations import quantiser_configurations

def quantise_signal(w, Qstep, Qtype):
    """
    Quantise a signal with given quantiser specifications
    """
    
    match Qtype:
        case "midtread":
            q = np.floor(w/Qstep + 0.5) # truncated/quantised value, mid-tread
        case "midriser":
            q = np.floor(w/Qstep) + 0.5 # truncated/quantised value, mid-riser
    
    return q

def generate_codes(q, Qstep, Qtype, Vmin):
    """
    Generate codes for quantised signal with given quantiser specifications
    """
    
    match Qtype:
        case "midtread":
            c = q - np.floor(Vmin/Qstep) # code, mid-tread
        case "midriser":
            c = q - np.floor(Vmin/Qstep) - 0.5 # code, mid-riser
    
    return c.astype(int)

def generate_dac_output(C, ML):
    """
    Table look-up to implement a simple static non-linear DAC model

    Parameters
    ----------
    C
        input codes, one channel per row, must be integers, 2d array
    ML
        static DAC model output levels, one channel per row, 2d array

    Returns
    -------
    Y
        emulated DAC output
    """
    
    if C.shape[0] > ML.shape[0]:
        print(C.shape[0])
        print(ML.shape[0])
        raise ValueError('Not enough channels in model.')

    Y = np.zeros(C.shape)
    
    match 2:
        case 1: # use loops
            for k in range(0,C.shape[0]):
                for j in range(0,C.shape[1]):
                    c = C[k,j]
                    ml = ML[k,c]
                    Y[k,j] = ml
        case 2: # use numpy indexing
            for k in range(0,C.shape[0]):
                Y[k,:] = ML[k,C[k,:]]
        
    return Y
