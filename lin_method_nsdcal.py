#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noise-shaping with digital calibration.

@author: Arnfinn Aas Eielsen
@date: 06.03.2024
@license: BSD 3-Clause
"""

import numpy as np
import math
from scipy import signal
from balreal import balreal

def nsdcal(X, Dq, YQns, MLns, Qstep, Vmin, Nb, QMODEL):
    """
    X
        input signal
    Dq
        re-quantiser dither
    YQns, 1d array
        ideal, uniform output levels (ideal model)
    MLns, 1d array
        measured, non-unform levels (calibration model)
    Qstep, Vmin, Nb
        quantiser params. (for re-quantisation and code generation)
    QMODEL
        choice of quantiser model
            1: ideal
            2: measured/calibrated
    """
    # Noise-shaping filter (using a simple double integrator)
    b = np.array([1, -2, 1])
    a = np.array([1, 0, 0])
    # Hns_tf = signal.TransferFunction(b, a, dt=1)  # double integrator
    Mns_tf = signal.TransferFunction(a-b, a, dt=1)  # Mns = 1 - Hns
    Mns = Mns_tf.to_ss()

    # AM = np.array([[0.0, 0.0], [1.0, 0.0]])
    # BM = np.array([[2.0], [0.0]])
    # CM = np.array([[1.0, -0.5]])
    # DM = np.array([[0.0]])

    # Make a balanced realisation.
    # Less sensitivity to filter coefficients in the IIR implementation.
    # (Useful if having to used fixed-point implementation and/or if the filter order is to be high.)
    Ad, Bd, Cd, Dd = balreal(Mns.A, Mns.B, Mns.C, Mns.D)
    # Ad, Bd, Cd, Dd = balreal(AM, BM, CM, DM)
    # Initialise state, output and error
    xns = np.zeros((Ad.shape[0], 1))  # noise-shaping filter state
    yns = np.zeros((1, 1))  # noise-shaping filter output
    e = np.zeros((1, 1))  # quantiser error

    C = np.zeros((1, X.size)).astype(int)  # save codes

    satcnt = 0  # saturation counter (generate warning if saturating)
    
    FB_ON = True  # turn on/off feedback, for testing
    
    for i in range(0, X.size):
        x = X[i]  # noise-shaper input
        d = Dq[i]  # re-quantisation dither
        
        if FB_ON: w = x - yns[0, 0]  # use feedback
        else: w = x
        
        u = w + d  # re-quantizer input
        
        # Re-quantizer (mid-tread)
        q = math.floor(u/Qstep + 0.5)  # quantize
        c = q - math.floor(Vmin/Qstep)  # code
        C[0, i] = c  # save code

        # Saturation (can't index out of bounds)
        if c > 2**Nb - 1:
            c = 2**Nb - 1
            satcnt = satcnt + 1
            if satcnt >= 10:
                print(f'warning: pos. sat. -- cnt: {satcnt}')
            
        if c < 0:
            c = 0
            satcnt = satcnt + 1
            if satcnt >= 10:
                print(f'warning: neg. sat. -- cnt: {satcnt}')
        
        # Output models
        yi = YQns[c]  # ideal levels
        ym = MLns[c]  # measured levels
        
        # Generate error
        match QMODEL:  # model used in feedback
            case 1:  # ideal
                e[0] = yi - w
            case 2:  # measured/calibrated
                e[0] = ym - w
        
        # Noise-shaping filter
        xns = Ad@xns + Bd@e  # update state
        yns = Cd@xns  # update filter output

    return C