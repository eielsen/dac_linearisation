#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic element matching.

@author: Arnfinn Aas Eielsen
@date: 07.03.2024
@license: BSD 3-Clause
"""

import numpy as np
from numpy import matlib
import math
from scipy import signal


def ssb(c, d):
    """
    Segmenting switching block.
    
    c
        input
    d
        randomiser
    """

    # d = np.random.randint(2)

    if c % 2: # c is odd
        s = 0
    else: # c is even
        if d:
            s = 1
        else:
            s = -1

    t = (c - 1 - s)/2
    b = 1 + s

    return int(t), int(b)  # top switch, bottom switch


def nssb(c, d):
    """
    Non-segmenting switching block.

    c
        input
    d
        randomiser
    """

    # d = np.random.randint(2)

    if c % 2: # c is odd
        if d:
            s = 1
        else:
            s = -1
    else: # c is even
        s = 0
    
    t = (c - s)/2
    b = (c + s)/2

    return int(t), int(b)  # top switch, bottom switch


def dem(X, Rng, Nb):
    """
    X
        input signal
    Rng, N
        quantiser params. (for re-quantisation and code generation)
    """
    
    # DEM code input range
    M = 2*(2**Nb - 1)
    cmin = 2**(Nb-1) - 1
    cmax = M - 2**(Nb-1) + 1
    Qseg = Rng/(cmax-cmin)  # segmented step-size (LSB)

    cin = 2**(Nb-1)  # when input it bipolar an offset is needed

    # DEM mapping from output segment weights to codes
    Ks = 2**np.arange(0, Nb).astype(int)
    Ks = matlib.repmat(Ks, 2, 1)

    # Store codes
    C = np.zeros((2, X.size)).astype(int)  # individual DAC codes (1 ch. per row)
    Ci = np.zeros(X.size).astype(int)  # initial codes
    #Csum = np.zeros(X.size).astype(int)  # sum of segmented codes (verification)

    # from lin_method_dem import ssb, nssb # 

    for i in range(0, X.size):
        w = X[i]
        
        # Re-quantizer for segmented DAC
        qs = math.floor(w/Qseg + 0.5) + cin # mid-tread
        
        # Generate DEM codes
        c = qs + cmin
        Ci[i] = c
                    
        # DEM 
        Ss = np.zeros((2, Nb)).astype(int) # segment switching block results
        c1 = c # initial switching block input
        
        # random switching sequence
        d = np.random.randint(2, size=2*Nb-1) # white
        
        for j in range(0, Nb-1):
            Sst, Ssb = ssb(c1, d[2*j]) # segmenting switching
            
            c1 = Sst  # save for next iteration
            c2 = Ssb  # feed to next switching block
            
            Snt, Snb = nssb(c2, d[2*j+1])  # non-segmenting switching

            Ss[0, j] = Snt
            Ss[1, j] = Snb
        
        c2 = Sst
        Snt, Snb = nssb(c2, d[-1]) # non-segmenting switching

        Ss[0, Nb-1] = Snt
        Ss[1, Nb-1] = Snb
        
        C[:, i] = np.sum(Ss*Ks, 1)
        #Csum[i] = np.sum(C[:, i]) # (verification)

    return C