#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a varitey of periodic dither signals

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import special as spcl

def periodic_dither(t, freq=49e3, type=1):
    """
    Periodic dither generation
    """
    dither_type = type          # DITHER TYPE

    ### DEFINITIONS ###
    UNIFORM_ADF_TRI_WAVE = 1    # UNIFORM ADF (TRIANGULAR WAVE)
    TRIANGULAR_ADF = 2          # TRIANGULAR ADF
    CAUCHY_ADF = 3              # CAUCHY_ADF (when tri_wave in ±pi/2 !)
    GAUSSIAN_ADF = 4            # GAUSSIAN_ADF
    
    # Generate triangular wave (can be transformed to other dither signal shapes)
    tri_wave = (2/np.pi)*np.arcsin(np.sin(2*np.pi*freq*t)) # Triangular wave vector
    
    if (dither_type == UNIFORM_ADF_TRI_WAVE) # UNIFORM ADF (Triangular wave)
        dither_wave = tri_wave

    elif (dither_type == TRIANGULAR_ADF): # TRIANGULAR ADF
        tri_wave = 0.99999*tri_wave
        dither_wave = np.empty(tri_wave.size)
        for i in range(len(tri_wave)):
            if tri_wave[i] > 0:
                dither_wave[i] = 1 - np.sqrt(1 - tri_wave[i])
            else:
                dither_wave[i] = np.sqrt(tri_wave[i] + 1) - 1

    elif (dither_type == CAUCHY_ADF): # CAUCHY_ADF (when tri_wave in ±pi/2 !)
        dither_wave = np.tan(0.95*np.pi/2*tri_wave)

    elif (dither_type == GAUSSIAN_ADF): # GAUSSIAN_ADF
        dither_wave = np.sqrt(2)*spcl.erfinv(0.99*tri_wave)

    else: # UNIFORM ADF (TRIANGULAR WAVE)
        dither_wave = tri_wave  
    
    dither_wave = dither_wave/np.max(dither_wave)  # Normalize the dither amplitude
    
    return dither_wave
