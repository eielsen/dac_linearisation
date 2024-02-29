#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a varitey of periodic dither signals

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import special as spcl

def periodic_dither(t, freq=49e3, dither_type=1):
    """
    Periodic dither signal generation

    Parameters
    ----------
    t
        time vector
    freq
        dither signal frequency
    dither_type
        chose the dither type (amplitude distribution function)

    Returns
    -------
    dither_signal
        the dither signal
    
    Raises
    ------
    No error handling.

    Examples
    --------
    TODO: Make an example
    """
    dither_type = type  # chose the dither type (amplitude distribution function)

    ### DEFINITIONS ###
    class adf: #  amplitude distribution function
        uniform = 1  # use a triangle wave
        triangular = 2
        cauchy = 3  # when triangle wave amplitude is ±pi/2
        gaussian = 4
    
    # Generate triangle wave (can be transformed to other dither signal shapes)
    triangle_wave = (2/np.pi)*np.arcsin(np.sin(2*np.pi*freq*t)) # triangle wave vector
    
    match dither_type:
        case adf.uniform:
            dither_signal = triangle_wave
        case adf.triangular:
            triangle_wave = 0.99999*triangle_wave
            dither_signal = np.empty(triangle_wave.size)
            for i in range(len(triangle_wave)):
                if triangle_wave[i] > 0:
                    dither_signal[i] = 1 - np.sqrt(1 - triangle_wave[i])
                else:
                    dither_signal[i] = np.sqrt(triangle_wave[i] + 1) - 1
        case adf.cauchy:
            dither_signal = np.tan(0.95*np.pi/2*triangle_wave)
        case adf.gaussian:
            dither_signal = np.sqrt(2)*spcl.erfinv(0.99*triangle_wave)
        case _:  # default to triangle wave
            dither_signal = triangle_wave  
    
    dither_signal = dither_signal/np.max(dither_signal)  # normalize the dither amplitude
    
    return dither_signal
