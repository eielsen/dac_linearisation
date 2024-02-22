#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a varitey of periodic dither signals

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import special as spcl

def periodic_dither(t, fd = 49e3, dither_type = 1):
    """
    Periodic dither generation
    """
    wd = 2*np.pi*fd  # fundamental frequency
    
    # Generate triangle-wave (can be transformed to other dither signal shapes)
    dd = (2/np.pi)*np.arcsin(np.sin(wd*t))  # triangle-wave vector
    
    match dither_type:
        case 1:  # uniform ADF (triangle-wave)
            dp = dd
        case 2:  # triangular ADF
            dd = 0.99999*dd
            dp = np.empty(dd.size)
            for i in range(len(dd)):
                if dd[i] > 0:
                    dp[i] = 1 - np.sqrt(1 - dd[i])
                else:
                    dp[i] = np.sqrt(dd[i] + 1) - 1
        case 3:  # Cauchy ADF when dd in ±pi/2 !
            dp = np.tan(0.95*np.pi/2*dd)
        case 4:  # Gaussian ADF
            dp = np.sqrt(2)*spcl.erfinv(0.99*dd)
        case _  :
            dp = dd  # uniform ADF (triangle-wave)
    
    dp = dp/np.max(dp)  # dither amplitude
    
    return dp
