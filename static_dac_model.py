#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate DAC output assuming a simple static non-linear model

@author: Arnfinn Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from configurations import quantiser_configurations

def generate_dac_output(X, QuantizerConfig, ML):
    """
    Table look-up to implement a simple static non-linear DAC model
    """
    
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QuantizerConfig)
    
    match Qtype:
        case "midtread":
            q = np.floor(X/Qstep + 0.5) # mid-tread
            c = q - np.floor(Vmin/Qstep) # mid-tread
        case "midriser":
            q = np.floor(X/Qstep) + 0.5 # mid-riser
            c = q - np.floor(Vmin/Qstep) - 0.5 # mid-riser
            
    YU = Qstep*q # ideal levels
    YM = ML[c.astype(int)] # measured levels
    
    return YU, YM
