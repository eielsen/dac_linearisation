#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate DAC output assuming a simple static non-linear model

@author: Arnfinn Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
try:
    from repos.DAC_Linearisation.configurations import quantiser_configurations
except:
    from configurations import quantiser_configurations

def generate_dac_output(input, QuantizerConfig, measured):
    """
    Table look-up to implement a simple static non-linear DAC model
    """
    
    Nb, Mq, Vmin, Vmax, Rng, LSb, YQ, Qtype = quantiser_configurations(QuantizerConfig)
    
    match Qtype:
        case "midtread":
            q = np.floor(input/LSb + 0.5) # mid-tread
            c = q - np.floor(Vmin/LSb) # mid-tread
        case "midriser":
            q = np.floor(input/LSb) + 0.5 # mid-riser
            c = q - np.floor(Vmin/LSb) - 0.5 # mid-riser
            
    ideal_output = LSb*q # ideal levels
    measured_output = measured[c.astype(int)] # measured levels
    
    return ideal_output, measured_output
