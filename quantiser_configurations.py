#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold some quantiser parameter configurations (matching various DAC implementations)

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import sys
import numpy as np

#%% 
def quantiser_configurations(QConfig):
    """
    Return specified configuration, given QConfig selector
    """
    
    match QConfig:
        case 1:
            Nb = 4 # word-size
            max_code = 2**Nb - 1; # max. code
            Vmin = -1 # volt
            Vmax = 1 # volt
            Qtype = "midtread"
        case 2:
            Nb = 6 # word-size
            max_code = 2**Nb - 1; # max. code
            Vmin = -0.3 # volt
            Vmax = 0.3 # volt
            Qtype = "midtread"
        case 3:
            Nb = 12 # word-size
            max_code = 2**Nb - 1; # max. code
            Vmin = -5 # volt
            Vmax = 5 # volt
            Qtype = "midtread"
        case 4:
            Nb = 16 # word-size
            max_code = 2**Nb - 1 # max. code
            Vmin = -10 # volt
            Vmax = 10 # volt
            Qtype = "midtread"
        case 5: # PURE DIGITAL CODE
            Nb = 12 # word-size
            max_code = 2**Nb - 1 # max. code
            Vmin = 0 # volt
            Vmax = max_code # volt
            Qtype = "midtread"
        case _:
            sys.exit("Invalid quantiser configuration selected.")

    range = Vmax - Vmin  # voltage range
    
    Qstep = range/max_code  # step-size (LSB)
    
    YQ = np.arange(Vmin,Vmax+Qstep,Qstep)  # ideal ouput levels (mid-tread quantizer)
    YQ = np.reshape(YQ, (-1, YQ.shape[0]))  # generate 2d array with 1 row
    
    return Nb, max_code, Vmin, Vmax, range, Qstep, YQ, Qtype
