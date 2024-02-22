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
def quantiser_configurations(QuantizerConfig):
    """
    Return specified configuration.
    """

    match QuantizerConfig:
        case 1:
            Nb = 4 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -1 # volt
            Vmax = 1 # volt
            Qtype = "midtread"
        case 2:
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = 0 # volt
            Vmax = 5 # volt
            Qtype = "midtread"
        case 3:
            Nb = 12 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -5 # volt
            Vmax = 5 # volt
            Qtype = "midtread"
        case 4:
            Nb = 16 # word-size
            Mq = 2**Nb - 1 # max. code
            Vmin = -10 # volt
            Vmax = 10 # volt
            Qtype = "midtread"
        case _:
            sys.exit("Invalid quantiser configuration selected.")

    Rng = Vmax - Vmin # voltage range
    
    Qstep = Rng/Mq # step-size (LSB)
    
    YQ = np.arange(Vmin,Vmax+Qstep,Qstep) # ideal ouput levels (mid-thread quantizer)
    
    return Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype
