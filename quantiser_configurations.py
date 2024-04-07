#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold some quantiser parameter configurations (matching various DAC implementations)

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import sys
import numpy as np

from static_dac_model import quantiser_type

class qws:  # quantiser_word_size
    w_04bit = 1
    w_06bit = 2
    w_12bit = 3
    w_16bit_NI_card = 4
    w_16bit_SPICE = 5
    w_6bit_ARTI = 6


def quantiser_configurations(QConfig):
    """
    Return specified configuration, given QConfig selector
    """
    
    match QConfig:
        case qws.w_04bit:
            Nb = 4 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -1 # volt
            Vmax = 1 # volt
            Qtype = quantiser_type.midtread
        case qws.w_06bit:
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -0.3 # volt
            Vmax = 0.3 # volt
            Qtype = quantiser_type.midtread
        case qws.w_12bit:
            Nb = 12 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -5 # volt
            Vmax = 5 # volt
            Qtype = quantiser_type.midtread
        case qws.w_16bit_NI_card:
            Nb = 16 # word-size
            Mq = 2**Nb - 1 # max. code
            Vmin = -10 # volt
            Vmax = 10 # volt
            Qtype = quantiser_type.midtread
        case qws.w_16bit_SPICE:
            Nb = 16 # word-size
            Mq = 2**Nb - 1 # max. code
            Vmin = -8 # volt
            Vmax = 8 # volt
            Qtype = quantiser_type.midtread
        case qws.w_6bit_ARTI:
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin =  0.020651606 # volt
            Vmax = -0.019920569 # volt
            Qtype = quantiser_type.midtread
        case _:
            sys.exit("Invalid quantiser configuration selected.")

    Rng = Vmax - Vmin  # voltage range
    
    Qstep = Rng/Mq  # step-size (LSB)
    
    YQ = np.arange(Vmin,Vmax+Qstep,Qstep)  # ideal ouput levels (mid-tread quantizer)
    YQ = np.reshape(YQ, (-1, YQ.shape[0]))  # generate 2d array with 1 row
    
    return Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype
