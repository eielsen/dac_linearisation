#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold some quantiser parameter configurations (matching various DAC implementations)

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import sys
import numpy as np
from numpy import matlib
import os
from lin_method_util import lm, dm
import scipy

from static_dac_model import quantiser_type

class qws:  # quantiser_word_size
    w_04bit = 1
    w_06bit = 2
    w_12bit = 3
    w_16bit = 4
    w_16bit_NI_card = 5
    w_16bit_SPICE = 6
    w_6bit_ARTI = 7
    w_16bit_ARTI = 8
    w_6bit_2ch_SPICE = 9
    w_16bit_2ch_SPICE = 10


def quantiser_configurations(QConfig):
    """
    Return specified quantiser model configuration, given QConfig selector.
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
            Vmin = -1 # volt
            Vmax = 1 # volt
            Qtype = quantiser_type.midtread
        case qws.w_12bit:
            Nb = 12 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -5 # volt
            Vmax = 5 # volt
            Qtype = quantiser_type.midtread
        case qws.w_16bit:
            Nb = 16 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -1 # volt
            Vmax = 1 # volt
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
        case qws.w_16bit_ARTI:
            Nb = 16 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin =  0.022894606 # volt
            Vmax = -0.020022067 # volt
            Qtype = quantiser_type.midtread
        case qws.w_6bit_2ch_SPICE:
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -8.00371104e-05 # volt
            Vmax = 7.99005702e-05 # volt
            Qtype = quantiser_type.midtread
        case qws.w_16bit_2ch_SPICE:
            Nb = 16 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin =  -0.08022664 # volt
            Vmax = 0.08024051 # volt
            Qtype = quantiser_type.midtread
        case _:
            sys.exit("Invalid quantiser configuration selected.")

    Rng = Vmax - Vmin  # voltage range
    
    Qstep = Rng/Mq  # step-size (LSB)
    
    YQ = np.arange(Vmin, Vmax+Qstep, Qstep)  # ideal ouput levels (mid-tread quantizer)
    YQ = np.reshape(YQ, (-1, YQ.shape[0]))  # generate 2d array with 1 row
    
    return Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype



def get_measured_levels(QConfig, lmethod=lm.BASELINE):
    """
    Load measured or generated output levels for a given quanstiser model.
    """

    inpath = 'measurements_and_data'
    infile = ''

    match QConfig:
        case qws.w_06bit:  # re-generate ideal levels
            Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)
            Nch = 2
            ML = matlib.repmat(YQ, Nch, 1)
        case qws.w_16bit:  # re-generate ideal levels
            Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)
            Nch = 2
            ML = matlib.repmat(YQ, Nch, 1)
        case qws.w_16bit_NI_card:  # load measured levels for given qconfig
            # load measured levels given linearisation method (measured for a given physical set-up)
            match lmethod:
                case lm.BASELINE | lm.DEM | lm.NSDCAL | lm.SHPD | lm.PHFD | lm.ILC | lm.MPC:
                    infile = 'level_measurements.mat'
                    if os.path.exists(os.path.join(inpath, infile)):
                        mat_file = scipy.io.loadmat(os.path.join(inpath, infile))
                    else: # can't recover from this
                        raise SystemExit('No level measurements file found.')
                        #sys.exit('No level measurements file found.')
                    
                    # static DAC model output levels, one channel per row
                    ML = mat_file['ML']  # measured levels

                case lm.PHYSCAL:
                    infile = 'PHYSCAL_level_measurements_set_2.mat'
                    if os.path.exists(os.path.join(inpath, infile)):
                        mat_file = scipy.io.loadmat(os.path.join(inpath, infile))
                    else: # can't recover from this
                        raise SystemExit('No level measurements file found.')

                    ML_1 = mat_file['PRILVLS'][0]  # measured levels for channel 1
                    ML_2 = mat_file['SECLVLS'][0]  # measured levels for channel 2

                    # static DAC model output levels, one channel per row
                    ML = np.stack((ML_1, ML_2))
            return ML
        case qws.w_16bit_SPICE:
            infile = 'SPICE_levels_16bit.npy'
        case qws.w_6bit_ARTI:
            infile = 'SPICE_levels_ARTI_6bit.npy'
        case qws.w_16bit_ARTI:
            CSV_file = os.path.join(inpath, 'ARTI_cs_dac_16b_levels.csv')
            infile = 'SPICE_levels_ARTI_16bit.npy'
            if (os.path.exists(os.path.join(inpath, infile)) is False):
                if (os.path.exists(CSV_file) is True):
                    ML = np.transpose(np.genfromtxt(CSV_file, delimiter=',', skip_header=1))[2:,:]
                    np.save(os.path.join(inpath, infile), ML)
                    return ML
        case qws.w_6bit_2ch_SPICE:
            infile = 'cs_dac_06bit_2ch_DC_levels.npy'
        case qws.w_16bit_2ch_SPICE:
            infile = 'cs_dac_16bit_2ch_DC_levels.npy'
    
    if os.path.exists(os.path.join(inpath, infile)):
        ML = np.load(os.path.join(inpath, infile))
    else: # can't recover from this
        raise SystemExit('No level measurements file found.')
    
    return ML




"""
case 1:  # load some generated levels
            infile_1 = os.path.join(os.getcwd(),
                                    'generated_output_levels',
                                    f'generated_output_levels_{Nb}_bit_{1}_QuantizerConfig_{QConfig}.npy')
            infile_2 = os.path.join(os.getcwd(),
                                    'generated_output_levels',
                                    f'generated_output_levels_{Nb}_bit_{2}_QuantizerConfig_{QConfig}.npy')

            if os.path.exists(infile_1):
                ML_1 = np.load(infile_1)  # generated/"measured" levels for ch. 1
            else:
                # can't recover from this
                sys.exit("YQ_1 - No level file found.")
            if os.path.exists(infile_2):
                ML_2 = np.load(infile_2)  # generated/"measured" levels for ch. 2
            else:
                # can't recover from this
                sys.exit("YQ_2 - No level file found.")

            ML = np.stack((ML_1, ML_2))
"""