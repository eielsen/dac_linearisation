#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define quantiser parameter configurations (matching various DAC implementations)

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import sys
import numpy as np
from numpy import matlib
import os
from LM.lin_method_util import lm, dm
import scipy

from utils.static_dac_model import quantiser_type


class qs:  # quantiser specification (given DAC implementation)
    w_4bit = 1
    w_6bit = 2
    w_12bit = 3
    w_16bit = 4
    w_16bit_NI_card = 5
    w_16bit_SPICE = 6
    w_6bit_ARTI = 7
    w_16bit_ARTI = 8
    w_6bit_2ch_SPICE = 9
    w_16bit_2ch_SPICE = 10
    w_16bit_6t_ARTI = 11
    w_10bit_ARTI = 12
    w_10bit_2ch_SPICE = 13
    w_6bit_ZTC_ARTI = 14
    w_10bit_ZTC_ARTI = 15


def quantiser_configurations(QConfig):
    """
    Return specified quantiser model configuration, given QConfig selector.
    """
    
    match QConfig:
        case qs.w_4bit:
            Nb = 4 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -1 # volt
            Vmax = 1 # volt
            Qtype = quantiser_type.midtread
        case qs.w_6bit:
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -1 # volt
            Vmax = 1 # volt
            Qtype = quantiser_type.midtread
        case qs.w_12bit:
            Nb = 12 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -5 # volt
            Vmax = 5 # volt
            Qtype = quantiser_type.midtread
        case qs.w_16bit:
            Nb = 16 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -1 # volt
            Vmax = 1 # volt
            Qtype = quantiser_type.midtread
        case qs.w_16bit_NI_card:
            Nb = 16 # word-size
            Mq = 2**Nb - 1 # max. code
            Vmin = -10 # volt
            Vmax = 10 # volt
            Qtype = quantiser_type.midtread
        case qs.w_16bit_SPICE:
            Nb = 16 # word-size
            Mq = 2**Nb - 1 # max. code
            Vmin = -8 # volt
            Vmax = 8 # volt
            Qtype = quantiser_type.midtread
        case qs.w_6bit_ARTI:
            # 6-bit DAC. All bits are binary-weighted
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -0.019294419 # Ampere
            Vmax = 0.019317969 # Ampere
            Qtype = quantiser_type.midtread
        case qs.w_6bit_ZTC_ARTI: # ZTC - Zero temperature coefficient
            # 6-bit DAC. All bits are binary-weighted
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -0.01990203 # Ampere
            Vmax = 0.019906069024198073 # Ampere
            #Vmin = -0.020022468 # ch1
            #Vmax = 0.020040675315428486 # ch1
            Qtype = quantiser_type.midtread
        case qs.w_10bit_ARTI:
            # 6-bit DAC. All bits are binary-weighted
            Nb = 10 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -0.018117286 # Ampere
            Vmax = 0.018076990 # Ampere
            Qtype = quantiser_type.midtread
        case qs.w_10bit_ZTC_ARTI: # ZTC - Zero temperature coefficient
            # 6-bit DAC. All bits are binary-weighted
            Nb = 10 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -0.019769005 # Ampere
            Vmax = 0.019769606441919794 # Ampere
            Qtype = quantiser_type.midtread
        case qs.w_16bit_ARTI:
            # 16-bit DAC. All bits are binary-weighted
            Nb = 16 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin =  -0.022337035 # Ampere
            Vmax = 0.022341269 # Ampere
            Qtype = quantiser_type.midtread
        case qs.w_16bit_6t_ARTI:
            # 16-bit DAC. The 10 first bits are binary-weighted, and the upper 6 bits are thermometer-weighted.
            Nb = 16 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -0.02060208 # Ampere
            Vmax = 0.020602487 # Ampere
            Qtype = quantiser_type.midtread
        case qs.w_6bit_2ch_SPICE:
            Nb = 6 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -8.00371104e-05 # volt
            Vmax = 7.99005702e-05 # volt
            Qtype = quantiser_type.midtread
        case qs.w_16bit_2ch_SPICE:
            Nb = 16 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin =  -0.08022664 # volt
            Vmax = 0.08024051 # volt
            Qtype = quantiser_type.midtread
        case qs.w_10bit_2ch_SPICE:
            Nb = 10 # word-size
            Mq = 2**Nb - 1; # max. code
            Vmin = -0.00528967841 # volt
            Vmax = 0.00528972540 # volt
            Qtype = quantiser_type.midtread
        case _:
            sys.exit("Invalid quantiser configuration selected.")

    Rng = Vmax - Vmin  # voltage range
    
    Qstep = Rng/Mq  # step-size (LSB)
    
    # YQ = np.arange(Vmin, Vmax+Qstep, Qstep)  # ideal output levels (mid-tread quantiser)
    YQ = np.linspace(Vmin, Vmax, Mq+1)
    YQ = np.reshape(YQ, (-1, YQ.shape[0]))  # generate 2d array with 1 row
    
    return Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype


def get_ML(inpath, infile, CSV_filename):
    CSV_file = os.path.join(inpath, CSV_filename)

    if (os.path.exists(os.path.join(inpath, infile)) is False): # Numpy file does not exist
        if (os.path.exists(CSV_file) is True):
            ML = np.transpose(np.genfromtxt(CSV_file, delimiter=',', skip_header=1))[1:,:]
            np.save(os.path.join(inpath, infile), ML)
            return ML
    elif os.path.exists(os.path.join(inpath, infile)): # Numpy file exists
        ML = np.load(os.path.join(inpath, infile))
        return ML
    else:
        raise SystemExit('No level measurements file found.')


def get_measured_levels(QConfig, lmethod=lm.BASELINE):
    """
    Load measured or generated output levels for a given quanstiser model.

    Arguments
        QConfig - the desired configuation
        lmethod - some configs have method dependent measurements (given lab setup)
    
    Returns
        ML - measured levels (rows corresponds to channels)
    """

    from pathlib import Path
    inpath = Path(__file__).parent / '../measurements_and_data'  # absolute path
    #inpath = '../measurements_and_data'
    infile = ''

    match QConfig:
        case qs.w_6bit:  # re-generate ideal levels
            Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)
            Nch = 2
            ML = matlib.repmat(YQ, Nch, 1)
            
        case qs.w_16bit:  # re-generate ideal levels
            Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)
            Nch = 2
            ML = matlib.repmat(YQ, Nch, 1)

        case qs.w_16bit_NI_card:  # load measured levels for given qconfig
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
        
        case qs.w_16bit_SPICE:
            infile = 'DC_levels_16bit.npy'

        case qs.w_6bit_ARTI:
            infile = 'DC_levels_ARTI_6bit.npy'
            CSV_filename = 'ARTI_cs_dac_6b_levels.csv'
            return get_ML(inpath, infile, CSV_filename)
        
        case qs.w_6bit_ZTC_ARTI:
            infile = 'DC_levels_ARTI_6bit_ZTC.npy'
            CSV_filename = 'ARTI_cs_dac_6b_ztc_levels.csv'
            return get_ML(inpath, infile, CSV_filename)
                
        case qs.w_10bit_ARTI:
            infile = 'DC_levels_ARTI_10bit.npy'
            CSV_filename = 'ARTI_cs_dac_10b_levels.csv'
            return get_ML(inpath, infile, CSV_filename)
        
        case qs.w_10bit_ZTC_ARTI:
            infile = 'DC_levels_ARTI_10bit_ZTC.npy'
            CSV_filename = 'ARTI_cs_dac_10b_ztc_levels.csv'
            return get_ML(inpath, infile, CSV_filename)
                
        case qs.w_16bit_ARTI:
            infile = 'DC_levels_ARTI_16bit.npy'
            CSV_filename = 'ARTI_cs_dac_16b_levels.csv'
            return get_ML(inpath, infile, CSV_filename)
    
        case qs.w_16bit_6t_ARTI:
            infile = 'DC_levels_ARTI_16bit_6t.npy'
            CSV_filename = 'ARTI_cs_dac_16b_6t_levels.csv'
            return get_ML(inpath, infile, CSV_filename)
        
        case qs.w_6bit_2ch_SPICE:
            infile = 'cs_dac_06bit_2ch_DC_levels.npy'

        case qs.w_16bit_2ch_SPICE:
            infile = 'cs_dac_16bit_2ch_DC_levels.npy'

        case qs.w_10bit_2ch_SPICE:
            infile = 'cs_dac_10bit_2ch_DC_levels.npy'
    
    print('Opening {}'.format(os.path.join(inpath, infile)))

    if os.path.exists(os.path.join(inpath, infile)):
        ML = np.load(os.path.join(inpath, infile))
    else:  # can't recover from this
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