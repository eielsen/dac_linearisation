#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run DAC simulations using various linearisation methods

@author: Arnfinn Eielsen, Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

%reload_ext autoreload
%autoreload 2

#  %%Imports
import sys
import numpy as np
from numpy import matlib
import os
import statistics
import scipy
from scipy import signal
from matplotlib import pyplot as plt
import tqdm
import math
import csv
import datetime

import dither_generation
from quantiser_configurations import quantiser_configurations, quantiser_word_size
from static_dac_model import generate_dac_output, quantise_signal, generate_codes, quantiser_type
from figures_of_merit import FFT_SINAD, TS_SINAD

from lin_method_nsdcal import nsdcal
from lin_method_dem import dem
from lin_method_ilc import get_control, learning_matrices
from lin_method_ilc_simple import ilc_simple
# from lin_method_mpc import MPC, dq, gen_ML, gen_C, gen_DO

from spice_utils import run_spice_sim, run_spice_sim_parallel, generate_spice_batch_file, read_spice_bin_file

class lin_method:
    BASELINE = 1  # baseline
    PHYSCAL = 2  # physical level calibration
    DEM = 3  # dynamic element matching
    NSDCAL = 4  # noise shaping with digital calibration (INL model)
    SHPD = 5  # stochastic high-pass noise dither
    PHFD = 6  # periodic high-frequency dither
    MPC = 7  # model predictive control (with INL model)
    ILC = 8  # iterative learning control (with INL model, periodic signals)
    ILC_SIMP = 9  # iterative learning control, basic implementation


class sinad_comp:
    FFT = 1  # FFT based
    CFIT = 2  # curve fit


class sim_config:
    def __init__(self, lin_method, dac_model, fs, fc, carrier_scale, carrier_freq):
        self.lin_method = lin_method
        self.dac_model = dac_model
        self.fs = fs
        self.fc = fc
        self.carrier_scale = carrier_scale
        self.carrier_freq = carrier_freq
    def __str__(self):
        s = ''
        match self.lin_method:
            case lin_method.BASELINE:
                s = s + 'baseline'
            case lin_method.PHYSCAL:
                s = s + 'physical level calibration'
            case lin_method.DEM:
                s = s + 'dynamic element matching'
            case lin_method.NSDCAL:
                s = s + 'digital calibration'
            case lin_method.SHPD:
                s = s + 'noise dither'
            case lin_method.PHFD:
                s = s + 'periodic dither'
            case lin_method.MPC:
                s = s + 'mpc'
            case lin_method.ILC:
                s = s + 'ilc'
            case lin_method.ILC_SIMP:
                s = s + 'ilc simple'

        if self.dac_model == 1:
            s = s + '\nstatic model'
        elif self.dac_model == 2:
            s = s + '\nspice model'

        s = s + '\nFs=' + str(self.fs)
        s = s + '\nFc=' + str(self.fc)
        s = s + '\nXs=' + str(self.carrier_scale)
        s = s + '\nFx=' + str(self.carrier_freq)

        return s + '\n'


def test_signal(SCALE, MAXAMP, FREQ, OFFSET, t):
    """
    Generate a test signal (carrier)

    Arguments
        SCALE - percentage of maximum amplitude
        MAXAMP - maximum amplitude
        FREQ - signal frequency in hertz
        OFFSET - signal offset
        t - time vector
    
    Returns
        x - sinusoidal test signal
    """
    return (SCALE/100)*MAXAMP*np.cos(2*np.pi*FREQ*t) + OFFSET


def get_output_levels(RUN_LIN_METHOD):
    """
    Load measured or generated output levels.
    """
    # TODO: This is a bit of a mess
    match 3:
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
        case 2:  # load measured levels
            # load measured levels given linearisation method (measured for a given physical set-up)
            match RUN_LIN_METHOD:
                case lin_method.BASELINE | lin_method.DEM | lin_method.NSDCAL | lin_method.SHPD | lin_method.PHFD | lin_method.ILC | lin_method.MPC:
                    infile = 'measurements_and_data/level_measurements.mat'
                    fileset = 2
                    if os.path.exists(infile):
                        mat_file = scipy.io.loadmat(infile)
                    else:
                        # can't recover from this
                        sys.exit("No level measurements file found.")
                    
                    # static DAC model output levels, one channel per row
                    ML = mat_file['ML']  # measured levels

                case lin_method.PHYSCAL:
                    infile = 'measurements_and_data/PHYSCAL_level_measurements_set_2.mat'
                    fileset = 2
                    if os.path.exists(infile):
                        mat_file = scipy.io.loadmat(infile)
                    else:
                        # can't recover from this
                        sys.exit("No level measurements file found.")

                    ML_1 = mat_file['PRILVLS'][0]  # measured levels for channel 1
                    ML_2 = mat_file['SECLVLS'][0]  # measured levels for channel 2

                    # static DAC model output levels, one channel per row
                    ML = np.stack((ML_1, ML_2))
        case 3:
            ML = np.load("SPICE_levels_16bit.npy") 
    return ML


# %% Configuration

# Choose which linearization method you want to test
# RUN_LIN_METHOD = lin_method.BASELINE
# RUN_LIN_METHOD = lin_method.PHYSCAL
RUN_LIN_METHOD = lin_method.PHFD
# RUN_LIN_METHOD = lin_method.SHPD
# RUN_LIN_METHOD = lin_method.NSDCAL
# RUN_LIN_METHOD = lin_method.DEM
# RUN_LIN_METHOD = lin_method.MPC
# RUN_LIN_METHOD = lin_method.ILC
# RUN_LIN_METHOD = lin_method.ILC_SIMP

# DAC_MODEL = 1  # use static non-linear quantiser model to simulate DAC
DAC_MODEL = 2  # use SPICE to simulate DAC output

# Chose how to compute SINAD
SINAD_COMP_SEL = sinad_comp.CFIT

# Output low-pass filter configuration
Fc_lp = 25e3  # cut-off frequency in hertz
N_lp = 3  # filter order

# Sampling rate
Fs = 2e6  # sampling rate (over-sampling) in hertz
Ts = 1/Fs  # sampling time

# Carrier signal (to be recovered on the output)
Xcs_SCALE = 100  # %
Xcs_FREQ = 999  # Hz

# Set quantiser model
QConfig = quantiser_word_size.w_16bit_SPICE
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

# %% Generate time vector
match 2:
    case 1:  # specify duration as number of samples and find number of periods
        Nts = 1e6  # no. of time samples
        Np = np.ceil(Xcs_FREQ*Ts*Nts).astype(int) # no. of periods for carrier
    case 2:  # specify duration as number of periods of carrier
        Np = 3  # no. of periods for carrier
        
Npt = 1  # no. of carrier periods to use to account for transients
Np = Np + 2*Npt

t_end = Np/Xcs_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

SC = sim_config(RUN_LIN_METHOD, DAC_MODEL, Fs, Fc_lp, Xcs_SCALE, Xcs_FREQ)

# %% Generate carrier/test signal
SIGNAL_MAXAMP = Rng/2 - Qstep  # make headroom for noise dither (see below)
SIGNAL_OFFSET = -Qstep/2  # try to center given quantiser type
Xcs = test_signal(Xcs_SCALE, SIGNAL_MAXAMP, Xcs_FREQ, SIGNAL_OFFSET, t)

# %% Linearisation methods
match RUN_LIN_METHOD:
    case lin_method.BASELINE:  # baseline, only carrier
        # Generate unmodified DAC output without any corrections.

        Nch = 2  # number of channels to use (averaging to reduce noise floor)

        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        X = Xcs + Dq  # quantiser input

        Q = quantise_signal(X, Qstep, Qtype)
        C = generate_codes(Q, Nb, Qtype)

        YQ = matlib.repmat(YQ, Nch, 1)

    case lin_method.PHYSCAL:  # physical level calibration
        # This method relies on a main/primary DAC operating normally
        # whilst a secondary DAC with a small gain tries to correct
        # for the level mismatches for each and every code.
        # Needs INL measurements and a calibration step.

        # Quantisation dither
        Nch_in = 1  # effectively 1 channel input (with 1 DAC pair)
        Dq = dither_generation.gen_stochastic(t.size, Nch_in, Qstep, dither_generation.pdf.triangular_hp)

        X = Xcs + Dq  # quantiser input

        # TODO: figure out a better way to deal with this file dependency
        LUTcal = np.load('LUTcal.npy')  # load calibration look-up table

        q = quantise_signal(X, Qstep, Qtype)
        c_pri = generate_codes(q, Nb, Qtype)

        c_sec = LUTcal[c_pri.astype(int)]

        C = np.stack((c_pri[0, :], c_sec[0, :]))

        # Zero contribution from secondary in ideal case
        Nch = 2  # number of physical channels
        YQ = np.stack((YQ[0, :], np.zeros(YQ.shape[1])))

    case lin_method.DEM:  # dynamic element matching
        # Here we assume standard off-the-shelf DACs which translates to
        # full segmentation, which then means we have 2 DACs to work with.

        Nch = 1  # DEM effectively has 1 channel input

        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)
        Dq = Dq[0]  # convert to 1d

        X = Xcs + Dq  # input

        C = dem(X, Rng, Nb)
            
        # two identical, ideal channels
        YQ = matlib.repmat(YQ, 2, 1)

    case lin_method.NSDCAL:  # noise shaping with digital calibration
        # Use a simple static model as an open-loop observer for a simple
        # noise-shaping feed-back filter. Model is essentially the INL.
        # Open-loop observer error feeds directly to output, so very
        # sensitive to model error.

        Nch = 1  # only supports a single channel (at this point)

        # Re-quantisation dither
        DITHER_ON = 1
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)
        Dq = DITHER_ON*Dq[0]  # convert to 1d, add/remove dither
        
        # The feedback generates an actuation signal that may cause the
        # quantiser to saturate if there is no "headroom"
        # Also need room for re-quantisation dither
        HEADROOM = 1  # %
        X = ((100-HEADROOM)/100)*Xcs  # input
        
        ML = get_output_levels(RUN_LIN_METHOD)  # TODO: Redundant re-calling below in this case

        YQns = YQ[0]  # ideal ouput levels
        MLns = ML[0]  # measured ouput levels (convert from 2d to 1d)

        # introducing some "measurement/model error" in the levels
        MLns_err = np.random.uniform(-Qstep/2, Qstep/2, MLns.shape)
        MLns = MLns + MLns_err

        QMODEL = 2  # 1: no calibration, 2: use calibration
        C = nsdcal(X, Dq, YQns, MLns, Qstep, Vmin, Nb, QMODEL)
        
    case lin_method.SHPD:  # stochastic high-pass noise dither
        # Adds a large(ish) high-pass filtered normally distributed noise dither.
        # The normal PDF has poor INL averaging properties.

        # most recent prototype has 4 channels, so limit to 4
        Nch = 2

        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        # Large high-pass dither set-up
        Xscale = 80  # carrier to dither ratio (between 0% and 100%)

        Dmaxamp = Rng/2  # maximum dither amplitude (volt)
        Dscale = 60  # %
        Ds = dither_generation.gen_stochastic(t.size, Nch, Dmaxamp/2, dither_generation.pdf.uniform)

        N_hf = 2
        Fc_hf = 150e3

        b, a = signal.butter(N_hf, Fc_hf/(Fs/2), btype='high', analog=False)#, fs=Fs)

        Dsf = signal.filtfilt(b, a, Ds, method="gust")
        
        X = (Xscale/100)*Xcs + (Dscale/100)*Dsf + Dq

        print(np.max(X))
        print(np.min(X))

        if np.max(X) > Rng/2:
            raise ValueError('Input out of bounds.') 
        if np.min(X) < -Rng/2:
            raise ValueError('Input out of bounds.')

        Q = quantise_signal(X, Qstep, Qtype)
        C = generate_codes(Q, Nb, Qtype)

        # two identical, ideal channels
        YQ = matlib.repmat(YQ, Nch, 1)

    case lin_method.PHFD:  # periodic high-frequency dither
        # Adds a large, periodic high-frequency dither. Uniform ADF has good
        # averaging effect on INL. May experiment with orther ADF for
        # smoothing results.

        # most recent prototype has 4 channels, so limit to 4
        Nch = 2  # this method requires even no. of channels

        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        # Scaling dither with respect to the carrier
        Xscale = 50  # carrier to dither ratio (between 0% and 100%)
        Dscale = 100 - Xscale  # dither to carrier ratio
        #Dfreq = 359e3  # Hz
        Dfreq = 299e3  # Hz
        Dadf = dither_generation.adf.uniform  # amplitude distr. funct. (ADF)
        # Generate periodic dither
        Dmaxamp = Rng/2  # maximum dither amplitude (volt)
        dp = Dmaxamp*dither_generation.gen_periodic(t, Dfreq, Dadf)
        
        # Opposite polarity for HF dither for pri. and sec. channel
        if Nch == 2:
            Dp = np.stack((dp, -dp))
        elif Nch == 4:
            Dp = np.stack((dp, -dp, dp, -dp))
        else:
            sys.exit("Invalid channel config. for periodic dithering.")

        X = (Xscale/100)*Xcs + (Dscale/100)*Dp + Dq

        Q = quantise_signal(X, Qstep, Qtype)
        C = generate_codes(Q, Nb, Qtype)

        # two/four identical, ideal channels
        YQ = matlib.repmat(YQ, Nch, 1)

    case lin_method.MPC:  # model predictive control (with INL model)
        # sys.exit("Not implemented yet - MPC")

        Nch = 1
        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)
         # Create levels dictionary
        Qlevels = YQ.squeeze()      # make array to list 

        # Unsigned integers representing the level codes
        level_codes = np.arange(0, 2**Nb,1) # Levels:  0, 1, 2, .... 2^(Nb)

        # Dictionary: Keys- Levels codes;  Values - DAC levels
        IL_dict = dict(zip(level_codes, Qlevels ))  # Each level represent the idea DAC levels

        # Measured level dictionary - Generated randomly for test 
        # ML_dict = generate_ML(Nb, Qstep, Q_levels)
        ML_values = get_output_levels(RUN_LIN_METHOD)
        ML_dict = dict(zip(level_codes, ML_values[0,:]))

        # % #  Reconstruction Filter
        b, a = signal.butter(2, Fc_lp/(Fs/2)) 
        # but = signal.dlti(b,a,dt = Ts)
        # ft, fi = signal.dimpulse(but, n = 2*len(Xcs))
        A, B, C, D = signal.tf2ss(b,a) # Transfer function to StateSpace

        N_PRED = 2     # prediction horizon

        # Initial Condition
        x0 = np.zeros(2).reshape(-1,1)

        # Loop counter
        len_MPC = len(Xcs) - N_PRED

        # Storage Container
        DAC_Q_MPC_Xcs = []

        # First we need to scale up the signal due to the constraint in the optimization problem 
        # Scale up the input signal 
        X = Xcs + Dq
        Xcs1 = X + Rng/2                  
        Xcs_new =  ((2**Nb) * (Xcs1 /Rng)) 

        for i in tqdm.tqdm(range(len_MPC)):
            Xcs_i = Xcs_new[i:i+N_PRED]

            # Get MPC control
            Q_MPC_Xcs_N = MPC(Nb, Xcs_i, N_PRED, x0, A, B, C, D)
            Q_MPC_Xcs_N =  Q_MPC_Xcs_N.astype(int)

            """
            CHOOSE- 
                1. IL_DICT (Ideal level dictionary) FOR IDEAL DAC
                2. ML_DICT (Measured level dictionary) FOR NONLINEAR DAC
            """

            # Generate DAC output
            DAC_Q_MPC_Xcs_N = gen_DO(Q_MPC_Xcs_N, ML_dict)  # Generate DAC output
        
            # Store optimal value
            DAC_Q_MPC_Xcs.append(DAC_Q_MPC_Xcs_N[0])
            
            # # State update for next horizon
            x0_new = A @ x0 + B * (DAC_Q_MPC_Xcs_N[0] - Xcs[i])  # State Prediction
            x0 = x0_new  # State Update

        # Convert list to array
        DAC_Q_MPC_Xcs = np.array(DAC_Q_MPC_Xcs)   

        yu = DAC_Q_MPC_Xcs
        ym = DAC_Q_MPC_Xcs

        tu = t[0:len(DAC_Q_MPC_Xcs)]
        tm = tu

        # rf = Xcs[0:len(DAC_Q_MPC_Xcs)]
        # headerlist = ["Time", "Ref", "MPC"]
        # datalist = zip(tu, rf, DAC_Q_MPC_Xcs )
        # with open("mpc_16bit_ml.csv", 'w') as f1:
        #     writer = csv.writer(f1, delimiter = '\t')
        #     writer.writerow(headerlist)
        #     writer.writerows(datalist)   

    case lin_method.ILC:  # iterative learning control (with INL model, only periodic signals)
    #     # sys.exit("Not implemented yet - ILC")
    #     Nch = 1
    #     # Quantisation dither
    #     Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)
    #      # Create levels dictionary
    #     Qlevels = YQ.squeeze()      # make array to list 

    #     # Unsigned integers representing the level codes
    #     level_codes = np.arange(0, 2**Nb,1) # Levels:  0, 1, 2, .... 2^(Nb)

    #     # Dictionary: Keys- Levels codes;  Values - DAC levels
    #     IL_dict = dict(zip(level_codes, Qlevels ))  # Each level represent the idea DAC levels

    #     # Measured level dictionary - Generated randomly for test 
    #     # ML_dict = generate_ML(Nb, Qstep, Q_levels)
    #     ML_values = get_output_levels(RUN_LIN_METHOD)
    #     ML_dict = dict(zip(level_codes, ML_values[0,:]))

    #     # % #  Reconstruction Filter
    #     b, a = signal.butter(3, Fc_lp/(Fs/2)) 
    #     but = signal.dlti(b,a,dt = Ts)
    #     ft, fi = signal.dimpulse(but, n = 2*len(Xcs))

         # % Period and padding length
        N = 2000
        #N_padding = 200
        #N_period = int(N + 2*N_padding)

        #QF_M, L_M, OUT_M = learning_matrices(len_X=N_period, im= fi)

    #     iter = 5
    #     X = Xcs + Dq
    #     # The difference in ideal and non-ideal DAC is the choice fo the IL_dict and ML_dict, dictionaries with the
    #     # ideal DAC levels and measured levels corresponding to the codes, respectively.

    #     # ILC uniform/ideal quantizer 
    #     ILC_U = get_control(N, N_padding, X.squeeze(), iter, QF_M, L_M, OUT_M, Qstep, Qlevels, Qtype, IL_dict)
        
    #     # ILC nonlinear quantizer with measured levels
    #     ILC_M = get_control(N, N_padding, X.squeeze(), iter, QF_M, L_M, OUT_M, Qstep, Qlevels, Qtype, ML_dict)

        # U_ILC stores values from all iterations, Extract only the last column for the output of last iteartion as follows    
       # ILC_yu = ILC_U[:,-1].reshape(1,-1)
      #  ILC_ym = ILC_U[:,-1].reshape(1,-1)
        # ILC_ym = ILC_M[:,-1].reshape(1,-1)
    case lin_method.ILC_SIMP:  # iterative learning control, basic implementation
        
        Nch = 1  # number of channels to use

        # Quantisation dither
        #Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        X = Xcs #+ Dq  # quantiser input
        x = X.squeeze()

        # Plant: Butterworth or Bessel reconstruction filter
        Wn = 2*np.pi*Fc_lp
        b, a = signal.butter(N_lp, Wn, 'lowpass', analog=True)
        Wlp = signal.lti(b, a)  # filter LTI system instance
        G = Wlp.to_discrete(dt=Ts, method='zoh')

        # Q filter
        M = 2001  # Support/filter length/no. of taps
        Q_Fc = 2.0e4  # Cut-off freq. (Hz)
        alpha = (np.sqrt(2)*np.pi*Q_Fc*M)/(Fs*np.sqrt(np.log(4)))
        sigma = (M - 1)/(2*alpha)
        Qfilt = signal.windows.gaussian(M, sigma)
        Qfilt = Qfilt/np.sum(Qfilt)

        # L filter tuning (for Fs = 1 MHz, Nb = 16 bit)
        kp = 0.3
        kd = 20
        Niter = 50

        c, y1 = ilc_simple(x, G, Qfilt, Qstep, Nb, Qtype, kp, kd, Niter)
        c_ = c.clip(0, 2**16-1)
        C = np.array([c_])
        print('** ILC simple end **')

# %% DAC output(s)

"""TODO:
    I did not wanted to change any thing in the code, just added  "lin.method.ILC" and this part for running it

    To run ILC uncomment this section
    1. select lin_method.ILC
    2. Uncomment the part below and jump to the filtering part 
"""
# %% DAC output(s)
if RUN_LIN_METHOD == lin_method.ILC:
    YU = ILC_yu     # ILC with ideal quantizer
    YM = ILC_ym     # ILC with nonlinear qunatizer;  

    # index for plotting; due to the padding and overlapping
    idx1 = int(N_padding/2)
    idx2 = int(idx1 + np.max(YU.shape))

    tu = t[idx1:idx2]
    tm = tu
    
    # plots
    fig, ax = plt.subplots()
    ax.plot(t,Xcs)
    ax.plot(t[idx1:idx2], YU.squeeze())

    yu = YU
    ym = YM
else:
    YU = generate_dac_output(C, YQ)  # using ideal, uniform levels
    tu = t

    match DAC_MODEL:
        case 1:  # use static non-linear quantiser model to simulate DAC
            ML = get_output_levels(RUN_LIN_METHOD)
            YM = generate_dac_output(C, ML)  # using measured or randomised levels
            tm = t
        case 2:  # use SPICE to simulate DAC output
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            
            outdir = 'spice_output/' + timestamp + '/'

            if os.path.exists(outdir):
                print('Putting output files in existing directory: ' + timestamp)
            else:
                os.mkdir(outdir)
            
            configf = 'sim_config.txt'
            with open(outdir + configf, 'w') as fout:
                fout.write(SC.__str__())

            spicef_list = []
            outputf_list = []
            
            for k in range(0,Nch):
                c = C[k,:]
                seed = k + 1
                spicef, outputf = generate_spice_batch_file(c, Nb, t, Ts, QConfig, seed, timestamp, k)
                spicef_list.append(spicef)
                outputf_list.append(outputf)
            
            if False:  # run SPICE
                spice_path = '/home/eielsen/ngspice_files/bin/ngspice'  # newest ver., fastest (local)

                if False:
                    for k in range(0,Nch):
                        run_spice_sim(spicef_list[k], outputf_list[k], outdir, spice_path)
                else:
                    run_spice_sim_parallel(spicef_list, outputf_list, outdir, spice_path)
                
                YM = np.zeros([Nch, t.size])
                tm = t
                for k in range(0,Nch):
                    t_spice, y_spice = read_spice_bin_file(outdir, outputf_list[k] + '.bin')
                    y_resamp = np.interp(t, t_spice, y_spice)  # re-sample
                    YM[k,:] = y_resamp
            else:  # leave files for later
                sys.exit('Files generated, exiting...')


# %% Summation stage
if RUN_LIN_METHOD == lin_method.DEM:
    K = np.ones((Nch,1))
if RUN_LIN_METHOD == lin_method.PHYSCAL:
    K = np.ones((Nch,1))
    K[1] = 1e-2
else:
    K = 1/Nch

yu = np.sum(K*YU, 0)
ym = np.sum(K*YM, 0)

# plt.plot(tu, yu)
# plt.show()

# %% Filter the output using a reconstruction (output) filter
# filter coefficients
Wn = 2*np.pi*Fc_lp
b, a = signal.butter(N_lp, Wn, 'lowpass', analog=True)
Wlp = signal.lti(b, a)  # filter LTI system instance

yu = yu.reshape(-1, 1)  # ensure the vector is a column vector
# filter the ideal output (using zero-order hold interp.)
yu_avg_out = signal.lsim(Wlp, yu, tu, X0=None, interp=False)

ym = ym.reshape(-1, 1)  # ensure the vector is a column vector
# filter the DAC model output (using zero-order hold interp.)
ym_avg_out = signal.lsim(Wlp, ym, tm, X0=None, interp=False)

# extract the filtered data; lsim returns (T, y, x) tuple, want output y
yu_avg = yu_avg_out[1]
ym_avg = ym_avg_out[1]

# %% Evaluate performance
TRANSOFF = np.floor(Npt*Fs/Xcs_FREQ).astype(int)  # remove transient effects from output
match SINAD_COMP_SEL:
    case sinad_comp.FFT:  # use FFT based method to detemine SINAD
        RU = FFT_SINAD(yu_avg[TRANSOFF:-TRANSOFF], Fs, 'Uniform')
        RM = FFT_SINAD(ym_avg[TRANSOFF:-TRANSOFF], Fs, 'Non-linear')
    case sinad_comp.CFIT:  # use time-series sine fitting based method to detemine SINAD
        RU = TS_SINAD(yu_avg[TRANSOFF:-TRANSOFF], t[TRANSOFF:-TRANSOFF])
        RM = TS_SINAD(ym_avg[TRANSOFF:-TRANSOFF], t[TRANSOFF:-TRANSOFF])

ENOB_U = (RU - 1.76)/6.02
ENOB_M = (RM - 1.76)/6.02

# %% Print FOM
print("SINAD uniform: {}".format(RU))
print("ENOB uniform: {}".format(ENOB_U))

print("SINAD non-linear: {}".format(RM))
print("ENOB non-linear: {}".format(ENOB_M))

# %%
# fig_wave, axs_wave = plt.subplots(6, 1, sharex=True)
# axs_wave[0].plot(t, X1)
# axs_wave[0].grid()
# axs_wave[1].plot(t, X2)
# axs_wave[1].grid()
# axs_wave[2].plot(t, y_ideal)
# axs_wave[2].grid()
# axs_wave[3].plot(t, y_nl)
# axs_wave[3].grid()
# axs_wave[4].plot(t, Dq)
# axs_wave[4].grid()
# axs_wave[5].plot(t, Dp)
# axs_wave[5].grid()

# fig_wave.savefig('run_me_dither_waveforms_5.pdf', bbox_inches='tight')
