#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run DAC simulations using various linearisation methods

@author: Arnfinn Eielsen, Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

%reload_ext autoreload
%autoreload 2

# Imports
import sys
import numpy as np
from numpy import matlib
import os
import scipy
from scipy import signal
from matplotlib import pyplot as plt

import math

import dither
from configurations import quantiser_configurations
from static_dac_model import generate_dac_output, quantise_signal, generate_codes
from figures_of_merit import FFT_SINAD, TS_SINAD

from lin_method_nsdcal import nsdcal
from lin_method_dem import dem

class lin_method:
    BASELINE = 1  # baseline
    PHYSCAL = 2  # physical level calibration
    DEM = 3  # dynamic element matching
    NSDCAL = 4  # noise shaping with digital calibration (INL model)
    SHPD = 5  # stochastic high-pass noise dither
    PHFD = 6  # periodic high-frequency dither
    MPC = 7  # model predictive control (with INL model)
    ILC = 8  # iterative learning control (with INL model, periodic signals)


def test_signal(SCALE, MAXAMP, FREQ, OFFSET, t):
    """
    Generate a test signal (carrier)

    Parameters
    ----------
    SCALE
        Percentage of maximum amplitude
    MAXAMP
        Maximum amplitude
    FREQ
        Signal frequency in hertz
    OFFSET
        Signal offset
    t
        Time vector

    Returns
    -------
    x
        Sinusoidal test signal

    Raises
    ------
    No error handling.

    Examples
    --------
    TODO: Make an example

    """
    return (SCALE/100)*MAXAMP*np.cos(2*np.pi*FREQ*t) + OFFSET

# %% Configuration

# Choose which linearization method you want to test
# RUN_LIN_METHOD = lin_method.BASELINE
# RUN_LIN_METHOD = lin_method.PHYSCAL
# RUN_LIN_METHOD = lin_method.PHFD
# RUN_LIN_METHOD = lin_method.SHPD
# RUN_LIN_METHOD = lin_method.NSDCAL
RUN_LIN_METHOD = lin_method.DEM

# Output low-pass filter configuration
Fc_lp = 10e3  # cut-off frequency in hertz
N_lf = 3  # filter order

# Sampling rate
Fs = 1e6  # sampling rate (over-sampling) in hertz
Ts = 1/Fs  # sampling time

# Carrier signal (to be recovered on the output)
Xcs_SCALE = 100  # %
Xcs_FREQ = 99  # Hz

# Set quantiser model
QConfig = 4
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

# %% Load measured or generated output levels
# TODO: This is a bit of a mess
match 2:
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
    case 2:  # load measured levels
        # load measured levels given linearisation method (measured for a given physical set-up)
        match RUN_LIN_METHOD:
            case lin_method.BASELINE | lin_method.DEM | lin_method.NSDCAL | lin_method.SHPD | lin_method.PHFD:
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

# %% Generate time vector

match 1:
    case 1:  # specify duration as number of samples and find number of periods
        Nts = 1e6  # no. of time samples
        Np = np.ceil(Xcs_FREQ*Ts*Nts).astype(int) # no. of periods for carrier
    case 2:  # specify duration as number of periods of carrier
        Np = 5  # no. of periods for carrier
        
Npt = 1  # no. of carrier periods to use to account for transients
Np = Np + Npt

t_end = Np/Xcs_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

# %% Generate carrier/test signal
SIGNAL_MAXAMP = Rng/2 - Qstep  # make headroom for noise dither (see below)
SIGNAL_OFFSET = -Qstep/2  # try to center given quantiser type
Xcs = test_signal(Xcs_SCALE, SIGNAL_MAXAMP, Xcs_FREQ, SIGNAL_OFFSET, t)

# %% Linearisation methods
match RUN_LIN_METHOD:
    case lin_method.BASELINE:  # baseline, only carrier
        Nch = 1  # number of channels to use (averaging to reduce noise floor)

        # Quantisation dither
        Dq = dither.gen_stochastic(t.size, Nch, Qstep, dither.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        X = Xcs + Dq  # quantiser input

        Q = quantise_signal(X, Qstep, Qtype)
        C = generate_codes(Q, Qstep, Qtype, Vmin)

    case lin_method.PHYSCAL:  # physical level calibration
        # Quantisation dither
        Dq = dither.gen_stochastic(t.size, 1, Qstep, dither.pdf.triangular_hp)

        X = Xcs + Dq  # quantiser input

        # TODO: figure out a better way to deal with this file dependency
        LUTcal = np.load('LUTcal.npy')  # load calibration look-up table

        q = quantise_signal(X, Qstep, Qtype)
        c_pri = generate_codes(q, Qstep, Qtype, Vmin)

        c_sec = LUTcal[c_pri.astype(int)]

        C = np.stack((c_pri[0, :], c_sec[0, :]))

        # Zero contribution from secondary in ideal case
        YQ = np.stack((YQ[0, :], np.zeros(YQ.shape[1])))

    case lin_method.DEM:  # dynamic element matching
        Nch = 1  # DEM effectively has 1 channel input

        # Quantisation dither
        Dq = dither.gen_stochastic(t.size, Nch, Qstep, dither.pdf.triangular_hp)
        Dq = Dq[0]  # convert to 1d

        X = Xcs + Dq  # input

        C = dem(X, Rng, Nb)
            
        # two identical, ideal channels
        YQ = np.stack((YQ[0, :], YQ[0, :]))

    case lin_method.NSDCAL:  # noise shaping with digital calibration
        Nch = 1  # only supports a single channel (at this point)

        # Re-quantisation dither
        DITHER_ON = 1
        Dq = dither.gen_stochastic(t.size, Nch, Qstep, dither.pdf.triangular_hp)
        Dq = DITHER_ON*Dq[0]  # convert to 1d, add/remove dither
        
        # The feedback generates an actuation signal that may cause the
        # quantiser to saturate if there is no "headroom"
        # Also need room for re-quantisation dither
        HEADROOM = 1  # %
        X = ((100-HEADROOM)/100)*Xcs  # input
        
        YQns = YQ[0]  # ideal ouput levels
        MLns = ML[0]  # measured ouput levels (convert from 2d to 1d)

        # introducing some "measurement error" in the levels
        MLns_err = np.random.uniform(-Qstep/2, Qstep/2, MLns.shape)
        MLns = MLns + MLns_err

        QMODEL = 2  # 1: no calibration, 2: use calibration
        C = nsdcal(X, Dq, YQns, MLns, Qstep, Vmin, Nb, QMODEL)
        
    case lin_method.SHPD:  # stochastic high-pass noise dither
        Nch = 2

        # Quantisation dither
        Dq = dither.gen_stochastic(t.size, Nch, Qstep, dither.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        # Large high-pass dither set-up
        Xscale = 80  # carrier to dither ratio (between 0% and 100%)

        Dmaxamp = Rng/2  # maximum dither amplitude (volt)
        Dscale = 60  # %
        Ds = dither.gen_stochastic(t.size, Nch, Dmaxamp/2, dither.pdf.uniform)

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
        C = generate_codes(Q, Qstep, Qtype, Vmin)

        # two identical, ideal channels
        YQ = np.stack((YQ[0, :], YQ[0, :]))

    case lin_method.PHFD:  # periodic high-frequency dither
        Nch = 2  # this method requires even no. of channels

        # Quantisation dither
        Dq = dither.gen_stochastic(t.size, Nch, Qstep, dither.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        # Scaling dither with respect to the carrier
        Xscale = 50  # carrier to dither ratio (between 0% and 100%)
        Dscale = 100 - Xscale  # dither to carrier ratio
        Dfreq = 49e3  # Hz
        Dadf = dither.adf.uniform  # amplitude distr. funct.
        # Generate periodic dither
        Dmaxamp = Rng/2  # maximum dither amplitude (volt)
        dp = Dmaxamp*dither.gen_periodic(t, Dfreq, Dadf)
        
        # Opposite polarity for HF dither for pri. and sec. channel
        Dp = np.stack((dp, -dp))

        X = (Xscale/100)*Xcs + (Dscale/100)*Dp + Dq

        Q = quantise_signal(X, Qstep, Qtype)
        C = generate_codes(Q, Qstep, Qtype, Vmin)

        # two identical, ideal channels
        YQ = np.stack((YQ[0, :], YQ[0, :]))
        
    case lin_method.MPC:  # model predictive control (with INL model)
        sys.exit("Not implemented yet - MPC")
    case lin_method.ILC:  # iterative learning control (with INL model, only periodic signals)
        sys.exit("Not implemented yet - ILC")

# %% DAC output(s)
YU = generate_dac_output(C, YQ)  # using ideal, uniform levels
YM = generate_dac_output(C, ML)  # using measured or randomised levels

# %% Summation stage
if RUN_LIN_METHOD == lin_method.DEM:
    K = 1
else:
    K = 1/Nch

yu = K*np.sum(YU, 0)
ym = K*np.sum(YM, 0)

plt.plot(t, yu)
plt.show()

# %% Filter the output using a reconstruction (output) filter
# filter coefficients
b, a = signal.butter(N_lf, 2*np.pi*Fc_lp, 'lowpass', analog=True)
Wlp = signal.lti(b, a)  # filter LTI system instance

yu = yu.reshape(-1, 1)  # ensure the vector is a column vector
# filter the ideal output (using zero-order hold interp.)
yu_avg_out = signal.lsim(Wlp, yu, t, X0=None, interp=False)

ym = ym.reshape(-1, 1)  # ensure the vector is a column vector
# filter the DAC model output (using zero-order hold interp.)
ym_avg_out = signal.lsim(Wlp, ym, t, X0=None, interp=False)

# extract the filtered data; lsim returns (T, y, x) tuple, want output y
yu_avg = yu_avg_out[1]
ym_avg = ym_avg_out[1]

# %% Evaluate performance
TRANSOFF = np.floor(Npt*Fs/Xcs_FREQ).astype(int)
match 1:
    case 1:  # use FFT based method to detemine SINAD
        RU = FFT_SINAD(yu_avg[TRANSOFF:-1], Fs, 'Uniform')
        RM = FFT_SINAD(ym_avg[TRANSOFF:-1], Fs, 'Non-linear')
    case 2:  # use time-series sine fitting based method to detemine SINAD
        RU = TS_SINAD(yu_avg[TRANSOFF:-1], t[TRANSOFF:-1])
        RM = TS_SINAD(ym_avg[TRANSOFF:-1], t[TRANSOFF:-1])

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
