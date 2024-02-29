#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run DAC simulations using various linearisation methods

@author: Arnfinn Eielsen, Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

#%% Imports
import sys
import numpy as np
import os
import scipy

from matplotlib import pyplot as plt

from configurations import quantiser_configurations
from static_dac_model import generate_dac_output, quantise_signal, generate_codes
from welch_psd import welch_psd
from periodic_dither import periodic_dither
from figures_of_merit import FFT_SINAD, TS_SINAD

def test_signal(SCALE, A, FREQ, OFFSET, t):
    """
    Generate a test signal (carrier)

    Parameters
    ----------
    SCALE
        Percentage of maximum amplitude
    A
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
    return (SCALE/100)*A*np.cos(2*np.pi*FREQ*t) + OFFSET

CURRENT_PATH = os.getcwd()
print(CURRENT_PATH)

### DEFINITIONS ###
# LINEARIZATION_METHODS (LM)
class linearisation_method:
    BASE = 1     # BASELINE
    PHYSCAL = 2  # PHYSICAL LEVEL CALIBRATION
    DEM = 3      # DYNAMIC ELEMENT MATCHING
    DIGCAL = 4   # NOISE SHAPING WITH DIGITAL CALIBRATION (INL model)
    HPNOISE = 5  # STOCHASTIC HIGH-PASS NOISE DITHER
    HFDITHER = 6 # PERIODIC HIGH-FREQUENCY DITHER
    MPC = 7      # MODEL PREDICTIVE CONTROL (with INL model)
    ILC = 8      # ITERATIVE LEARNING CONTROL (with INL model, only periodic signals)

# DITHER TYPES (DT)
DT_UNIFORM_ADF_TRI_WAVE = 1  # UNIFORM ADF (TRIANGULAR WAVE)
DT_TRIANGULAR_ADF = 2  # TRIANGULAR ADF
DT_CAUCHY_ADF = 3  # CAUCHY_ADF (when tri_wave in ±pi/2 !)
DT_GAUSSIAN_ADF = 4  # GAUSSIAN_ADF

### DITHER ###
DITHER_FREQ = 49e3
DITHER_TYPE = DT_UNIFORM_ADF_TRI_WAVE

### SETUP - START ###
# Choose which linearization method you want to use
LINEARIZATION_METHOD = linearisation_method.PHYSCAL

## FILTER SETTINGS
FILTER_FREQ = 20e3
FILTER_ORDER = 2
# 'low', 'high, 'band', 'stop'
# 'lowpass', 'highpass', 'bandpass', 'bandstop'
FILTER_PASS_TYPE = 'low' 

## SAMPLING SETTINGS
SAMPLE_FREQ = 1e6  # Sampling rate (over-sampling)
Fs = SAMPLE_FREQ
Ts = 1/Fs  # Sampling time/interval

## TEST SIGNAL (to be recovered)
SIGNAL_SCALE = 100 # Test Signal Scaling [%]
CARRIER_FREQ = 99e0 # [Hz]

CARRIER_RATIO = 0.75 # Carrier scaling
DITHER_RATIO = 1 - CARRIER_RATIO # Scaling dither with respect to the carrier scaling

## Quantiser model
QConfig = 4
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

### SETUP - END ###

# load measured or generated output levels
match 2:
    case 1: # load some generated levels
        infile_1 = os.path.join(CURRENT_PATH, 'generated_output_levels', f'generated_output_levels_{Nb}_bit_{1}_QuantizerConfig_{QConfig}.npy')
        infile_2 = os.path.join(CURRENT_PATH, 'generated_output_levels', f'generated_output_levels_{Nb}_bit_{2}_QuantizerConfig_{QConfig}.npy')

        if os.path.exists(infile_1):
            ML_1 = np.load(infile_1)  # generated/"measured" levels for channel 1
        else:
            sys.exit("YQ_1 - No level file found.") # can't recover from this
        if os.path.exists(infile_2):
            ML_2 = np.load(infile_2)  # generated/"measured" levels for channel 2
        else:
            sys.exit("YQ_2 - No level file found.") # can't recover from this
    case 2: # load measured levels
        infile = 'measurements_and_data/PHYSCAL_level_measurements_set_2.mat'; fileset = 2
        if os.path.exists(infile):
            mat = scipy.io.loadmat(infile)
        else:
            sys.exit("No level measurements file found.") # can't recover from this

        ML_1 = mat['PRILVLS'][0]  # measured levels for channel 1
        ML_2 = mat['SECLVLS'][0]  # measured levels for channel 2

# TODO: Make variables and constants to describe and chose what to do here.
match 1:
    case 1:  # specify number of samples and find number of periods
        TRANSOFF = int(1e3)  # number of samples to shave off to remove transient effects
        Nts = 1e6 + TRANSOFF  # no. of time samples
        Np = CARRIER_FREQ*Ts*Nts  # no. of periods for carrier
    case 2:  # specify number of periods
        Np = 5  # no. of periods for carrier

t_end = Np/CARRIER_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

# GENERATE CARRIER/TEST SIGNAL
SIGNAL_MAXAMP = Rng/2 - Qstep # make headroom for noise dither (see below)
SIGNAL_OFFSET = -Qstep/2

# Xcs - Input carrier signal
Xcs = test_signal(SIGNAL_SCALE, SIGNAL_MAXAMP, CARRIER_FREQ, SIGNAL_OFFSET, t) # Carrier signal

# Dq - Dither for quantisation error
# TODO: Generate triangular PDF dither for linearising quantiser, not uniform
Dq = np.random.uniform(-Qstep/2, Qstep/2, t.size)

# LINEARIZATION METHODS
# TODO: Replace hard coded numbers
match LINEARIZATION_METHOD:
    case linearisation_method.BASE:  # LINEARIZATION_METHOD: None / BASELINE
        X = Xcs + Dq
        
        y_ideal = generate_dac_output(X, QConfig, YQ)
        y_nl = generate_dac_output(X, QConfig, ML_1)
    case linearisation_method.PHYSCAL:  # physical level calibration
        X = Xcs + Dq

        # load calibation look-up table
        LUTcal = np.load('LUTcal.npz')
        ML = ML_1  # use channel 1 as Main/primary (measured levels)
        CL = ML_2  # use channel 2 to calibrate/secondary (measured levels)
        
        y_ideal = generate_dac_output(X, QConfig, YQ) # ideal out

        y_nl_ch1 = generate_dac_output(X, QConfig, ML)

        q = quantise_signal(X, Qstep, Qtype)
        c = generate_codes(q, Qstep, Qtype, Vmin)
        c = c.reshape(1,-1)
        y_nl_ch2 = CL[LUTcal[c.astype(int)]]  # calibration levels

        y_nl = y_nl_ch1 + y_nl_ch2  # non-linear out
        #Y1(i) = Qstep*q; % ideal quantized output
        #Y2(i) = ML(c+1);
        #Y3(i) = ML(c+1) + CL(LUTcal(c+1));
    case linearisation_method.DEM:  # dynamic element matching (DEM)
        sys.exit("Not implemented yet - DEM")
    case linearisation_method.DIGCAL:  # noise shaping with digital calibration (INL model)
        sys.exit("Not implemented yet - noise shaping with digital calibration (INL model)")
    case linearisation_method.HPNOISE:  # stochastic high-pass noise dither
        sys.exit("Not implemented yet - stochastic high-pass noise dither")
    case linearisation_method.HFDITHER:  # periodic high-frequency dither
        Dp = periodic_dither(t, DITHER_FREQ, DITHER_TYPE)
        X1 = CARRIER_RATIO*Xcs + DITHER_RATIO*Dp + Dq
        X2 = CARRIER_RATIO*Xcs - DITHER_RATIO*Dp + Dq
        
        y_ideal_ch1 = generate_dac_output(X1, QConfig, YQ)
        y_ideal_ch2 = generate_dac_output(X2, QConfig, YQ)

        y_nl_ch1 = generate_dac_output(X1, QConfig, ML_1)
        y_nl_ch2 = generate_dac_output(X2, QConfig, ML_2)
        
        y_ideal = y_ideal_ch1 + y_ideal_ch2  # ideal out
        y_nl = y_nl_ch1 + y_nl_ch2  # non-linear out
    case linearisation_method.MPC: # MODEL PREDICTIVE CONTROL (with INL model)
        sys.exit("Not implemented yet - MPC")
    case linearisation_method.ILC:  # ITERATIVE LEARNING CONTROL (with INL model, only periodic signals)
        sys.exit("Not implemented yet - ILC")

# FILTERING
# Filter the output using a reconstruction (output) filter
b, a = scipy.signal.butter(FILTER_ORDER, 2*np.pi*FILTER_FREQ, FILTER_PASS_TYPE, analog=True)  # filter coefficients
Wlp = scipy.signal.lti(b, a)  # filter LTI system instance
output_ideal_filtered = scipy.signal.lsim(Wlp, y_ideal, t, X0=None, interp=False)  # filter the ideal output using zero-order hold
output_meas_filtered = scipy.signal.lsim(Wlp, y_nl, t, X0=None, interp=False)  # filter the measured output

# Extract the filtered data; lsim returns (T, y, x) tuple, want output y
YUp = output_ideal_filtered[1]
YMp = output_meas_filtered[1]

# TODO: Make variables and constants to describe and chose what to do here.
match 1:
    case 1:
        RU = FFT_SINAD(YUp[TRANSOFF:-1], Fs,'RU_run_me')
        RM = FFT_SINAD(YMp[TRANSOFF:-1], Fs,'RM_run_me')
    case 2:
        RU = TS_SINAD(YUp[TRANSOFF:-1], t[TRANSOFF:-1])
        RM = TS_SINAD(YMp[TRANSOFF:-1], t[TRANSOFF:-1])

ENOB_U = (RU - 1.76)/6.02
ENOB_M = (RM - 1.76)/6.02

#%% Print FOM
print("SINAD uniform: {}".format(RU))
print("ENOB uniform: {}".format(ENOB_U))

print("SINAD non-linear: {}".format(RM))
print("ENOB non-linear: {}".format(ENOB_M))

fig_wave, axs_wave = plt.subplots(6, 1, sharex=True)
axs_wave[0].plot(t, X1)
axs_wave[0].grid()
axs_wave[1].plot(t, X2)
axs_wave[1].grid()
axs_wave[2].plot(t, y_ideal)
axs_wave[2].grid()
axs_wave[3].plot(t, y_nl)
axs_wave[3].grid()
axs_wave[4].plot(t, Dq)
axs_wave[4].grid()
axs_wave[5].plot(t, Dp)
axs_wave[5].grid()

#fig_wave.savefig('run_me_dither_waveforms_5.pdf', bbox_inches='tight')
