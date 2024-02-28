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

from os.path import exists
from scipy import signal
from matplotlib import pyplot as plt

from configurations import quantiser_configurations
from static_dac_model import generate_dac_output
from welch_psd import welch_psd
from periodic_dither import periodic_dither
from figures_of_merit import FFT_SINAD, TS_SINAD

def test_signal(SCALE, A, FREQ, OFFSET, t):
    """
    Generate a test signal (carrier)
    """
    return (SCALE/100)*A*np.cos(2*np.pi*FREQ*t) + OFFSET

CURRENT_PATH = ('\\').join(__file__.split('\\')[0:-1]) + '\\'
print(CURRENT_PATH)

### DEFINITIONS ###
# LINEARIZATION_METHODS (LM)
# TODO: Write names for the remaining methods (2, 4, 5)
LM_NONE = 1     # BASELINE
LM_NONE = 2     # PHYSICAL LEVEL CALIBRATION
LM_DEM = 3      # DYNAMIC ELEMENT MATCHING
LM_NONE = 4     # NOISE SHAPING WITH DIGITAL CALIBRATION (INL model)
LM_NONE = 5     # STOCHASTIC HIGH-PASS NOISE DITHER
LM_DITHER_PERIODIC_HIGH_FREQ = 6 # PERIODIC HIGH-FREQUENCY DITHER
LM_MPC = 7      # MODEL PREDICTIVE CONTROL (with INL model)
LM_ILC = 8      # ITERATIVE LEARNING CONTROL (with INL model, only periodic signals)

# DITHER TYPES (DT)
DT_UNIFORM_ADF_TRI_WAVE = 1    # UNIFORM ADF (TRIANGULAR WAVE)
DT_TRIANGULAR_ADF = 2          # TRIANGULAR ADF
DT_CAUCHY_ADF = 3              # CAUCHY_ADF (when tri_wave in ±pi/2 !)
DT_GAUSSIAN_ADF = 4            # GAUSSIAN_ADF

### DITHER ###
DITHER_FREQ = 49e3
DITHER_TYPE = DT_UNIFORM_ADF_TRI_WAVE

### SETUP - START ###
# Choose which linearization method you want to use
LINEARIZATION_METHOD = LM_DITHER_PERIODIC_HIGH_FREQ

## FILTER SETTINGS
FILTER_FREQ = 20e3
FILTER_ORDER = 2
# 'low', 'high, 'band', 'stop'
# 'lowpass', 'highpass', 'bandpass', 'bandstop'
FILTER_PASS_TYPE = 'low' 

## SAMPLING SETTINGS
SAMPLE_FREQ = 1e6   # Sampling rate (over-sampling)
Fs = SAMPLE_FREQ
Ts = 1/Fs           # Sampling time/interval

## TEST SIGNAL (to be recovered)
SIGNAL_SCALE = 100 # Test Signal Scaling [%]
CARRIER_FREQ = 99e0 # [Hz]

SIGNAL_CARRIER_RATIO = 0.75 # Carrier scaling
SIGNAL_DITHER_RATIO = 1 - SIGNAL_CARRIER_RATIO # Scaling dither with respect to the carrier scaling

## Quantiser model
QuantizerConfig = 5
Nb, Mq, Vmin, Vmax, Rng, LSb, YQ, Qtype = quantiser_configurations(QuantizerConfig)

### SETUP - END ###

# load measured or generated output levels
infile_1 = CURRENT_PATH + f'generated_output_levels_{Nb}_bit_{1}_QuantizerConfig_{QuantizerConfig}.npy'
infile_2 = CURRENT_PATH + f'generated_output_levels_{Nb}_bit_{2}_QuantizerConfig_{QuantizerConfig}.npy'

YQ_1 = np.zeros(YQ.size)
YQ_2 = np.zeros(YQ.size)

if exists(infile_1): YQ_1 = np.load(infile_1)
else: print("YQ_1 - No level file found.") # sys.exit
if exists(infile_2): YQ_2 = np.load(infile_2)
else: print("YQ_2 - No level file found.") # sys.exit

# TODO: Make variables and constants to describe and chose what to do here.
match 1:
    case 1:  # sepcify number of samples and find number of periods
        TRANSOFF = int(1e3)
        Nts = 1e6+TRANSOFF  # no. of time samples
        Ncyc = CARRIER_FREQ*Ts*Nts  # no. of periods for carrier
    case 2:  # sepcify number of periods
        Ncyc = 5  # no. of periods for carrier

t_end = Ncyc/CARRIER_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

# GENERATE CARRIER/TEST SIGNAL
SIGNAL_MAXAMP = Rng/2 - LSb # make headroom for noise dither (see below)
SIGNAL_OFFSET = -LSb/2

# Xsc - Input signal carrier
Xcs = test_signal(SIGNAL_SCALE, SIGNAL_MAXAMP, CARRIER_FREQ, SIGNAL_OFFSET, t) # Carrier signal

# Dq - Dither for quantisation error
Dq = np.random.uniform(-LSb/2, LSb/2, t.size)

# LINEARIZATION METHODS
# TODO: Replace hard coded numbers in the if-elif statements.
if LINEARIZATION_METHOD == LM_NONE:  # LINEARIZATION_METHOD: None / BASELINE
    X = Ysc + Dq
    
    YU, YM = generate_dac_output(X, QuantizerConfig, YQ_1)
elif LINEARIZATION_METHOD == 2:  # physical level calibration
    sys.exit("Not implemented yet - physical level calibration")
elif LINEARIZATION_METHOD == LM_DEM:  # DYNAMIC ELEMENT MATCHING
    sys.exit("Not implemented yet - DEM")
elif LINEARIZATION_METHOD == 4:  # noise shaping with digital calibration (INL model)
    sys.exit("Not implemented yet - noise shaping with digital calibration (INL model)")
elif LINEARIZATION_METHOD == 5:  # stochastic high-pass noise dither
    sys.exit("Not implemented yet - stochastic high-pass noise dither")
elif LINEARIZATION_METHOD == LM_DITHER_PERIODIC_HIGH_FREQ:  # periodic high-frequency dither
    Dp = periodic_dither(t, DITHER_FREQ, DITHER_TYPE)
    X1 = SIGNAL_CARRIER_RATIO*Ysc + SIGNAL_DITHER_RATIO*Dp + Dq
    X2 = SIGNAL_CARRIER_RATIO*Ysc - SIGNAL_DITHER_RATIO*Dp + Dq
    
    output_ideal_ch1, output_meas_ch1 = generate_dac_output(X1, QuantizerConfig, YQ_1)
    output_ideal_ch2, output_meas_ch2 = generate_dac_output(X2, QuantizerConfig, YQ_2)
    
    output_ideal = output_ideal_ch1 + output_ideal_ch2
    output_meas = output_meas_ch1 + output_meas_ch2
elif LINEARIZATION_METHOD == LM_MPC: # MODEL PREDICTIVE CONTROL (with INL model)
    sys.exit("Not implemented yet - MPC")
elif LINEARIZATION_METHOD == LM_ILC:  # ITERATIVE LEARNING CONTROL (with INL model, only periodic signals)
    sys.exit("Not implemented yet - ILC")

# FILTERING
# Filter the output using a reconstruction (output) filter
b, a = signal.butter(FILTER_ORDER, 2*np.pi*FILTER_FREQ, FILTER_PASS_TYPE, analog=True) # Filter coefficients
Wlp = signal.lti(b, a) # Filter system instance
output_ideal_filtered = signal.lsim(Wlp, output_ideal, t, X0=None, interp=False) # Filter the ideal output
output_meas_filtered = signal.lsim(Wlp, output_meas, t, X0=None, interp=False) # Filter the measured output

# Extract the filtered data (?)
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
axs_wave[2].plot(t, YU)
axs_wave[2].grid()
axs_wave[3].plot(t, YM)
axs_wave[3].grid()
axs_wave[4].plot(t, Dq)
axs_wave[4].grid()
axs_wave[5].plot(t, Dp)
axs_wave[5].grid()

fig_wave.savefig('run_me_dither_waveforms_5.pdf', bbox_inches='tight')