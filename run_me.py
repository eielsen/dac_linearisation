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

def test_signal(SCALE, MAXAMP, FREQ, OFFSET, t):
    """
    Generate a test signal (carrier)
    """
    return (SCALE/100)*MAXAMP*np.cos(2*np.pi*FREQ*t) + OFFSET

#%% Sampling configuration
Fs = 1e6  # set sampling rate (over-sampling)
Ts = 1/Fs  # sampling time

#%% Quantiser model
QuantizerConfig = 3
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QuantizerConfig)

# load measured or generated output levels
infile_1 = "generated_output_levels_{0}_bit_{1}.npy".format(Nb, 1)
infile_2 = "generated_output_levels_{0}_bit_{1}.npy".format(Nb, 2)

YQ_1 = np.zeros(YQ.size)
YQ_2 = np.zeros(YQ.size)

if exists(infile_1):
    YQ_1 = np.load(infile_1)
else:
    sys.exit("No level file found.")

if exists(infile_2):
    YQ_2 = np.load(infile_2)
else:
    sys.exit("No level file found.")

#%%  Test signal (to be recovered)

SCALE = 100 # Test Signal Scaling
FREQ = 99 # Test Signal Frequency

match 1:
    case 1:  # sepcify number of samples and find number of periods
        TRANSOFF = int(1e3)
        Nts = 1e6+TRANSOFF  # no. of time samples
        NP = FREQ*Ts*Nts  # no. of periods for carrier
    case 2:  # sepcify number of periods
        NP = 5  # no. of periods for carrier

t_end = NP/FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

MAXAMP = Rng/2 - Qstep # make headroom for noise dither (see below)
OFFSET = -Qstep/2

Xc = test_signal(SCALE, MAXAMP, FREQ, OFFSET, t) # Carrier signal

#%% Dither for quantisation error

Dq = np.random.uniform(-Qstep/2, Qstep/2, t.size)

#%% Linearisation methods
match 2:
    case 1:  # baseline
        X = Xc + Dq
        
        YU, YM = generate_dac_output(X, QuantizerConfig, YQ_1)
    case 2:  # physical level calibration
        sys.exit("Not implemented yet.")
    case 3:  # dynamic element matching
        sys.exit("Not implemented yet.")
    case 4:  # noise shaping with digital calibration (INL model)
        sys.exit("Not implemented yet.")
    case 5:  # stochastic high-pass noise dither
        sys.exit("Not implemented yet.")
    case 6:  # periodic high-frequency dither
        Dp = periodic_dither(t, 49e3, 1)
        Dp_SCALE = 0.25
        Xc_SCALE = 1 - Dp_SCALE
        X1 = Xc_SCALE*Xc + Dp_SCALE*Dp + Dq
        X2 = Xc_SCALE*Xc - Dp_SCALE*Dp + Dq
        
        YU1, YM1 = generate_dac_output(X1, QuantizerConfig, YQ_1)
        YU2, YM2 = generate_dac_output(X2, QuantizerConfig, YQ_2)
        
        YU = YU1 + YU2
        YM = YM1 + YM2
    case 7:  # model predictive control (with INL model)
        sys.exit("Not implemented yet.")
    case 8:  # iterative learning control (with INL model, only periodic signals)
        sys.exit("Not implemented yet.")

#%% Reconstruciton (output) filter
b, a = signal.butter(2, 2*np.pi*20e3, 'low', analog=True)
Wlp = signal.lti(b, a)
YUf = signal.lsim(Wlp, YU, t, X0=None, interp=False)
YMf = signal.lsim(Wlp, YM, t, X0=None, interp=False)

YUp = YUf[1]
YMp = YMf[1]

match 1:
    case 1:
        RU = FFT_SINAD(YUp[TRANSOFF:-1], Fs)
        RM = FFT_SINAD(YMp[TRANSOFF:-1], Fs)
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
