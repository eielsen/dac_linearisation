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
import numpy.matlib
import os
import scipy
import dither
from matplotlib import pyplot as plt

from configurations import quantiser_configurations
from static_dac_model import generate_dac_output, quantise_signal, generate_codes
from figures_of_merit import FFT_SINAD, TS_SINAD


class lin_method:
    BASELINE = 1  # baseline
    PHYSCAL = 2  # physical level calibration
    DEM = 3  # dynamic element matching
    NSDCAL = 4  # noise shaping with digital calibration (INL model)
    SHPD = 5  # stochastic high-pass noise dither
    PHFD = 6  # periodic high-frequency dither
    MPC = 7  # model predictive control (with INL model)
    ILC = 8  # iterative learning control (with INL model, periodic signals)


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

# Choose which linearization method you want to test
# RUN_LIN_METHOD = lin_method.BASELINE
# RUN_LIN_METHOD = lin_method.PHYSCAL
RUN_LIN_METHOD = lin_method.PHFD

# Output filter configuration
FILT_FREQ = 20e3
FILT_ORDER = 3

# Sampling rate
Fs = 1e6  # sampling rate (over-sampling) in hertz
Ts = 1/Fs  # sampling time

# Test signal; carrier (to be recovered on the output)
CARRIER_SCALE = 100  # %
CARRIER_FREQ = 99  # Hz

# Set quantiser model
QConfig = 4
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

# Load measured or generated output levels
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
        # Linearisation methods
        match RUN_LIN_METHOD:
            case lin_method.BASELINE | lin_method.PHFD:
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
                infile = 'measurements_and_data/\
                    PHYSCAL_level_measurements_set_2.mat'
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

# Generate time vector
match 1:
    case 1:  # specify duration as number of samples and find number of periods
        TRANSOFF = int(1e3)  # no. of samples to use to account for transients
        Nts = 1e6 + TRANSOFF  # no. of time samples
        Np = CARRIER_FREQ*Ts*Nts  # no. of periods for carrier
    case 2:  # specify duration as number of periods of carrier
        Np = 5  # no. of periods for carrier

t_end = Np/CARRIER_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

# Generate carrier/test signal
SIGNAL_MAXAMP = Rng/2 - Qstep  # make headroom for noise dither (see below)
SIGNAL_OFFSET = -Qstep/2
# Carrier signal
Xcs = test_signal(CARRIER_SCALE, SIGNAL_MAXAMP, CARRIER_FREQ, SIGNAL_OFFSET, t)

# Linearisation methods
match RUN_LIN_METHOD:
    case lin_method.BASELINE:  # baseline, only carrier
        Nch = 1  # number of channels to use (averaging to reduce noise floor)

        # quantisation dither
        Dq = dither.gen_stochastic(t.size, Nch, Qstep,
                                   dither.pdf.triangular_hp)

        Xcs = numpy.matlib.repmat(Xcs, Nch, 1)

        X = Xcs + Dq  # quantiser input

        Q = quantise_signal(X, Qstep, Qtype)
        C = generate_codes(Q, Qstep, Qtype, Vmin)

    case lin_method.PHYSCAL:  # physical level calibration
        # quantisation dither
        Dq = dither.gen_stochastic(t.size, 1, Qstep, dither.pdf.triangular_hp)

        X = Xcs + Dq  # quantiser input

        # TODO: figure out a better way to deal with this file dependency
        LUTcal = np.load('LUTcal.npy')  # load calibation look-up table

        q = quantise_signal(X, Qstep, Qtype)
        c_pri = generate_codes(q, Qstep, Qtype, Vmin)

        c_sec = LUTcal[c_pri.astype(int)]

        C = np.stack((c_pri[0, :], c_sec[0, :]))

        # zero contribution from secondary in ideal case
        YQ = np.stack((YQ[0, :], np.zeros(YQ.shape[1])))

    case lin_method.DEM:  # dynamic element matching
        sys.exit("Not implemented yet - DEM")
    case lin_method.NSDCAL:  # noise shaping with digital calibration
        sys.exit("Not implemented yet - \
                 noise shaping with digital calibration")
    case lin_method.SHPD:  # stochastic high-pass noise dither
        sys.exit("Not implemented yet - stochastic high-pass noise dither")
    case lin_method.PHFD:  # periodic high-frequency dither
        # Scaling dither with respect to the carrier scaling
        CARRIER_RATIO = 0.6  # carrier scaling
        DITHER_FREQ = 49e3
        DITHER_TYPE = dither.adf.uniform

        DITHER_RATIO = 1 - CARRIER_RATIO
        
        # quantisation dither
        Dq = dither.gen_stochastic(t.size, 2, Qstep, dither.pdf.triangular_hp)

        # same carrier to both channels
        Xcs = numpy.matlib.repmat(Xcs, 2, 1)
        
        DAMP = Rng/2  # maximum dither amplitude (volt)
        dp = DAMP*dither.gen_periodic(t, DITHER_FREQ, DITHER_TYPE)
        
        # opposite polarity for HF dither
        Dp = np.stack((dp, -dp))

        X = CARRIER_RATIO*Xcs + DITHER_RATIO*Dp + Dq

        Q = quantise_signal(X, Qstep, Qtype)
        C = generate_codes(Q, Qstep, Qtype, Vmin)

        # two identical, ideal channels
        YQ = np.stack((YQ[0, :], YQ[0, :]))
        
    case lin_method.MPC:  # model predictive control (with INL model)
        sys.exit("Not implemented yet - MPC")
    case lin_method.ILC:  # iterative learning control (with INL model, only periodic signals)
        sys.exit("Not implemented yet - ILC")

# DAC output(s)
YU = generate_dac_output(C, YQ)  # using ideal, uniform levels
YM = generate_dac_output(C, ML)  # using measured or randomised levels

# Summation stage
yu = np.sum(YU, 0)/YU.shape[0]
ym = np.sum(YM, 0)/YM.shape[0]

# Filter the output using a reconstruction (output) filter
# filter coefficients
b, a = scipy.signal.butter(FILT_ORDER, 2*np.pi*FILT_FREQ, 'low', analog=True)
Wlp = scipy.signal.lti(b, a)  # filter LTI system instance

yu = yu.reshape(-1, 1)
# filter the ideal output using zero-order hold
yu_avg_out = scipy.signal.lsim(Wlp, yu, t, X0=None, interp=False)

ym = ym.reshape(-1, 1)
# filter the measured output
ym_avg_out = scipy.signal.lsim(Wlp, ym, t, X0=None, interp=False)

# Extract the filtered data; lsim returns (T, y, x) tuple, want output y
yu_avg = yu_avg_out[1]
ym_avg = ym_avg_out[1]

# %% Evaluate performance
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
