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
import statistics
import scipy
from scipy import signal
from matplotlib import pyplot as plt
import tqdm
import math
import csv
import datetime
import pickle
from prefixed import Float
from tabulate import tabulate

import dither_generation
from dual_dither import dual_dither, hist_and_psd
from quantiser_configurations import quantiser_configurations, get_measured_levels, qws
from static_dac_model import generate_dac_output, quantise_signal, generate_codes, quantiser_type
from figures_of_merit import FFT_SINAD, TS_SINAD
from balreal import balreal_ct

from lin_method_nsdcal import nsdcal
from lin_method_dem import dem
# from lin_method_ilc import get_control, learning_matrices
# from lin_method_ilc_simple import ilc_simple
from lin_method_mpc import MPC
from lin_method_ILC_DSM import learningMatrices, get_ILC_control

from lin_method_util import lm, dm

from inl_processing import get_physcal_gain

from spice_utils import run_spice_sim, run_spice_sim_parallel, generate_spice_batch_file, read_spice_bin_file, sim_config, process_sim_output, sinad_comp


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


# Configuration

##### METHOD CHOICE - Choose which linearization method you want to test
# RUN_LM = lm.BASELINE
# RUN_LM = lm.PHYSCAL
# RUN_LM = lm.PHFD
# RUN_LM = lm.SHPD
# RUN_LM = lm.NSDCAL
# RUN_LM = lm.DEM
RUN_LM = lm.MPC
# RUN_LM = lm.ILC
# RUN_LM = lm.ILC_SIMP

lin = lm(RUN_LM)

##### MODEL CHOICE
dac = dm(dm.STATIC)  # use static non-linear quantiser model to simulate DAC
# dac = dm(dm.SPICE)  # use SPICE to simulate DAC output

# Chose how to compute SINAD
SINAD_COMP_SEL = sinad_comp.CFIT

# Output low-pass filter configuration
Fc_lp = 100e3  # cut-off frequency in hertz
N_lp = 3  # filter order

# Sampling rate
Fs = 1e6  # sampling rate (over-sampling) in hertz
# Fs = 25e6  # sampling rate (over-sampling) in hertz
# Fs = 250e6  # sampling rate (over-sampling) in hertz
# Fs = 130940928  # sampling rate (over-sampling) in hertz
# Fs = 261881856
Ts = 1/Fs  # sampling time

# Punkter: 1048576

# Carrier signal (to be recovered on the output)
Xcs_SCALE = 100  # %
Xcs_FREQ = 999  # Hz

##### Set quantiser model
#QConfig = qws.w_16bit_ARTI
#QConfig = qws.w_6bit_ARTI
QConfig = qws.w_6bit_2ch_SPICE
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

# Generate time vector
match 2:
    case 1:  # specify duration as number of samples and find number of periods
        Nts = 1e6  # no. of time samples
        Np = np.ceil(Xcs_FREQ*Ts*Nts).astype(int) # no. of periods for carrier
    case 2:  # specify duration as number of periods of carrier
        Np = 3  # no. of periods for carrier
        
Npt = 1/2  # no. of carrier periods to use to account for transients
Np = Np + 2*Npt

t_end = Np/Xcs_FREQ  # time vector duration
t = np.arange(0, t_end, Ts)  # time vector

SC = sim_config(lin, dac, Fs, t, Fc_lp, N_lp, Xcs_SCALE, Xcs_FREQ)

# Generate carrier/test signal
SIGNAL_MAXAMP = Rng/2 - Qstep  # make headroom for noise dither (see below)
SIGNAL_OFFSET = -Qstep/2  # try to center given quantiser type
Xcs = test_signal(Xcs_SCALE, SIGNAL_MAXAMP, Xcs_FREQ, SIGNAL_OFFSET, t)

# Linearisation methods
match SC.lin.method:
    case lm.BASELINE:  # baseline, only carrier
        # Generate unmodified DAC output without any corrections.

        if QConfig == qws.w_6bit_2ch_SPICE:
            Nch = 2  # number of channels to use (averaging to reduce noise floor)
        else:
            Nch = 1
        
        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        X = Xcs + Dq  # quantiser input

        Q = quantise_signal(X, Qstep, Qtype)
        C = generate_codes(Q, Nb, Qtype)

    case lm.PHYSCAL:  # physical level calibration
        # This method relies on a main/primary DAC operating normally
        # whilst a secondary DAC with a small gain tries to correct
        # for the level mismatches for each and every code.
        # Needs INL measurements and a calibration step.

        # Quantisation dither
        Nch_in = 1  # effectively 1 channel input (with 1 DAC pair)
        Dq = dither_generation.gen_stochastic(t.size, Nch_in, Qstep, dither_generation.pdf.triangular_hp)

        X = Xcs + Dq  # quantiser input
        
        lutfile = os.path.join('generated_physcal_luts', 'LUTcal_' + str(QConfig) + '.npy')
        LUTcal = np.load(lutfile)  # load calibration look-up table
        
        q = quantise_signal(X, Qstep, Qtype)
        c_pri = generate_codes(q, Nb, Qtype)

        c_sec = LUTcal[c_pri.astype(int)]

        C = np.stack((c_pri[0, :], c_sec[0, :]))

        # Zero contribution from secondary in ideal case
        Nch = 2  # number of physical channels
        YQ = np.stack((YQ[0, :], np.zeros(YQ.shape[1])))

    case lm.DEM:  # dynamic element matching
        # Here we assume standard off-the-shelf DACs which translates to
        # full segmentation, which then means we have 2 DACs to work with.

        Nch = 1  # DEM effectively has 1 channel input

        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)
        Dq = Dq[0]  # convert to 1d

        X = Xcs + Dq  # input

        C = dem(X, Rng, Nb)
            
        # two identical, ideal channels
        # Nch = 2  # number of physical channels
        # YQ = matlib.repmat(YQ, 2, 1)

    case lm.NSDCAL:  # noise shaping with digital calibration
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
        #HEADROOM = 1  # 16 bit DAC
        HEADROOM = 17.5  # 6 bit DAC
        X = ((100-HEADROOM)/100)*Xcs  # input
        
        ML = get_measured_levels(QConfig, SC.lin.method) # get_measured_levels(lm.NSDCAL)  # TODO: Redundant re-calling below in this case

        YQns = YQ[0]  # ideal ouput levels
        MLns = ML[0]  # measured ouput levels (convert from 2d to 1d)

        # introducing some "measurement/model error" in the levels
        MLns_err = np.random.uniform(-Qstep, Qstep, MLns.shape)  # 16 bit DAC
        MLns_err = np.random.uniform(-Qstep/1024, Qstep/1024, MLns.shape)  # 6 bit DAC
        MLns = MLns + MLns_err

        QMODEL = 2  # 1: no calibration, 2: use calibration
        C = nsdcal(X, Dq, YQns, MLns, Qstep, Vmin, Nb, QMODEL)

        if QConfig == qws.w_6bit_2ch_SPICE:
            #Nch = 2  # channels in file
            C = np.stack((C[0, :], np.zeros(C.shape[1])))
        
    case lm.SHPD:  # stochastic high-pass noise dither
        # Adds a large(ish) high-pass filtered normally distributed noise dither.
        # The normal PDF has poor INL averaging properties.

        # most recent prototype has 4 channels, so limit to 4
        Nch = 2

        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)

        # Repeat carrier on all channels
        Xcs = matlib.repmat(Xcs, Nch, 1)

        # Large high-pass dither set-up
        Xscale = 25  # carrier to dither ratio (between 0% and 100%)

        match 2:
            case 1:
                Dmaxamp = Rng/2  # maximum dither amplitude (volt)
                Dscale = 100  # %
                Ds = dither_generation.gen_stochastic(t.size, Nch, Dmaxamp, dither_generation.pdf.uniform)
                Dsf = Ds
            case 2:
                #Ds = dither_generation.gen_stochastic(t.size, Nch, 1, dither_generation.pdf.uniform)
                Ds = np.random.normal(0, 1.0, [Nch, t.size])  # normally distr. noise

                N_hf = 1
                Fc_hf = 150e3

                b, a = signal.butter(N_hf, Fc_hf/(Fs/2), btype='high', analog=False)#, fs=Fs)

                Dsf = signal.filtfilt(b, a, Ds, method="gust")
                Dsf[0,:] = 2.*(Dsf[0,:] - np.min(Dsf[0,:]))/np.ptp(Dsf[0,:]) - 1
                Dsf[1,:] = 2.*(Dsf[1,:] - np.min(Dsf[1,:]))/np.ptp(Dsf[1,:]) - 1
                
                Dmaxamp = Rng/2  # maximum dither amplitude (volt)
                Dscale = 75  # %
                Dsf = Dmaxamp*Dsf
            case 3:
                ds = np.random.normal(0, 1.0, [1, t.size])  # normally distr. noise
                
                N_hf = 1
                Fc_hf = 150e3

                b, a = signal.butter(N_hf, Fc_hf/(Fs/2), btype='high', analog=False)#, fs=Fs)

                dsf = signal.filtfilt(b, a, ds, method="gust")
                dsf = dsf.squeeze()
                dsf = 2.*(dsf - np.min(dsf))/np.ptp(dsf) - 1
                
                # Opposite polarity for HF dither for pri. and sec. channel
                if Nch == 2:
                    Dsf = np.stack((dsf, -dsf))
                elif Nch == 4:
                    Dsf = np.stack((dsf, -dsf, dsf, -dsf))
                else:
                    sys.exit("Invalid channel config. for stoch. dithering.")

                Dmaxamp = Rng/2  # maximum dither amplitude (volt)
                Dscale = 50  # %
                Dsf = Dmaxamp*Dsf
            case 4:
                Ds = np.random.normal(0, 1.0, [Nch, t.size])  # normally distr. noise

                N_hf = 1
                Fc_hf = 1700e3

                b, a = signal.butter(N_hf, Fc_hf/(Fs/2), btype='high', analog=False)#, fs=Fs)

                Dsf = signal.filtfilt(b, a, Ds, method="gust")
                Dsf[0,:] = 2.*(Dsf[0,:] - np.min(Dsf[0,:]))/np.ptp(Dsf[0,:]) - 1
                Dsf[1,:] = 2.*(Dsf[1,:] - np.min(Dsf[1,:]))/np.ptp(Dsf[1,:]) - 1
                
                Dmaxamp = Rng/2  # maximum dither amplitude (volt)
                Dscale = 50  # %
                Dsf = Dmaxamp*Dsf

            case 5:
                Dsf = np.zeros((Nch, t.size))
                Dmaxamp = Rng/2  # maximum dither amplitude (volt)
                Dscale = 100 - Xscale  # dither to carrier ratio
                Dsf[0,:] = 0.99*Dmaxamp*dual_dither(N=t.size)
                Dsf[1,:] = 0.99*Dmaxamp*dual_dither(N=t.size)
        
        hist_and_psd(Dsf[0,:].squeeze())

        # for k in range(0,Nch):
        #     dsf = Dsf[k,:]
        #     dsf = 2.*(dsf - np.min(dsf))/np.ptp(dsf) - 1
        #     Dsf[k,:] = dsf

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

    case lm.PHFD:  # periodic high-frequency dither
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
        
        #Dfreq = 1.592e6 # Fs10MHz - 16bit
        #Dfreq = 2.99e6 # Fs25Mhz - 16bit
        #Dfreq = 2.98e6 # Fs250Mhz - 6 bit
        #Dfreq = 3.11e6 # Fs130Mhz - 6 bit
        Dfreq = 2.99e6 # 
        Dadf = dither_generation.adf.uniform  # amplitude distr. funct. (ADF)
        # Generate periodic dither
        Dmaxamp = Rng/2  # maximum dither amplitude (volt)
        dp = 0.99*Dmaxamp*dither_generation.gen_periodic(t, Dfreq, Dadf)
        
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

    case lm.MPC:  # model predictive control (with INL model)
        Nch = 1
        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)
         # Create levels dictionary
        Qlevels = YQ.squeeze()      # make array to list 

        # Unsigned integers representing the level codes
        level_codes = np.arange(0, 2**Nb,1) # Levels:  0, 1, 2, .... 2^(Nb)

        ML = get_measured_levels(QConfig, SC.lin.method) # get_measured_levels(lm.MPC)
        MLns = ML[0]

        # Adding "measurement error"
        MLns_err = np.random.uniform(-Qstep, Qstep, MLns.shape)
        MLns = MLns + MLns_err

        if QConfig == 6:
            MLns = np.flip(MLns)

        # Reconstruction filter
        match 2:
            case 1:
                Wn = Fc_lp/(Fs/2)
                b1, a1 = signal.butter(2, Wn)
                A1, B1, C1, D1 = signal.tf2ss(b1,a1) # Transfer function to StateSpace
            case 2:  # bilinear transf., seems to work ok, not a perfect match to physics
                Wn = Fc_lp/(Fs/2)
                b1, a1 = signal.butter(N_lp, Wn)
                A1, B1, C1, D1 = signal.tf2ss(b1,a1) # Transfer function to StateSpace
            case 3:  # zoh interp. matches physics, SciPi impl. causes numerical problems??
                Wn = 2*np.pi*Fc_lp
                b1, a1 = signal.butter(N_lp, Wn, 'lowpass', analog=True)
                Wlp = signal.TransferFunction(b1, a1)  # filter LTI system instance
                Wlp_ss = Wlp.to_ss()  # controllable canonical form
                Ac = Wlp_ss.A
                Bc = Wlp_ss.B
                Cc = Wlp_ss.C
                Dc = Wlp_ss.D
                A_, B_, C_, D_ = balreal_ct(Ac, Bc, Cc, Dc)
                Wlp_ss_d = signal.cont2discrete((A_, B_, C_, D_), dt=1e-6, method='zoh')
                A1, B1, C1, D1, dt = Wlp_ss_d
                # A1 = Wlp_ss_d.A
                # B1 = Wlp_ss_d.B
                # C1 = Wlp_ss_d.C
                # D1 = Wlp_ss_d.D

        N_PRED = 2  # prediction horizon

        # Initial Condition
        x0 = np.ones(2).reshape(-1,1)

        # Add dither
        X = Xcs + Dq
        X = X.squeeze()

        C = MPC(Nb, N_PRED, X, MLns, Qstep, A1, B1, C1, D1, x0)

        t = t[0:C.size]

        if QConfig == qws.w_6bit_2ch_SPICE:
            #Nch = 2  # channels in file
            C = np.stack((C[0, :], np.zeros(C.shape[1])))

    case lm.ILC:  # iterative learning control (with INL model, only periodic signals)

        Nch = 1

        # The feedback generates an actuation signal that may cause the
        # quantiser to saturate if there is no "headroom"
        # Also need room for re-quantisation dither
        HEADROOM = 1  # %
        Xcs = ((100-HEADROOM)/100)*Xcs  # input
        
        # Quantisation dither
        Dq = dither_generation.gen_stochastic(t.size, Nch, Qstep, dither_generation.pdf.triangular_hp)
        # Ideal Levels
        ILns = YQ.squeeze()
        # % Measured Levels
        ML = get_measured_levels(QConfig, SC.lin.method) # get_measured_levels(lm.ILC)
        MLns = ML[0] # one channel only
        
        # Adding "measurement error"
        MLns_err = np.random.uniform(-Qstep, Qstep, MLns.shape)
        MLns = MLns + MLns_err
    
        # Reconstruction filter
        match 2:
            case 1:
                Wn = Fc_lp/(Fs/2)
                b1, a1 = signal.butter(2, Wn)
                l_dlti = signal.dlti(b1, a1, dt=Ts)
            case 2:  # bilinear transf., seems to work ok, not a perfect match to physics
                Wn = Fc_lp/(Fs/2)
                #b1, a1 = signal.butter(N_lp, Wn, btype='low', analog=False, output='ba', fs=1/Ts)
                b1, a1 = signal.butter(N_lp, Wn)
                l_dlti = signal.dlti(b1, a1, dt=Ts)
            case 3:  # zoh interp. matches physics, SciPi impl. causes numerical problems??
                Wn = 2*np.pi*Fc_lp
                b1, a1 = signal.butter(N_lp, 1, 'lowpass', analog=True)
                Wlp = signal.lti(b1, a1)  # filter LTI system instance
                Wlp_ss = Wlp.to_ss()  # controllable canonical form
                Ac = Wlp_ss.A
                Bc = Wlp_ss.B
                Cc = Wlp_ss.C
                Dc = Wlp_ss.D
                Ac = Wn*Ac  # scale to get correct cut-off /SciPi garbage butter workaround
                Bc = Wn*Bc

                A_, B_, C_, D_ = balreal_ct(Ac, Bc, Cc, Dc)
                l_lti = signal.lti(A_, B_, C_, D_)
                l_dlti = l_lti.to_discrete(dt=Ts, method='zoh')
        
        len_X = len(Xcs)
        ft, fi = signal.dimpulse(l_dlti, n=2*len_X)
        
        #plt.plot(ft, fi[0].squeeze())

        # Learning and Output Matrices
        Q, L, G = learningMatrices(len_X, fi)

        # ILC with DSM
        itr = 10

        # Get ILC output codes
        #CU1 = get_ILC_control(Nb, Xcs, Dq.squeeze(), Q, L, G, itr, b1, a1, Qstep, Vmin, Qtype, YQ, ILns)
        CM1 = get_ILC_control(Nb, Xcs, Dq.squeeze(), Q, L, G, itr, b1, a1, Qstep, Vmin, Qtype, YQ, MLns)

        # Silce 1 - 1 from the front and back to remove the transient
        N_padding = int(len_X/Np)
        #CU1 = CU1.reshape(1,-1)
        CM1 = CM1.reshape(1,-1)
        # Reshape into column vector
        #CU1 = CU1[:,N_padding:-N_padding]
        CM1 = CM1[:,N_padding:-N_padding]

        # Add multiple periods
        #CU = [] 
        CM = []
        for i in range(round(Np)):
            #CU = np.append(CU, CU1[0, :-1])
            CM = np.append(CM, CM1[0, :-1])
        #CU = np.append(CU, CU1[0,-1]).astype(int)
        CM = np.append(CM, CM1[0,-1]).astype(int)
        #CU  = CU.reshape(1,-1)
        CM  = CM.reshape(1,-1)
        C = CM
        t = np.arange(0,Ts*C.size , Ts)
    case lm.ILC_SIMP:  # iterative learning control, basic implementation
        
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

        c, y1 = ilc_simple(x, G, Qfilt, Qstep, Nb, Qtype, kp, kd, Niter)  # TODO: Get this running again
        c_ = c.clip(0, 2**16-1)
        C = np.array([c_])
        print('** ILC simple end **')

# %% Post processing
SAVE_CODES_TO_FILE_AND_STOP = False  # TODO: Messy
if SAVE_CODES_TO_FILE_AND_STOP:  # save codes to file
    outfile = 'generated_codes/' + str(SC.lin).replace(" ", "_")
    np.save(outfile, C)
else:  # generate DAC output
    run_SPICE = False  # TODO: Messy
    
    match SC.dac.model:
        case dm.STATIC:  # use static non-linear quantiser model to simulate DAC
            ML = get_measured_levels(QConfig, SC.lin.method)
            YM = generate_dac_output(C, ML)  # using measured or randomised levels
            tm = t
        case dm.SPICE:  # use SPICE to simulate DAC output
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            
            outdir = 'spice_output/' + timestamp + '/'

            if os.path.exists(outdir):
                print('Putting output files in existing directory: ' + timestamp)
            else:
                os.mkdir(outdir)
            
            configf = 'sim_config'
            with open(os.path.join(outdir, configf + '.txt'), 'w') as fout:
                fout.write(SC.__str__())

            with open(os.path.join(outdir, configf + '.pickle'), 'wb') as fout:
                pickle.dump(SC, fout)
            
            spicef_list = []
            outputf_list = []
            
            if QConfig == qws.w_6bit_2ch_SPICE:
                SEPARATE_FILE_PER_CHANNEL = False  # TODO: Mr. Tidy and Mr. Neat cannot stand a mess
            else:
                SEPARATE_FILE_PER_CHANNEL = True
            
            if SEPARATE_FILE_PER_CHANNEL:
                for k in range(0,Nch):
                    c = C[k,:]
                    seed = k + 1
                    spicef, outputf = generate_spice_batch_file(c, Nb, t, Ts, QConfig, timestamp, seed, k)
                    spicef_list.append(spicef)
                    outputf_list.append(outputf)
            else:
                spicef, outputf = generate_spice_batch_file(C, Nb, t, Ts, QConfig, timestamp)
            

            if run_SPICE:  # run SPICE
                spice_path = '/home/eielsen/ngspice_files/bin/ngspice'  # newest ver., fastest (local)
                #spice_path = 'ngspice'  # 

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

    if run_SPICE or SC.dac.model == dm.STATIC:
        # Summation stage TODO: This case dependent
        if SC.lin.method == lm.BASELINE:
            K = np.ones((Nch,1))
            K[1] = 0.0  # null secondary channel (want single channel resp.)
        elif SC.lin.method == lm.NSDCAL or lm.MPC or lm.ILC:
            K = np.ones((Nch,1))
            K[1] = 0.0  # secondary channel will have zero input, null to remove any noise
        elif SC.lin.method == lm.DEM:
            K = np.ones((Nch,1))
        elif SC.lin.method == lm.PHYSCAL:
            K = np.ones((Nch,1))
            K[1] = 1e-2
        else:
            K = 1/Nch

        #yu = np.sum(K*YU, 0)
        ym = np.sum(K*YM, 0)

        TRANSOFF = np.floor(Npt*Fs/Xcs_FREQ).astype(int)  # remove transient effects from output
        #yu_avg, ENOB_U = process_sim_output(tu, yu, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, False, 'uniform')
        ym_avg, ENOB_M = process_sim_output(tm, ym, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, True, 'non-linear')

        

        results = [['Method', 'Model', 'Fs', 'Fc', 'Fx', 'ENOB'],
                [str(SC.lin), str(SC.dac), f'{Float(SC.fs):.1h}', f'{Float(SC.fc):.1h}', f'{Float(SC.carrier_freq):.1h}', f'{Float(ENOB_M):.3h}']]

        print(tabulate(results))


