#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate random INL values to emulate a non-uniform quantiser (DAC)

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
import scipy.io
import os
import sys
from tabulate import tabulate
from scipy import signal
from matplotlib import pyplot as plt

import sys
sys.path.append('../')

from utils.quantiser_configurations import quantiser_configurations, get_measured_levels, qs

def generate_random_output_levels(QConfig=4):
    """
    Generate random errors for the output levels
    with a deviation not exceeding +/-1 LSB (to maintain monotone transfer)
    and save to file.
    """
    # Quantiser model
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

    CD = np.arange(0, 2**Nb)
    CD = np.append(CD, CD[-1]+1)

    # Generate non-linearity
    #RL = 1e-1*(YQ**3)
    RL = Qstep*(-1 + 2*np.random.randn(YQ.size))
    YQn = YQ + RL

    INL = (YQn - YQ)/Qstep
    DNL = INL[1:-1] - INL[0:-2]

    # Plot
    fig, ax1 = plt.subplots() # create a figure containing a single axes
    ax1.plot(RL)

    fig, ax2 = plt.subplots()
    ax2.stairs(YQ, CD)
    ax2.stairs(YQn, CD)

    fig, ax2 = plt.subplots()
    ax2.stairs(INL, CD)

    # Save to file
    gno = 1
    while True:
        outpath = "generated_output_levels"
        outfile_str = "generated_output_levels_{0}_bit_{1}_QuantiserConfig_{2}.npy".format(Nb, gno, QConfig)
        outfile = os.path.join(outpath, outfile_str)
        if os.path.exists(outfile):
            gno = gno + 1
        else:
            np.save(outfile, YQn)
            break


def get_physcal_gain(QConfig):
    """
    Tuned gain coefficients for the secondary channel when using physical calibration (PHYCAL).

    This is tune manually at this point
        - ensure full range og secondary DAC matches maximum linear error (i.e. INL) on the primary DAC
        - reduce gain to ensure some headroom, mainly to be able to compensate for errors in gain and offset 
    """
    
    match QConfig:  # gain tuning (make sure secondary DAC does not saturate)
        case qs.w_16bit_NI_card: K_SEC = 1
        case qs.w_16bit_SPICE: K_SEC = 1e-2
        case qs.w_6bit_ARTI: K_SEC = 7.5e-2
        case qs.w_16bit_ARTI: K_SEC = 2e-1  # find out
        case qs.w_6bit_2ch_SPICE: K_SEC = 0.1 #12.5e-2
        case qs.w_16bit_2ch_SPICE: K_SEC = 1e-2
        case qs.w_16bit_6t_ARTI: K_SEC = 2e-2
        case qs.w_6bit_ZTC_ARTI: K_SEC = 0.007
        case qs.w_10bit_ZTC_ARTI: K_SEC = 0.005
        case _: K_SEC = 1
    return K_SEC


def plot_inl(QConfig=qs.w_16bit_NI_card, Ch_sel=0):
    """
    Make an INL plot, according to best practice, i.e. removing linear trend and offset.
    """

    # Quantiser model
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)
    qconfig_tab = [[Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype]]
    print(tabulate(qconfig_tab))
    
    ML = get_measured_levels(QConfig)
    
    LVLS1 = np.array(ML[0])#Ch_sel])
    LVLS2 = np.array(ML[1])#Ch_sel])
    
    qs = np.arange(-2**(Nb-1), 2**(Nb-1), 1) # possible quantisation steps/codes (recall arange() is not inclusive)
    qs = qs.reshape(-1, 1) # ensure column vector for codes

    print(np.max(LVLS1))
    print(np.min(LVLS1))
    
    #from matplotlib import rc
    #rc('font',**{'family':'sans-serif'})
    #rc('font',**{'family':'serif','serif':['Times']})
    #rc('text', usetex=False)

    plt.figure(10)
    plt.plot(qs, LVLS1/Qstep, label='Channel 1')
    plt.plot(qs, LVLS2/Qstep, label='Channel 2')
    plt.legend()
    plt.xlabel("Input code")
    plt.ylabel("Output (least significant bits)")
    plt.grid()
    plt.show()

    plt.figure(20)
    plt.plot(qs, signal.detrend(LVLS1)/Qstep, label='Channel 1')
    plt.plot(qs, signal.detrend(LVLS2)/Qstep, label='Channel 2')
    plt.legend()
    plt.xlabel("Input code")
    plt.ylabel("INL (least significant bits)")
    plt.grid()
    plt.show()

    #plt.savefig('figures/INL_plot.pdf', format='pdf', bbox_inches='tight')
    #plt.savefig('figures/INL_plot.svg', format='svg', bbox_inches='tight')
    #fig.savefig('Stylized Plots.png', dpi=300, bbox_inches='tight', transparent=True)


def generate_physcal_lut(QConfig=qs.w_16bit_NI_card, UNIFORM_SEC=1, SAVE_LUT=0):
    """
    Generate physical level calibration look up table.

    Least-squares minimisation of element mismatch via a look-up table (LUT)
    to be used when a secondary calibration DAC is available.
    """

    outpath = '../generated_physcal_luts'

    # Quantiser model
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)
    qconfig_tab = [[Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype]]
    print(tabulate(qconfig_tab))

    ML = get_measured_levels(QConfig)
    PRILVLS = ML[0,:]
    SECLVLS = ML[1,:]
    
    K_SEC = get_physcal_gain(QConfig)

    PRILVLS = np.array(PRILVLS)
    SECLVLS = K_SEC*np.array(SECLVLS)

    qs = np.arange(-2**(Nb-1), 2**(Nb-1), 1) # possible quantisation steps/codes (recall arange() is not inclusive)
    qs = qs.reshape(-1, 1) # ensure column vector for codes

    YQ = YQ.reshape(-1,1) # ensure column vector for ideal levels
    YQ = YQ - np.mean(YQ)

    QQ = np.hstack([qs, np.ones(qs.shape)])  # codes matrix for straight line least-squares fit
    #YY = np.hstack([YQ, np.ones(qs.shape)])  # ideal levels matrix

    MLm = PRILVLS;  # use channel 1 as main/primary (measured levels)
    MLm = MLm.reshape(-1, 1)  # ensure column vector

    match 3:
        case 1:
            thetam = np.linalg.lstsq(QQ, MLm, rcond=None)[0]  # straight line fit; theta[0] is slope, theta[1] is offset
            ML = MLm - thetam[1]  # remove fitted offset for measured levels
        case 2:
            MLmmax = np.max(MLm)
            MLmmin = np.min(MLm)
            ML = MLm - (MLmmax + MLmmin)/2
        case 3:
            ML = MLm - np.mean(MLm)
        case 4:
            ML = MLm
    
    #YQ = thetam[0]*qs  # ideal levels (given curve-fit)
    INL = (ML - YQ)/Qstep  # find the INL

    plt.figure(10)
    plt.plot(qs, YQ, label='Uniform')
    plt.plot(qs, PRILVLS, label='Measured')
    plt.xlabel('Code')
    plt.ylabel('Ideal output level')
    plt.legend()

    CLm = SECLVLS  # use channel 2 to calibrate/secondary (measured levels)
    CLm = CLm.reshape(-1, 1)  # ensure column vector

    if UNIFORM_SEC:
        thetacq = np.linalg.lstsq(QQ, CLm, rcond=None)[0]  # straight line fit
        print(thetacq)

        Qcal = thetacq[0]  # effective quantization step for secondary channel (effective gain/scale)
        CL = Qcal*qs  # resort to using scaled, ideal output for secondary calibration channel
        # (level measurements for secondary too noisy for monotonic behaviour, i.e. does more harm than good)
    else:
        CL = CLm
    
    # Generate the look-up table by minimising the primary output deviation from ideal
    # for every code value by adding or subtracting using the secondary
    Nl = Mq + 1  # number of output levels
    LUTcal = np.zeros(Nl)  # initialise look-up table (LUT)
    err = ML - YQ  # compute level errors (same as INL*Qstep)
    for k in range(0,Nl):
        errc = abs(err[k] + CL)  # given all secondary outputs, compute the errors for a given primary code
        LUTcal[k] = np.argmin(errc)  # save the secondary code that yields the smallest error

    LUTcal = LUTcal.astype(np.uint16)  # convert to integers

    if SAVE_LUT:
        lutfile = os.path.join(outpath, 'LUTcal_' + str(QConfig))
        np.save(lutfile, LUTcal)

    #%%
    plt.figure(1)
    plt.plot(YQ, LUTcal, label='Calibrated look-up table (LUT)')
    plt.xlabel('Ideal output level')
    plt.ylabel('Calibrated Code')
    plt.legend()

    if UNIFORM_SEC:
        plt.figure(2)
        plt.plot(YQ, CL[LUTcal], label='Scaled ideal levels as secondary\n(error from INL is tiny for secondary due to small gain)')
        plt.plot(YQ, CLm[LUTcal], label='Measured levels as secondary\n(ostensibly better with good enough measurements)')
        plt.plot(YQ, INL*Qstep, label=r"$INL \cdot Q$")
        plt.xlabel('Ideal output level')
        plt.ylabel('Secondary voltage output')
        plt.legend()

        plt.figure(3)
        plt.plot(YQ, CL[LUTcal]-CLm[LUTcal], label='Error using ideal secondary levels vs measured')
        plt.xlabel('Ideal output level')
        plt.ylabel('Difference, secondary voltage output')
        plt.legend()

    plt.figure(4)
    plt.plot(YQ, CL[LUTcal] + INL*Qstep, label=r"INL calibration results (residual: $INL \cdot Q + Y_{C}$)")
    plt.xlabel('Ideal output level')
    plt.ylabel('Combined voltage output')
    plt.legend()


