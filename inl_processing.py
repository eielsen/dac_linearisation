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
from quantiser_configurations import quantiser_configurations, get_measured_levels, qws

def generate_random_output_levels(QuantizerConfig=4):
    """
    Generates random errors for the output levels +/-1 LSB and saves to file
    """
    # Quantiser model
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QuantizerConfig)

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
        outfile_str = "generated_output_levels_{0}_bit_{1}_QuantizerConfig_{2}.npy".format(Nb, gno, QuantizerConfig)
        outfile = os.path.join(outpath, outfile_str)
        if os.path.exists(outfile):
            gno = gno + 1
        else:
            np.save(outfile, YQn)
            break


def get_physcal_gain(QConfig):
    match QConfig:  # gain tuning (make sure secondary DAC does not saturate)
        case qws.w_16bit_NI_card:
            K_SEC = 1
        case qws.w_16bit_SPICE:
            K_SEC = 1e-2
        case qws.w_6bit_ARTI:
            K_SEC = 7.5e-2
        case qws.w_16bit_ARTI:
            K_SEC = 2e-1  # find out
        case qws.w_6bit_2ch_SPICE:
            K_SEC = 12.5e-2
        case qws.w_16bit_2ch_SPICE:
            K_SEC = 1e-2
        case _:
            K_SEC = 1
    return K_SEC


def plot_inl(QConfig=qws.w_16bit_NI_card, Ch_sel=0):
    """
    Make an INL plot, according to best practice, i.e. removing linear trend and offset.
    """

    # Quantiser model
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)
    qconfig_tab = [[Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype]]
    print(tabulate(qconfig_tab))
    
    ML = get_measured_levels(QConfig)
    
    LVLS = np.array(ML[Ch_sel])
    
    qs = np.arange(-2**(Nb-1), 2**(Nb-1), 1) # possible quantisation steps/codes (recall arange() is not inclusive)
    qs = qs.reshape(-1, 1) # ensure column vector for codes

    from matplotlib import rc

    from matplotlib import rc
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)

    plt.plot(qs, signal.detrend(LVLS)/Qstep)
    plt.xlabel("Input code")
    plt.ylabel("Least significant bits")
    plt.grid()

    plt.savefig('figures/INL_plot.pdf', format="pdf", bbox_inches="tight")
    #fig.savefig('Stylized Plots.png', dpi=300, bbox_inches='tight', transparent=True)


def gen_physcal_lut(QConfig=qws.w_16bit_NI_card, SAVE_LUT=0):
    """
    Generate physical level calibration look up table.

    Least-squares minimisation of element mismatch via a look-up table (LUT)
    to be used when a secondary calibration DAC is available
    """

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
    
    #%%
    plt.figure(10)
    plt.plot(qs, YQ, label='Uniform')
    plt.plot(qs, PRILVLS, label='Measured')
    plt.xlabel('Code')
    plt.ylabel('Ideal output level')
    plt.legend()

    QQ = np.hstack([qs, np.ones(qs.shape)]) # codes matrix for straight line least-squares fit
    YY = np.hstack([YQ, np.ones(qs.shape)]) # ideal levels matrix

    MLm = PRILVLS; # use channel 1 as Main/primary (measured levels)
    MLm = MLm.reshape(-1, 1) # ensure column vector

    thetam = np.linalg.lstsq(QQ, MLm, rcond=None)[0] # staight line fit; theta[0] is slope, theta[1] is offset

    ML = MLm - thetam[1] # remove fitted offset for measured levels
    INL = (ML - YQ)/Qstep # find the INL

    CLm = SECLVLS; # use channel 2 to calibrate/secondary (measured levels)
    CLm = CLm.reshape(-1, 1) # ensure column vector

    thetacq = np.linalg.lstsq(QQ, CLm, rcond=None)[0] # staight line fit
    print(thetacq)

    Qcal = thetacq[0] # effective quantization step for secondary channel (effective gain/scale)
    CL = Qcal*qs # resort to using scaled, ideal output for secondary calibration channel
    # (level measurements for seconduary too noisy for monotonic behavior, i.e. does more harm than good)

    # Generate the look-up table by minimising the primary output deviation from ideal
    # for every code value by adding or subtracting using the secondary
    Nl = Mq + 1  # number of output levels
    LUTcal = np.zeros(Nl)  # initalise look-up table (LUT)
    err = ML - YQ  # compute level errors (same as INL*Qstep)
    for k in range(0,Nl):
        errc = abs(err[k] + CL)  # given all secondary outputs, compute the errors for a given primary code
        LUTcal[k] = np.argmin(errc)  # save the secodary code that yields the smallest error

    LUTcal = LUTcal.astype(np.uint16)  # convert to integers

    if SAVE_LUT:
        lutfile = os.path.join('generated_physcal_luts', 'LUTcal_' + str(QConfig))
        np.save(lutfile, LUTcal)

    #%%
    plt.figure(1)
    plt.plot(YQ, LUTcal, label='Calibrated look-up table (LUT)')
    plt.xlabel('Ideal output level')
    plt.ylabel('Calibrated Code')
    plt.legend()

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


