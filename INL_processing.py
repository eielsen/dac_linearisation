#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate random INL values to emulate a non-uniform quantiser (DAC)

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
import scipy.io

from matplotlib import pyplot as plt
from configurations import quantiser_configurations
from os.path import exists

def generate_random_output_levels():
    """
    Generates random errors for the output levels +/-1 LSB and saves to file
    """
    # Quantiser model
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(3)

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
    ax2.stairs(YQ,CD)
    ax2.stairs(YQn,CD)

    fig, ax2 = plt.subplots()
    ax2.stairs(INL,CD)

    # Save to file
    gno = 1
    while True:
        outfile = "generated_output_levels_{0}_bit_{1}.npy".format(Nb, gno)
        if exists(outfile):
            gno = gno + 1
        else:
            np.save(outfile, YQn)
            break

def generate_physical_level_calibration_look_up_table(SAVE_LUT=0):
    """
    Least-squares minimisation of element mismatch via a look-up table (LUT)
    to be used when a secondary calibration DAC is available
    """
    
    # Quantiser model
    Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(4)

    # load level measurements (or randomly generated)
    #mat = scipy.io.loadmat('measurements_and_data\PHYSCAL_level_measurements_set_1.mat'); fileset = 1    
    mat = scipy.io.loadmat('measurements_and_data/PHYSCAL_level_measurements_set_2.mat'); fileset = 2

    PRILVLS = mat['PRILVLS']
    SECLVLS = mat['SECLVLS']
        
    qs = np.arange(-2**(Nb-1), 2**(Nb-1), 1) # possible quantisation steps/codes
    qs = qs.reshape(-1, 1) # ensure column vector for codes

    YQ = YQ.reshape(-1,1) # ensure column vector for ideal levels

    QQ = np.hstack([qs, np.ones(qs.shape)]) # codes matrix for straight line least-squares fit
    YY = np.hstack([YQ, np.ones(qs.shape)]) # ideal levels matrix

    MLm = PRILVLS; # use channel 1 as Main/primary (measured levels)
    MLm = MLm.reshape(-1, 1) # ensure column vector

    thetam = np.linalg.lstsq(QQ, MLm, rcond=None)[0] # staight line fit; theta[0] is slope, theta[1] is offset

    ML = MLm - thetam[1] # remove fitted offset for measured levels
    INL = (ML - YQ)/Qstep # find the INL

    CLm = SECLVLS; # use channel 2 to Calibrate/secondary (measured levels)
    CLm = CLm.reshape(-1, 1) # ensure column vector

    thetacq = np.linalg.lstsq(QQ, CLm, rcond=None)[0] # staight line fit

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
        outfile = "LUTcal"
        np.savez(outfile, LUTcal, MLm, ML, CLm, CL)

    #%%
    plt.figure(1)
    plt.plot(YQ, LUTcal, label='Calibrated look-up table (LUT)')
    plt.xlabel('Ideal output level')
    plt.ylabel('Code')
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
