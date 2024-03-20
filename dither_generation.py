#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a varitey of periodic dither signals

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import special as spcl

class pdf: #  amplitude distribution function (stochastic)
    uniform = 1
    triangular_white = 2
    triangular_hp = 3

class adf: #  amplitude distribution function (periodic)
    uniform = 1  # use a triangle wave
    triangular = 2
    cauchy = 3  # when triangle wave amplitude is ±pi/2
    gaussian = 4

def gen_stochastic(Nsamp, Nch, W, pdf_type):
    """
    Stochastic dither signal generation

    Parameters
    ----------
    Nsamp
        number of samples
    Nch
        number of channels (independent dithers)
    W
        dither amplitude
    pdf_type
        chose the dither pdf (probability distribution function)

    Returns
    -------
    dn
        the dither signal
    """

    Na = -W/2
    Nb = W/2

    match pdf_type:
        case pdf.uniform: # Rectangular PDF
            dn = np.random.uniform(low=Na, high=Nb, size=[Nch,Nsamp])
        case pdf.triangular_white: # Triangular PDF (TPDF) (white)
            d1 = np.random.uniform(low=Na, high=Nb, size=[Nch,Nsamp])
            d2 = np.random.uniform(low=Na, high=Nb, size=[Nch,Nsamp])
            dn = d1 + d2
        case pdf.triangular_hp: # Triangular PDF (TPDF) (high-pass)
            dd = np.random.uniform(low=Na, high=Nb, size=[Nch,Nsamp+1])
            dn = dd[:,0:-1] - dd[:,1:]

    return dn


def gen_periodic(t, freq=49e3, adf_type=adf.uniform):
    """
    Periodic dither signal generation

    Parameters
    ----------
    t
        time vector
    freq
        dither signal frequency
    d
        chose the dither type (amplitude distribution function)

    Returns
    -------
    dp
        the dither signal
    """

    # Generate triangle wave (can be transformed to other dither signal shapes)
    tw = (2/np.pi)*np.arcsin(np.sin(2*np.pi*freq*t))  # triangle wave vector
    
    match adf_type:
        case adf.uniform:
            dp = tw
        case adf.triangular:
            k = 0.99999  # fudge sqrt() numerical issues
            dp = np.empty(tw.size)
            for i in range(len(tw)):
                if tw[i] > 0:
                    dp[i] = 1 - np.sqrt(1 - k*tw[i])
                else:
                    dp[i] = np.sqrt(k*tw[i] + 1) - 1
        case adf.cauchy:
            k = 0.95 # fudge tan() numerical issues
            dp = np.tan(k*np.pi/2*tw)
        case adf.gaussian:
            k = 0.99 # fudge erfinv() numerical issues
            dp = np.sqrt(2)*spcl.erfinv(k*tw)
        case _:  # default to triangle wave
            dp = tw
    
    dp = dp/np.max(dp)  # normalize the dither amplitude
    
    return dp
