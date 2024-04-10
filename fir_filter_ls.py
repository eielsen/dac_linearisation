#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthesise FIR filter from complex valued frequency response samples.

@author: Arnfinn Eielsen
@date: 08.04.2024
@license: BSD 3-Clause
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
import control as ct


def fir_filter_ls(H, p):
    """
    Synthesise FIR filter from complex valued frequency response samples,
    using least-squares. No restriction on phase response.
    
    Inputs:
        H - frequency response samples from 0 to 2*pi
        p - desired FIR filter length
    
    Outputs:
        alpha      - FIR filter coefficients
        alpha_win - windows FIR filter coefficients
        beta      - denominator that can be used to null the phase when computing the frequency response
    """
    
    M = H.size  # no. of frequency samples

    X = np.zeros((M,p), dtype=np.complex_)

    if np.mod(p,2):  # odd
        q = int((p-1)/2)
    else:  # even
        q = int(p/2)

    K = 2*np.pi/M
    ps = np.arange(0,p) #0:p-1
    pq = K*(ps-q)

    for k in range(0, M):  # iterate over frequency samples
        X[k,:] = np.exp(-1j*k*pq)
    
    b = H.reshape(-1, 1)  # column vector

    # min(||b - X*alpha||)
    alpha = np.linalg.lstsq(X, b, rcond=None)[0]  # FIR coefficients by least-squares
    alpha = np.real(alpha.reshape(1, -1))  # real valued row vector
    
    win = np.hanning(alpha.size)
    alpha_win = win*alpha # reduce ripples due to implicit rectangular windowing

    beta = np.r_[1.0, np.zeros(q-1)]

    return alpha, alpha_win, beta




M = int(1e4)  # no. of frequency samples
w = np.linspace(0.0, 2*np.pi, M)  # sample whole circle

match 1:
    case 1:
        sys = ct.drss(5, outputs=1, inputs=1)  # random, stable LTI system
        Hss = signal.dlti(sys.A, sys.B, sys.C, sys.D, dt=1)
        wH_fr, H_fr = signal.dfreqresp(Hss, w)
    case 2:
        hu = np.linspace(0, 1, int(w.size/2))
        hd = np.linspace(1, 0, int(w.size/2))
        H_fr = np.r_[hu, hd]
    case 3:
        hs = 0.0125*np.ones(int(1000))
        hp = 3*np.ones(int(8000))
        H_fr = np.r_[hs, hp, hs]


I = np.argwhere(w < np.pi)

plt.plot(w[I], abs(H_fr[I]))
plt.show()

p = 200  # filter length
alpha, alpha_win, beta = fir_filter_ls(H_fr, p)


wh, h = signal.freqz(alpha_win.squeeze())
H = signal.dlti(alpha, beta, dt=1)
#wh, h = signal.dfreqresp(H, w[I])

#fig, ax1 = plt.subplots()
#ax1.set_title('Digital filter frequency response')
# ax1.plot(w, 20*np.log10(h), 'b')
#ax1.plot(wh, abs(h), 'b')
plt.plot(wh, abs(h), 'b')

# ax1.set_ylabel('Amplitude [dB]', color='b')
# ax1.set_xlabel('Frequency [rad/sample]')
# ax2 = ax1.twinx()
# angles = np.unwrap(np.angle(h))
# ax2.plot(wh, angles, 'g')
# ax2.set_ylabel('Angle (radians)', color='g')
# ax2.grid(True)
# ax2.axis('tight')
plt.show()

"""
    switch nargin
        case 2

        case 1
            p = 50; % filter length
        otherwise
            HM = drss(5); % random LTI system prototype/reference

            HMtf = tf(HM);
            bref = HMtf.Numerator{:};
            aref = HMtf.Denominator{:};

            M = 1e3; % no. of frequency samples
            w = linspace(0,2*pi,M); % sample whole circle
            H = freqz(bref,aref,w); % frequency response samples

            p = 50; % filter length
    end
"""