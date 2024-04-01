#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative learning control (ILC) using PD type learning filter.

This is amongst the simplest ILC implementations.

@author: Arnfinn Aas Eielsen
@date: 25.03.2024
@license: BSD 3-Clause
"""

# %%
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

import dither_generation
from static_dac_model import generate_dac_output, quantise_signal, generate_codes, quantiser_type
from quantiser_configurations import quantiser_configurations, quantiser_word_size

def ilc_simple(r, G, Qfilt, Qstep, Nb, Qtype=quantiser_type.midtread, kp=0.25, kd=10.0, Niter=25):
    """
    ILC using PD-type learning filter
    """

    Dq = dither_generation.gen_stochastic(r.size, 1, Qstep, dither_generation.pdf.triangular_hp)
    Dq = Dq.squeeze()

    M = Qfilt.size
    MM = int((M-1)/2)

    rq = quantise_signal(r + Dq, Qstep, Qtype)
    y0_out = signal.dlsim(G, Qstep*rq)  # initial open-loop response
    y0 = y0_out[1].flatten()

    e0 = r - y0  # initial error
    e = np.insert(e0, 0, 0.0)  # pad for D term
    print(e.size)

    u = np.zeros(r.size)  # init
    for j in range(1, Niter): 
        s = u + kp*e[1:] + kd*np.diff(e)
        u = np.convolve(Qfilt, s)  # Q filter (zero-phase)
        u = u[MM:-MM]
        uq = quantise_signal(u + Dq, Qstep, Qtype)
        y1_out = signal.dlsim(G, Qstep*uq)
        y1 = y1_out[1].flatten()
        e1 = r - y1
        e = np.insert(e1, 0, 0.0) # pad for D term

    # Generate codes
    c = generate_codes(uq, Nb, Qtype)

    return c, y1


def plot_freq_resp(H):
    w, h = signal.freqz(H)
    w = w/(np.pi)
    angles = np.unwrap(np.angle(h))

    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, 20*np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')
    
    ax2 = ax1.twinx()
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    plt.show()


def plot_errors(t, ei, ef):
    fig, ax1 = plt.subplots()
    ax1.set_title('Error comparison')
    ax1.plot(t, ei, 'b', label='Init. err.')
    ax1.set_ylabel('Amplitude', color='b')
    ax1.set_xlabel('Time [s]')
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(t, ef, 'g', label='Final err.')
    ax2.set_ylabel('Amplitude', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    plt.legend()
    plt.show()


def main():
    """
    Test the method.
    """
    # Sampling config.
    Fs = 1e6  # sampling frequency (Hz)
    Ts = 1/Fs  # sampling time (s)

    # Plant: Butterworth or Bessel reconstruction filter
    G_Fc = 1e4
    match 1:
        case 1:
            Wn = 2*np.pi*G_Fc
            b, a = signal.butter(3, Wn, 'lowpass', analog=True)
            Wlp = signal.lti(b, a)
            G = Wlp.to_discrete(dt=Ts, method='zoh')  # exact
        case 2:
            Wn = G_Fc/(Fs/2)  # Normalized cutoff frequency
            b, a = signal.butter(3, Wn)  # bilinear (?)
            G = signal.dlti(b, a, dt=Ts)

    # Quantiser config.
    if False:
        Nb = 16  # word-size
        Mq = 2**Nb-1  # max. code
        Vmin = -1  # volt
        Vmax = 1  # volt
        Rng = Vmax - Vmin  # voltage range
        Qstep = Rng/Mq  # step-size (LSB)
        Qtype = quantiser_type.midtrea
    else:
        QConfig = quantiser_word_size.w_16bit_SPICE
        Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

    # Generate reference signal
    AMP = 0.95*(Rng/2)  # amplitude
    FREQ = 99  # fundamental frequency
    tau = 1/FREQ

    NP = 4  # number of periods
    t = np.arange(0, round((NP*tau)/Ts)+1, 1)*Ts  # wonky due to round-off errors (?)

    # Use floor() and abs() to compute triangle wave signal
    r = np.sin(2*np.pi*FREQ*t)
    #DT = 1/(4*FREQ)
    #r = 2*np.abs(2*FREQ*(t-DT) - 2*np.floor((FREQ*(t-DT)))-1) - 1
    r = AMP*r

    # Q filter
    M = 2001  # Support/filter length/no. of taps
    Q_Fc = 2.0e4  # Cut-off freq. (Hz)
    alpha = (np.sqrt(2)*np.pi*Q_Fc*M)/(Fs*np.sqrt(np.log(4)))
    sigma = (M - 1)/(2*alpha)
    Qfilt = signal.windows.gaussian(M, sigma)
    Qfilt = Qfilt/np.sum(Qfilt)

    plot_freq_resp(Qfilt)

    rf = np.convolve(Qfilt, r) # Q filter
    MM = int((M-1)/2)
    rff = rf[MM:-MM]
    # plt.plot(t,r,t,yff)  # BW limit effect on ref. tracking potential
    # plt.show

    #q = dither_generation.gen_stochastic(t.size, 1, Qstep, dither_generation.pdf.triangular_hp)

    x = r

    kp = 0.3
    kd = 20
    Niter = 75

    c, y1 = ilc_simple(x, G, Qfilt, Qstep, Nb, Qtype, kp, kd, Niter)

    print(min(c))
    print(max(c))

    #rq = Qstep*np.floor(r/Qstep + 0.5)  # mid-tread quantiser
    rq = quantise_signal(r, Qstep, quantiser_type.midtread)

    y0_out = signal.dlsim(G, Qstep*rq)  # initial open-loop response
    y0 = y0_out[1].flatten()

    Ntrans = round(tau/(2*Ts))

    t = t[Ntrans:-Ntrans]
    r = r[Ntrans:-Ntrans]
    y0 = y0[Ntrans:-Ntrans]
    y1 = y1[Ntrans:-Ntrans]

    e0 = r - y0  # initial error
    e1 = r - y1  # final error

    plt.figure()
    plt.plot(t, r, label='Ref.')
    plt.plot(t, y0-0.1, label='Init. out.')
    plt.plot(t, y1+0.1, label='Final. out.')
    plt.legend()
    plt.show()

    plot_errors(t, e0, e1)

    #plt.figure()
    #plt.plot(t,c)
    #plt.show()


if __name__ == "__main__":
    main()
