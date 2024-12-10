#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthesise dual specification dither.

@author: Ahmad Faza, Arnfinn Eielsen
@date: 10.04.2024
@license: BSD 3-Clause
"""

import numpy as np
from numpy import linalg
from scipy import signal
from scipy import special
import matplotlib.pyplot as plt

from utils.fir_filter_ls import fir_filter_ls


def psd_fr_2norm(S, w):
    norm = np.sum(np.abs(S)*np.mean(np.diff(w)))/(2*np.pi)
    return norm


def dual_dither(N=int(1e6), make_plots=False):
    # Specify S(omega) = G*G'; Fourier transform of R(tau)
    M = 1024  # no. of frequency samples
    w = np.linspace(0, 2*np.pi, M)  # sample whole circle
    Ihlf = np.argwhere(w < np.pi)  # indices to half-circle

    match 8:
        case 1:
            S_fr = np.tan(w[Ihlf]/2.15 + 0.1)
            S_fr = np.r_[S_fr, np.flipud(S_fr)]  # make symmetric
        case 2:
            n = np.arange(0,M)-M/2
            fc = 0.5/2  # cutoff frequency = 0.5
            g = 2*fc*np.sinc(2*fc*n)*np.kaiser(M, 1)  # windowed sinc function
            g = -g # spectral inversion
            g[int(M/2)] = g[int(M/2)] + 1
            _, g_fr = signal.freqz(g, 1, w)
            S_fr = abs(g_fr)**2
        case 3:
            #win = np.hanning(int(M/2))
            win = np.kaiser(M/2, 5)
            S_fr = np.r_[win, np.flipud(win)]  # make symmetric
        case 4:
            #win = np.hamming(M)
            win = np.kaiser(M, 5)
            #win = signal.windows.gaussian(M, 175)
            S_fr = win
        case 5:
            hu = np.linspace(0.2, 2, int(w.size/2))
            hd = np.linspace(2, 0.2, int(w.size/2))
            S_fr = np.r_[hu**2, hd**2]
        case 6:
            MM = 256 
            hs = 0.0125*np.ones(int(MM))
            hp = 3*np.ones(int(M - 2*MM))
            S_fr = np.r_[hs, hp, hs]
        case 7:
            b, a = signal.butter(1, 0.2, btype='high', analog=False)#, fs=Fs)
            signal.freqz(b, a, w)
            _, g_fr = signal.freqz(b, a, w)
            S_fr = abs(g_fr)**2
        case 8:
            d = 1e-2
            wn = np.pi/3
            ws = w - np.pi
            S_fr = wn**2/np.sqrt(4*d**2*ws**2*wn**2 + ws**4 - 2*ws**2*wn**2 + wn**4)
            Irhlf = np.argwhere(w > np.pi)  # indices to half-circle
            S_fr = np.r_[np.flipud(S_fr[Irhlf]), S_fr[Irhlf]]  # make symmetric
            #S_fr = np.r_[S_fr[Irhlf], np.flipud(S_fr[Irhlf])]  # make symmetric

    # Determining the norm/variance and analytical S(omega)
    S_fr_2norm = psd_fr_2norm(S_fr, w) 
    S_fr_ = (S_fr/S_fr_2norm)*(4/12);  # scale response to correct variance (uniform pdf)
    # recall var(y) = norm(G)^2, when y = G v, and v unity variance white noise

    if make_plots:
        fig, ax1 = plt.subplots()
        ax1.set_title('$S(\omega)$ prototype frequency response')
        ax1.plot(w.squeeze(), 10*np.log10(np.abs(S_fr.squeeze())))
        ax1.grid(True)

    # Synthesise FIR filter
    S_fr_pr = np.abs(S_fr).squeeze()  # hack to ensure positive real
    N_fir = 1024
    [R, R_win, R_beta] = fir_filter_ls(S_fr_pr, N_fir)

    wS, S_fir_fr = signal.freqz(R_win, R_beta, w)  # frequency response samples

    if make_plots:
        fig, ax2 = plt.subplots()
        ax2.set_title('$S(\omega)$ prototype vs. FIR approximation')
        ax2.plot(w.squeeze(), 10*np.log10(np.abs(S_fr.squeeze())), wS.squeeze(), 10*np.log10(np.abs(S_fir_fr)))
        ax2.grid(True)
    
    # Compute phi(omega)
    R_win_inf = np.max(R_win) + np.sqrt(np.finfo(float).eps)  # inf norm
    R_win_ = R_win/R_win_inf
    phi = 2*np.sin((np.pi/6)*R_win_)  # compensation filter coeffs.

    if make_plots:
        fig, ax3 = plt.subplots()
        ax3.set_title('$R$ filter vs. $\phi$ filter coeffs.')
        ax3.stem(np.real(phi), 'b')
        ax3.stem(np.real(R_win_), 'r')
        plt.show()

    wphi_tf, Phi_fr = signal.freqz(phi, 1, w)

    match 1:
        case 1:  # use FFT/IFFT to synth. H (same as Sondhi - 1983)
            # phi_ = circshift(phi,128)
            phi_ = np.roll(phi, int(phi.size/2))
            
            # plot coeffs.
            if make_plots:
                fig, ax4 = plt.subplots()
                ax4.set_title('$\phi$ circularly shifted filter coeffs.')
                ax4.stem(phi_)
                plt.show()

            Phi = np.fft.rfft(phi_)
            #Phi = np.fft.fft(phi_)
            
            Phi_Re = np.real(Phi)
            Inp = np.argwhere(Phi_Re < 0)
            Phi_Re[Inp] = 0

            mus = np.sqrt(np.real(Phi_Re))  # realisable filter FFT

            if make_plots:  # plot FFTs
                fig, (ax5, ax6) = plt.subplots(2)
                fig.suptitle('rfft real and imag components of $\Phi$ and $\mu$')
                ax5.plot(np.real(Phi))
                ax5.plot(np.real(mus))
                ax6.plot(np.imag(Phi))
                ax6.plot(np.imag(mus))
                plt.show()

            h = np.fft.irfft(np.real(mus)) #
            #h = np.fft.ifft(np.real(mus)) #
            h = np.roll(h, int(phi.size/2))

            if make_plots:
                fig, ax7 = plt.subplots()
                ax7.set_title('$h$ comp. filter coeffs.')
                ax7.stem(h)
                plt.show()
        case 2:  # use LS on frequency response to synth. H
            mus = np.sqrt(abs(Phi_fr))
            [h_alpha,h_alpha_win,h_beta] = fir_filter_ls(mus, 1000)
            h = h_alpha_win

    Phi_fr_2norm = psd_fr_2norm(Phi_fr, w)  # verification; should be 1

    if make_plots:
        fig, ax8= plt.subplots()
        ax8.set_title('$S(\omega)$ vs. $\Phi(\omega)$')
        ax8.plot(w/(2*np.pi), 10*np.log10(np.abs(S_fr)))
        ax8.plot(w/(2*np.pi), 10*np.log10(np.abs(Phi_fr)))
        plt.show()

    # generate coloured input to non-lin.
    v = np.random.normal(0, 1.0, N)  # normally distr. noise

    #match 1:
    #    case 1: # use comp. filter
    zi = signal.lfilter_zi(h, 1)
    z, _ = signal.lfilter(h, 1, v, zi=zi*v[0])
            #x_ = filter(h, 1, v)
    #    case 2: # use "original" filter, not impl.
    #        zi = signal.lfilter_zi(g, 1)
    #        z, _ = signal.lfilter(g, 1, v, zi=zi*v[0])
            #x_ = filter(g, 1, v)

    x = z/np.std(z) # normalise (should strictly not be neccessary)

    # non-lin. transform
    y = 2*(0.5*(1 - special.erf(-x/np.sqrt(2))) - 0.5)

    # %%
    if make_plots:
        hist_and_psd_cmp(w, Ihlf, S_fr_, y)

    return y

def hist_and_psd(y):
    # histogram to check PDF specification
    _ = plt.hist(y, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram of non-lin. transform output")
    plt.show()
    
    # PSD to check spectral specifcation
    fy, Pyy = signal.welch(y, fs=1, nperseg=y.size/500)

    plt.plot(fy, 10*np.log10(Pyy))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (V$^2$/Hz)')
    plt.show()


def hist_and_psd_cmp(w, Ihlf, S_fr_, y):
    # histogram to check PDF specification
    _ = plt.hist(y, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram of non-lin. transform output")
    plt.show()
    
    # PSD to check spectral specifcation
    fy, Pyy = signal.welch(y, fs=1, nperseg=y.size/500)
    fn = w[Ihlf]/(2*np.pi)
    fn = fn.squeeze()

    PSS = 2*abs(S_fr_[Ihlf])  # analytical desired resp.
    PSS = PSS.squeeze()

    plt.plot(fy, 10*np.log10(Pyy))
    plt.plot(fn, 10*np.log10(PSS))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (V$^2$/Hz)')
    plt.show()


def main():
    """
    Test the method.
    """
    
    y = dual_dither(N=int(1e6), make_plots=True)


if __name__ == "__main__":
    main()

# #Nh = 2047
# Nh = 1024
# #n = np.arange(0,Nh)-(Nh-1)/2
# n = np.arange(0,Nh)-Nh/2

# fc = 0.5/2  # cutoff frequency = 0.5
# g_lp = 2*fc*np.sinc(2*fc*n)*np.kaiser(Nh, 1)  # windowed sinc function

# w, g_lp_fr = signal.freqz(g_lp, 1)

# g = -g_lp # spectral inversion
# g[int(Nh/2)] = g[int(Nh/2)] + 1

# w, g_fr = signal.freqz(g, 1)

# fig, ax = plt.subplots()
# ax.plot(w, 20*np.log10(np.abs(g_lp_fr)))
# ax.grid(True)

# fig, ax = plt.subplots()
# ax.plot(w, 20*np.log10(np.abs(g_fr)))
# ax.grid(True)

# S_num = np.convolve(g, np.flipud(g))
# S_den = 1


# w, S_fr = signal.freqz(S_num, 1)

# fig, ax = plt.subplots()
# ax.plot(w, np.log10(np.abs(S_fr)))
# ax.plot(w, np.log10(np.abs(g_fr**2)))
# ax.grid(True)
# %%
