#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from scipy import linalg
from matplotlib import pyplot as plt
from numpy import matlib

import control as ct

from utils.quantiser_configurations import quantiser_configurations

from utils.balreal import balreal, balreal_ct

%reload_ext autoreload
%autoreload 2

# Fc_lp = 10e3
# N_lp = 3


# Wn = 2*np.pi*Fc_lp
# b1, a1 = signal.butter(N_lp, Wn, 'lowpass', analog=True)
# Wlp = signal.TransferFunction(b1, a1)  # filter LTI system instance
# Wlp_ss = Wlp.to_ss()  # controllable canonical form
# A = Wlp_ss.A
# B = Wlp_ss.B
# C = Wlp_ss.C
# D = Wlp_ss.D
# A_, B_, C_, D_ = balreal_ct(A, B, C, D)
# Wlp_ss_d = signal.cont2discrete((A_, B_, C_, D_), dt=1e-6, method='zoh')



Wn = 100e3/(25e6/2)
bd, ad = signal.butter(3, Wn)
Wlpd = signal.dlti(bd, ad, dt=1/25e6)

Nsamp = int(1e6)
y = np.random.uniform(low=-1.0, high=1.0, size=[1,Nsamp])

y = y.reshape(-1, 1)  # ensure the vector is a column vector

#y_avg_out = signal.lfilter(bd, ad, y)

y_avg_out = signal.dlsim(Wlpd, y) # filter the output
y_avg = y_avg_out[1] # extract the filtered data; lsim returns (T, y, x) tuple, want output y


"""
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
       A1 = Wlp_ss_d.A
       B1 = Wlp_ss_d.B
       C1 = Wlp_ss_d.C
       D1 = Wlp_ss_d.D
"""


#dt = 1e-6
#A = np.eye(3, k=-1) 
#A[0,:] = a1[1:4]
#B = np.eye(3,1, k=0)
#C = np.zeros((1,3))
#C[0,2] = b1[-1]
#D = np.array([0])

#Wlp_ss = Wlp.to_ss()
#Wlp_ss_d = signal.cont2discrete((Wlp_ss.A, Wlp_ss.B, Wlp_ss.C, Wlp_ss.D), dt, method='zoh')
#Ad = Wlp_ss_d[0]
#Bd = Wlp_ss_d[1]
#Cd = Wlp_ss_d[2]
#Dd = Wlp_ss_d[3]
#A, B, C, D = balreal(Ad, Bd, Cd, Dd)



# Wn = 2*np.pi*Fc_lp
# b, a = signal.butter(N_lp, Wn, 'lowpass', analog=True)
# Wlp = signal.lti(b, a)  # filter LTI system instance
# Wlp_ss = Wlp.to_ss()
# dt = 1e-3
# Wlp_ss_d = signal.cont2discrete((Wlp_ss.A, Wlp_ss.B, Wlp_ss.C, Wlp_ss.D), dt, method='zoh')
# Ad = Wlp_ss_d[0]
# Bd = Wlp_ss_d[1]
# Cd = Wlp_ss_d[2]
# Dd = Wlp_ss_d[3]

# G = Wlp.to_discrete(dt, method='zoh')

#Ad, Bd, Cd, Dd = balreal(Wlp_ss_d[0], Wlp_ss_d[1], Wlp_ss_d[2], Wlp_ss_d[3])


# import tkinter as tk
# root = tk.Tk()
# root.configure(bg='red')
# root.overrideredirect(True)
# root.state('normal')
# root.after(100, root.destroy) # set the flash time to 100 milliseconds
# root.mainloop()



#N = 1
#for k in range(0,N):
#    print(k)

#QConfig = 4
#Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

# DEM code input range
#M = 2*(2**Nb - 1)
#cmin = 2**(Nb-1) - 1
#cmax = M - 2**(Nb-1) + 1
#Qseg = Rng/(cmax-cmin)  # segmented step-size (LSB)

#if RUN_LIN_METHOD == lin_method.DEM:
#    SIGNAL_OFFSET = Rng/2# - Qstep/2
#else:
    

"""
Ks =

           1           1
           2           2
           4           4
           8           8
          16          16
          32          32
          64          64
         128         128
         256         256
         512         512
        1024        1024
        2048        2048
        4096        4096
        8192        8192
       16384       16384
       32768       32768
"""

# DEM mapping from output segment weights to codes
#Ks = 2**np.arange(0,Nb)
#Ks = matlib.repmat(Ks.reshape(-1, 1),1,2)

# from balreal import balreal

# Hns_tf = signal.TransferFunction([1, -2, 1], [1, 0, 0], dt=1)
# Mns_tf = signal.TransferFunction([2, -1], [1, 0, 0], dt=1)  # Mns = 1 - Hns

# Mns = Mns_tf.to_ss()

# A = Mns.A
# B = Mns.B
# C = Mns.C
# D = Mns.D


# A_, B_, C_, D_ = balreal(A, B, C, D)


# %%



#M = np.random.normal(0, 2.5, size=(4,10))

#x = np.array([1, 2, 3, 4, 3, 2, 1, 0, 1, 1, 1], np.int32)

#for k in range(0,M.shape[0]):
#    print(k)
#    print(M[k,x])


#M[x]

# w, h = signal.freqz(b, a)
# h[0] = h[1]
# f = Fs*w/(2*np.pi)

# fig, ax1 = plt.subplots()
# ax1.set_title('Digital filter frequency response')
# ax1.semilogx(f, 20*np.log10(abs(h)), 'b')
# ax1.set_ylabel('Amplitude (dB)', color='b')
# ax1.set_xlabel('Frequency (Hz)')
# ax1.grid(True)

# fig, ax2 = plt.subplots()
# angles = (180/np.pi)*np.unwrap(np.angle(h))
# ax2.semilogx(f, angles, 'g')
# ax2.set_ylabel('Angle (degrees)', color='g')
# ax2.set_xlabel('Frequency (Hz)')
# ax2.grid(True)

# plt.show()

# dsf = Dsf[1, :]
# f, Pxx_den = signal.welch(dsf, Fs)

# fig, ax1 = plt.subplots()
# ax1.loglog(f, Pxx_den)
# ax1.set_xlabel('frequency [Hz]')
# ax1.set_ylabel('PSD [V**2/Hz]')
# plt.show()