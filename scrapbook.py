#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from scipy import linalg
from matplotlib import pyplot as plt

import control as ct

Hns_tf = signal.TransferFunction([1, -2, 1], [1, 0, 0], dt=1)
Mns_tf = signal.TransferFunction([2, -1], [1, 0, 0], dt=1)  # Mns = 1 - Hns

Mns = Mns_tf.to_ss()







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