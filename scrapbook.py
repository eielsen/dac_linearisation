#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

Hns_tf = signal.TransferFunction([1, -2, 1], [1, 0, 0], dt=1)
Mns_tf = signal.TransferFunction([2, -1], [1, 0, 0], dt=1)  # Mns = 1 - Hns

Mns = Mns_tf.to_ss()


# %%

b, a = signal.butter(3, 0.25, 'lowpass', analog=False)
Wlp = signal.lti(b, a)  # filter LTI system instance
sys = Wlp.to_ss()

A = sys.A
B = sys.B
C = sys.C
D = sys.D



#sys_init = ss(A,B,C,D,1);

Wr = dlyap(A,B*B.');
Wo = dlyap(A.',C.'*C);

% Wr = Lr*Lr'

% A = R'*R
Lr_ = chol(Wr);
Lr = Lr_.';

Lo_ = chol(Wo);
Lo = Lo_.';

% A = U*S*V'
[U,S,V] = svd(Lo.'*Lr);

T = (Lr*V)*S^(-1/2);

A_ = inv(T)*A*T
B_ = inv(T)*B
C_ = C*T

sys_bal = ss(A_,B_,C_,D,1);

figure(1)
step(sys_init)
figure(2)
step(sys_bal)

if 0 % test if Wr = Wo, different methods
    inv(T)*Wr*inv(T.')
    T.'*Wo*T
    
    lyap(A_,B_*B_.')
    lyap(A_.',C_.'*C_)
end








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