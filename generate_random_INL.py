#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate random INL values to emulate a non-uniform quantiser (DAC)

@author: Arnfinn Aas Eielsen
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from matplotlib import pyplot as plt
from configurations import quantiser_configurations
from os.path import exists

#%% Quantiser model
Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(3)

CD = np.arange(0, 2**Nb)
CD = np.append(CD, CD[-1]+1)

#%% Generate non-linearity
#RL = 1e-1*(YQ**3)
RL = Qstep*(-1 + 2*np.random.randn(YQ.size))
YQn = YQ + RL

INL = (YQn - YQ)/Qstep
DNL = INL[1:-1] - INL[0:-2]

#%% Plot
fig, ax1 = plt.subplots() # create a figure containing a single axes
ax1.plot(RL)

fig, ax2 = plt.subplots()
ax2.stairs(YQ,CD)
ax2.stairs(YQn,CD)

fig, ax2 = plt.subplots()
ax2.stairs(INL,CD)

#%% Save to file
gno = 1
while True:
    outfile = "generated_output_levels_{0}_bit_{1}.npy".format(Nb, gno)
    if exists(outfile):
        gno = gno + 1
    else:
        np.save(outfile, YQn)
        break
