#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Returns optimal filter paramters for the MPC with respect to the sampling frequency Fs and cutoff frequency Fc.

@author: Bikash Adhikari 
@date: 05.02.2025 
@license: BSD 3-Clause
"""

import numpy as np
from scipy import signal

# The cutoff frequency and filter order are constant as follows
# Cutoff Frequency (Fc) = 100 kHz
# Filter order (n) = 3
# The filter parameters are derived for the  various sampling frequencies.

def mpc_filter_parameters(FS_CHOICE):

    match FS_CHOICE:
        case 1:     #Fs = 1e6
            b1 = np.array([1.000000000000000,  -0.749062760083214,   0.353567447503785 , -0.050452041460215])
            a1 = np.array([1.000000000000000,  -1.760042814801001 ,  1.182897276395584 , -0.278062036214375])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

        case 2:     #Fs = 25e6
            b1 = np.array([1.000000000000000,  -1.990229008894999,   1.323868332535894,  -0.267430343168738])
            a1 = np.array([ 1.000000000000000,  -2.955289705643260 ,  2.911906871095763,  -0.956567417062835])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

        case 3:  #Fs = 250e6
            b1 = np.array([1.000000000000000,  -2.470675393484270,   2.043773794471696,  -0.561431085692549])
            a1 = np.array([ 1.000000000000000,-3.060784192584217,3.120349920951655,-1.059572867383303])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

        case 4:  #Fs = 1022976
            b1 = np.array([ 1.000000000000000,  -0.771083876166916 ,  0.362729852857121,  -0.052704512877984])
            a1 = np.array([ 1.000000000000000 , -1.787286425183212,   1.210436647081342 , -0.286499423621113])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace
        
        case 5:  #Fs = 16367616
            b1 = np.array([1.000000000000000,  -1.864068128668581,   1.217419475964637,  -0.264801035128820])
            a1 = np.array([ 1.000000000000000,  -2.922882099215159,   2.848783841143716,  -0.925783393814905])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

        case 6:  # Fs = 32735232
            b1 = np.array([1.000000000000000,  -2.357352708532972,   1.872697932999750,  -0.493866883597173])
            a1 = np.array([ 1.000000000000000 , -2.961378308694018 ,  2.923464166872809,  -0.962072077069174])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

        case 7:  # Fs = 65470464
            b1 = np.array([1.000000000000000,  -2.452079101922663,   2.012594245568572 , -0.546936407329641])
            a1 = np.array([1.000000000000000,  -2.991353980486378,   2.982661760283041,  -0.991306520914544])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

        case 8:  # Fs = 130940928
            b1 = np.array([1.000000000000000,  -2.421037749701214,   1.964403625777694,  -0.528090479775633])
            a1 = np.array([1.000000000000000,  -3.063446439199524,   3.125081334761360,  -1.061647191680632])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

        case 9:  #Fs = 261881856
            b1 = np.array([1.000000000000000 , -2.461113819828369,   2.025309446029859,  -0.550808973990026])
            a1 = np.array([1.000000000000000,  -3.065904839601346,   3.130535515512198,  -1.064637825987226])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

        case 10:  #Fs = 209715200
            b1 = np.array([1.000000000000000,  -2.459972414709681,   2.018602635755858,  -0.546295559927195])
            a1 = np.array([1.000000000000000,  -3.053527636182428,   3.106090383754101,  -1.052566995464895])
            A1, B1, C1, D1 = signal.tf2ss(b1, a1) # Transfer function to StateSpace

    return A1, B1, C1, D1
