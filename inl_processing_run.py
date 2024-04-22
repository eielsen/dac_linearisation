#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run INL processing

@author: Arnfinn Eielsen
@date: 28.02.2024
@license: BSD 3-Clause
"""

%reload_ext autoreload
%autoreload 2

import csv

import numpy as np

from matplotlib import pyplot as plt

from inl_processing import gen_physcal_lut, plot_inl
from quantiser_configurations import qws

from spice_utils import run_spice_sim, read_spice_bin_file_with_most_recent_timestamp

#gen_physcal_lut(QConfig=5, FConfig=3, SAVE_LUT=1) # Trond 16bit
#gen_physcal_lut(QConfig=4, FConfig=2, SAVE_LUT=1)
#gen_physcal_lut(QConfig=qws.w_6bit_ARTI, SAVE_LUT=1) # ARTI 6bit
#gen_physcal_lut(QConfig=qws.w_16bit_ARTI, SAVE_LUT=1) # ARTI 6bit
#gen_physcal_lut(QConfig=qws.w_6bit_2ch_SPICE, SAVE_LUT=True) # Trond 2ch 6bit
#gen_physcal_lut(QConfig=qws.w_16bit_2ch_SPICE, SAVE_LUT=True) # Trond 2ch 16bit
#gen_physcal_lut(QConfig=qws.w_16bit_6t_ARTI, SAVE_LUT=True) # ARTI 16bit

#generate_random_output_levels(QuantizerConfig=4)

#plot_inl(QConfig=qws.w_16bit_NI_card, Ch_sel=1)
#plot_inl(QConfig=qws.w_16bit_ARTI, Ch_sel=0)
#plot_inl(QConfig=qws.w_6bit_ARTI, Ch_sel=0)
plot_inl(QConfig=qws.w_16bit_6t_ARTI, Ch_sel=0)


# Nb = 16
# path = './spice_output/'
# N = 2**Nb - 1
# ML = np.zeros(N)
# MLnsamp = np.zeros(N)

# for k in range(0,N):
#     c = np.array([1, 1])*(k*1000)
    
#     Ts = 1e-4
#     t = np.array([0, 1])*Ts
    
#     QConfig = 4

#     run_spice_sim(c, Nb, t, Ts, QConfig)

#     t_spice, y_spice = read_spice_bin_file_with_most_recent_timestamp(path)
    
#     i_transient = np.ceil(y_spice.size/10).astype(int)
#     ML[k] = np.mean(y_spice[i_transient:])


# outfile = 'SPICE_levels_16bit_seed_1'
# np.save(outfile, ML)

# ML = np.load('SPICE_levels_ARTI_6bit.npy')  # measured levels

def read_csv_levels():
    codes = []
    nominal = []
    actual_1 = []
    actual_2 = []

    #infile = 'measurements_and_data/cs_dac_16bit_04_levels.csv'
    infile = 'measurements_and_data/ARTI_cs_dac_6b_levels.csv'

    with open(infile, newline='') as csv_file:
        #csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        csv_reader = csv.reader(csv_file)
        line_count = 0    
        for row in csv_reader:
            #if line_count > 1:
            if line_count > 0:
                codes.append(int(row[0]))
                nominal.append(float(row[1]))
                actual_1.append(float(row[2]))
                actual_2.append(float(row[3]))
            line_count += 1

    codes = np.array(codes)
    nominal = np.array(nominal)
    actual_1 = np.array(actual_1)
    actual_2 = np.array(actual_2)

    plt.plot(actual_1)
    plt.plot(actual_2)

    ML = np.array([actual_1, actual_2])

    #outfile = measurements_and_data/SPICE_levels_16bit'
    outfile = 'measurements_and_data/SPICE_levels_ARTI_6bit'

    np.save(outfile, ML)
