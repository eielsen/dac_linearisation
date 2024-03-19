#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File wrangling, input/output

@author: trond ytterdal, bikash adhikari, arnfinn a. eielsen
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import fileinput
import subprocess
import datetime
from scipy import signal

def addtexttofile(filename, text):
    f = open(filename, 'w')
    f.write(text)
    f.close()


def get_bit(value, bit_index):
    rval = value & (1 << bit_index)
    if rval != 0:
        return 1
    else:
        return 0


def get_pwl_string(codes, ts, nsamples, dnum, vbpc, vdd, trisefall):
    if get_bit(codes[0], dnum) == 0:
        rval = "0," + vdd + " "
    else:
        rval = "0," + vbpc + " "
    deltat = trisefall/2
    for i in range(0, nsamples-1):
        time = (i+1)*ts*1e6  # microseconds
        if get_bit(codes[i], dnum) == 0 and get_bit(codes[i+1], dnum) == 1:
            rval += " " + str(time - deltat) + "u," + vdd + " " \
                + str(time + deltat) + "u," + vbpc
        elif get_bit(codes[i], dnum) == 1 and get_bit(codes[i+1], dnum) == 0:
            rval += " " + str(time - deltat) + "u," + vbpc + " " \
                + str(time + deltat) + "u," + vdd
    rval = rval + "\n"
    return rval


def get_inverted_pwl_string(codes, ts, nsamples, dnum, vbpc, vdd, trisefall):
    if get_bit(codes[0], dnum) == 0:
        rval = "0," + vbpc + " "
    else:
        rval = "0," + vdd + " "
    deltat = trisefall/2
    for i in range(0, nsamples-1):
        time = (i+1)*ts*1e6  # microseconds
        if get_bit(codes[i], dnum) == 0 and get_bit(codes[i+1], dnum) == 1:
            rval += " " + str(time - deltat) + "u," + vbpc + " " \
                + str(time + deltat) + "u," + vdd
        elif get_bit(codes[i], dnum) == 1 and get_bit(codes[i+1], dnum) == 0:
            rval += " " + str(time - deltat) + "u," + vdd + " " \
                + str(time + deltat) + "u," + vbpc
    rval = rval + "\n"
    return rval


def run_spice_sim(codes)
    
    codes = codes.astype(int)

    # # %% Plots
    # fig, ax0 = plt.subplots()  # Create a figure containing a single axes.
    # ax0.plot(t, s, t, codes*step)
    # ax0.set_xlabel('Time (s)')
    # ax0.set_title('Input signal vs. quantised signal')
    # ax0.legend(['Input signal', 'Quantised signal'])

    # %% Generate PWL strings
    outputfile = "waveform_for_spice_sim.txt"
    nsamples = len(codes)

    vbpc = "3.28"
    vdd = "5.0"
    Tr = 1e-3 # 1e-9  # the rise-time for edges

    t1 = ""
    t2 = ""

    for i in range(0, nbits):
        m = pow(2, i)
        jstr = str(i)
        t1 += "vdp" + jstr + " data" + jstr + " 0 pwl " + \
            gs.get_pwl_string(codes, Ts, nsamples, i, vbpc, vdd, Tr)
        t2 += "vdn" + jstr + " datai" + jstr + " 0 pwl " + \
            gs.get_inverted_pwl_string(codes, Ts, nsamples, i, vbpc, vdd, Tr)

    gs.addtexttofile(outputfile, t1 + t2)

    ctrl_str = '\n' + '.save v(outf)' + \
        '\n' + '.tran 10u ' + str(t_end) + '\n'

    gs.addtexttofile('spice_cmds.txt', ctrl_str)


    #spicef = '/Volumes/Work/Codes/SPICE/SPICE_MyFiles/cs_dac_06bit_ng_spice_99kHz.cir'
    #circf = '/Volumes/Work/Codes/SPICE/SPICE_MyFiles/cs_dac_06bit_02_ngspice.cir'
    spicef = './cs_dac_06bit_ng_spice_99kHz.cir'
    circf = './cs_dac_06bit_02_ngspice.cir'

    with open(spicef, 'w') as fout:
        fin = fileinput.input([circf, outputfile,
                            'spice_cmds.txt'])
        for line in fin:
            fout.write(line)
        fin.close()

    outputf = 'output_' + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    subprocess.run(['ngspice', '-o', './spice_output/' + outputf + '.log',
                    '-r', './spice_output/' + outputf + '.bin',
                    '-b', spicef])
    





def read_bin(path='/Volumes/Work/Codes/SPICE/SPICE_MyFiles/spice_output/'):

    binfiles = [file for file in os.listdir(path) if file.endswith('.bin')]
    fid = open('/Volumes/Work/Codes/SPICE/SPICE_MyFiles/spice_output/'+ binfiles[-1],'rb')

    read_new_line = True
    count = 0
    while read_new_line:
        tline = fid.readline()

        if b'Binary:' in tline:
            read_new_line = False

        if b'No. Variables: ' in tline:
            nvars = int(tline.split(b':')[1])

        if b'No. Points: ' in tline:
            npoints = int(tline.split(b':')[1])
            
    data = np.fromfile(fid, dtype='float64')
    t_spice = data[::2]
    y_spice = data[1::2]

    # plt.plot(t_spice,y_spice)
    # plt.show()

    return t_spice,y_spice