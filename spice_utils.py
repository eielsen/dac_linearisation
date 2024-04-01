#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File wrangling for SPICE simulations.

@author: Trond Ytterdal, Bikash Adhikari, Arnfinn Eielsen
@date: 19.03.2024
@license: BSD 3-Clause
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import fileinput
import subprocess
import datetime
from scipy import signal

from quantiser_configurations import quantiser_word_size


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


def get_pwl_string(c, Ts, Ns, dnum, vbpc, vdd, trisefall):
    """
    Generate picewise linear (PWL) waveform description string to be read by SPICE.
    
    Arguments
        c - codes
        Ts - sampling time (in microseconds)
        Ns - number of samples
        vbpc, vdd, trisefall - waveform specs.
    
    Returns
        rval - PWL string
    """
    if get_bit(c[0], dnum) == 0:
        rval = "0," + vdd + " "
    else:
        rval = "0," + vbpc + " "
    deltat = trisefall/2
    for i in range(0, Ns-1):
        time = (i+1)*Ts*1e6  # microseconds
        if get_bit(c[i], dnum) == 0 and get_bit(c[i+1], dnum) == 1:
            rval += " " + str(time - deltat) + "u," + vdd + " " \
                + str(time + deltat) + "u," + vbpc
        elif get_bit(c[i], dnum) == 1 and get_bit(c[i+1], dnum) == 0:
            rval += " " + str(time - deltat) + "u," + vbpc + " " \
                + str(time + deltat) + "u," + vdd
    rval = rval + "\n"

    return rval


def get_inverted_pwl_string(c, Ts, Ns, dnum, vbpc, vdd, trisefall):
    """
    Generate inverted picewise linear (PWL) waveform description string to be read by SPICE.
    
    Arguments
        c - codes
        Ts - sampling time (in microseconds)
        Ns - number of samples
        vbpc, vdd, trisefall - waveform specs.
    
    Returns
        rval - PWL string
    """
    if get_bit(c[0], dnum) == 0:
        rval = "0," + vbpc + " "
    else:
        rval = "0," + vdd + " "
    deltat = trisefall/2
    for i in range(0, Ns-1):
        time = (i+1)*Ts*1e6  # microseconds
        if get_bit(c[i], dnum) == 0 and get_bit(c[i+1], dnum) == 1:
            rval += " " + str(time - deltat) + "u," + vbpc + " " \
                + str(time + deltat) + "u," + vdd
        elif get_bit(c[i], dnum) == 1 and get_bit(c[i+1], dnum) == 0:
            rval += " " + str(time - deltat) + "u," + vdd + " " \
                + str(time + deltat) + "u," + vbpc
    rval = rval + "\n"

    return rval


def run_spice_sim(spicef, outdir, outputf):
    """
    Run SPICE simulaton using provided filenames

    Arguments
        spicef - SPICE batch file
        outputf - Output files name
    """
    
    print(spicef)
    print(outputf)

    # outdir = 'spice_output/'

    subprocess.run(['ngspice', '-o', outdir + outputf + '.log',
                    '-r', outdir + outputf + '.bin',
                    '-b', outdir + spicef])


def generate_spice_batch_file(c, Nb, t, Ts, QConfig, seed, timestamp, seq):
    """
    Set up SPICE simulaton for a given DAC circuit description

    Arguments
        c - codes
        Nb - no. of bit
        t - time vector
        Ts - sampling time
        timestamp
        seq - sequence
    """
    c = c.astype(int)
    nsamples = len(c)
    waveformfile = 'waveform_for_spice_sim.txt'
    
    tempdir = 'spice_temp/'
    circdir = 'spice_circuits/'
    outdir = 'spice_output/' + timestamp + '/'

    if os.path.exists(outdir):
        print('Putting output files in existing directory: ' + timestamp)
    else:
        os.mkdir(outdir) 

    t1 = "\n"
    t2 = "\n"

    match QConfig:
        case quantiser_word_size.w_06bit:  # 6 bit DAC
            vbpc = "3.28"
            vdd = "5.0"
            Tr = 1e-3  # the rise-time for edges, in µs
            for k in range(0, Nb):  # generate PWL strings
                k_str = str(k)
                t1 += "vdp" + k_str + " data" + k_str + " 0 pwl " + \
                    get_pwl_string(c, Ts, nsamples, k, vbpc, vdd, Tr)
                t2 += "vdn" + k_str + " datai" + k_str + " 0 pwl " + \
                    get_inverted_pwl_string(c, Ts, nsamples, k, vbpc, vdd, Tr)
            
            circf = 'cs_dac_06bit_ngspice.cir'  # circuit description
            spicef = 'cs_dac_06bit_ngspice_batch.cir'  # complete spice input file

            ctrl_str = '\n' + '.save v(outf)' + '\n' + '.tran 10u ' + str(t[-1]) + '\n'

        case quantiser_word_size.w_16bit_SPICE:  # 16 bit DAC
            vbpc = "0"
            vdd = "1.5"
            Tr = 1e-3  # the rise-time for edges, in µs
            for k in range(0, Nb):  # generate PWL strings
                k_str = str(k+1)
                t1 += "vb" + k_str + " b" + k_str + " 0 pwl " + \
                    get_pwl_string(c, Ts, nsamples, k, vbpc, vdd, Tr)
                t2 += "vbb" + k_str + " bb" + k_str + " 0 pwl " + \
                    get_inverted_pwl_string(c, Ts, nsamples, k, vbpc, vdd, Tr)

            seed_str = ''
            if seed == 1:
                seed_str = 'seed_1'
            elif seed == 2:
                seed_str = 'seed_2'

            # circuit description file
            circf = 'cs_dac_16bit_ngspice_' + seed_str + '.cir'
            # spice input file
            spicef = 'cs_dac_16bit_ngspice_batch_' + str(seq) + '.cir'

            ctrl_str = '\n' + '.save v(out)' + '\n' + '.tran 10u ' + str(t[-1]) + '\n'

    addtexttofile(tempdir + waveformfile, t1 + t2)

    addtexttofile(tempdir + 'spice_cmds.txt', ctrl_str)

    with open(outdir + spicef, 'w') as fout:
        fins = [circdir + circf, tempdir + waveformfile, tempdir + 'spice_cmds.txt']
        fin = fileinput.input(fins)
        for line in fin:
            fout.write(line)
        fin.close()

    outputf = 'output_' + str(seq)

    print(circf)
    print(spicef)
    print(outputf)
    
    return spicef, outputf


def read_spice_bin_file(path, fname):
    """
    Read a given SPICE ouput file (assuming a certain format, i.e. not general)
    """

    fid = open(path + fname, 'rb')
    # print("Opening file: " + fname)

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

    # plt.plot(t_spice, y_spice)
    # plt.show()

    return t_spice, y_spice


def read_spice_bin_file_with_most_recent_timestamp(path):
    """
    Read SPICE ouput file (assuming a certain format, i.e. not general)
    """

    binfiles = [file for file in os.listdir(path) if file.endswith('.bin')]
    binfiles.sort()
    fname = binfiles[-1]
    
    t_spice, y_spice = read_spice_bin_file(path, fname)

    return t_spice, y_spice


def main():
    """
    Testing 
    """
    path = './spice_output/'


    t_spice, y_spice = read_spice_bin_file_with_most_recent_timestamp(path)
    YM = np.zeros([1,y_spice.size])
    YM[0,:] = y_spice

    plt.plot(y_spice)

    print(YM)


if __name__ == "__main__":
    main()
