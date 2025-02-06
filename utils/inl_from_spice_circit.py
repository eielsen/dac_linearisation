#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate INL values for a DAC design in a given SPICE file.

This is specific for the current steering DAC topologies used in this project,
and requires suitably prepared SPICE netlists.

@author: Arnfinn Eielsen, Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
import os
import fileinput
import datetime
import subprocess
import csv
from scipy import signal

from matplotlib import pyplot as plt


def addtexttofile(filename, text):
    f = open(filename, 'w')
    f.write(text)
    f.close()


def generate_dc_input(Nb, v, tempdir='spice_temp', geninputfile='input_for_spice_sim.txt'):
    """
    Generate all the required inputs for the SPICE simulation.
    Defining voltage sources for the input pattern with the required voltage levels
    as well as the inverse pattern. 
    """
    
    bstr = np.base_repr(v, base=2, padding=Nb)
    bstri = ''.join('1' if x == '0' else '0' for x in bstr)
    bstr_len = len(bstr)
    idx = bstr_len - Nb

    tvb1 = ""
    tvb2 = ""
    tvbb1 = ""
    tvbb2 = ""

    bstr_ = bstr[idx:]
    bstri_ = bstri[idx:]

    for k, b in enumerate(bstr_[::-1]):  # input pattern for given DAC
        tvb1_str = "vb1" + str(k+1) + " b1" + str(k+1) + " 0 dc " + str(int(b)*1.5)
        tvb1 += tvb1_str + "\n"
        tvb2_str = "vb2" + str(k+1) + " b2" + str(k+1) + " 0 dc " + str(int(b)*1.5)
        tvb2 += tvb2_str + "\n"
    for k, b in enumerate(bstri_[::-1]):  # inverse pattern
        tvbb1_str = "vbb1" + str(k+1) + " bb1" + str(k+1) + " 0 dc " + str(int(b)*1.5)
        tvbb1 += tvbb1_str + "\n"
        tvbb2_str = "vbb2" + str(k+1) + " bb2" + str(k+1) + " 0 dc " + str(int(b)*1.5)
        tvbb2 += tvbb2_str + "\n"

    addtexttofile(os.path.join(tempdir, geninputfile), tvb1 + tvbb1 + tvb2 + tvbb2)


def generate_and_run_dc_spice_batch_file(timestamp, circname, tempdir='spice_temp', geninputfile='input_for_spice_sim.txt'):
    """
    Run operating point analysis for a given bit pattern input stored in a file.
    """
    
    circdir = '../spice_sim/circuits'  # location of the SPICE circuit files
    
    outdir = os.path.join('spice_output_dc', circname + '_' + timestamp)  # desintation

    if os.path.exists(outdir):
        print('Putting output files in existing directory: ' + outdir)
    else:
        os.mkdir(outdir) 

    circf = circname + '.cir'  # circuit description
    spicelogf = circname + '_batch.log'  # SPICE log file
    spicef = circname + '_batch.cir'  # complete SPICE input file
    
    # joining the circuit description file and the input file
    with open(os.path.join(outdir, spicef), 'w') as fout:
        fins = [os.path.join(circdir, circf),
                os.path.join(tempdir, geninputfile)]
        fin = fileinput.input(fins)
        for line in fin:
            fout.write(line)
        fin.close()
    
    #spice_path = '/home/eielsen/ngspice_files/bin/ngspice'  # newest ver., fastest (local)
    spice_path = 'ngspice'  # assume PATH is set to ngspice
    cmd = [spice_path, '-o', os.path.join(outdir, spicelogf),
           '-b', os.path.join(outdir, spicef)]  # generate CLI string to run SPICE

    print(cmd)
    
    subprocess.run(cmd)  # run SPICE, hope for the best


def read_spice_output_and_save_to_npy(circname, timestamp):
    actual_1 = []
    actual_2 = []
    
    outdir = os.path.join('spice_output_dc', circname + '_' + timestamp)
    lvlsfile = circname + '_levels.txt'

    with open(os.path.join(outdir, lvlsfile), newline='') as in_file:
        lvlsreader = csv.reader(in_file, delimiter=' ', skipinitialspace=True)
        for row in lvlsreader:
            actual_1.append(float(row[1]))
            actual_2.append(float(row[3]))


    actual_1 = np.array(actual_1)
    actual_2 = np.array(actual_2)

    plt.plot(actual_1)
    plt.plot(actual_2)
    plt.show()

    plt.plot(signal.detrend(actual_1))
    plt.plot(signal.detrend(actual_2))

    ML = np.array([actual_1, actual_2])

    outfile = os.path.join('../measurements_and_data', circname + '_levels')

    np.save(outfile, ML)


def run_dc_analysis_for_all_bit_pattern_combinations(Nb, circname, timestamp):
    """
    Generate all bit patterns for Nb bits (all codes between 0 and 2^Nb-1)
    and run DC analysis for each bit pattern;
    this yields the static response for each code, i.e. the INL
    """

    for v in range(0, 2**Nb):  # generate all bit pattern combinations 
        generate_dc_input(Nb, v)
        generate_and_run_dc_spice_batch_file(timestamp, circname)


# %%

match 2:
    case 1:
        Nb = 6  ## no. of bits
        circname = 'cs_dac_06bit_2ch_DC'
    case 2:
        Nb = 10  ## no. of bits
        circname = 'cs_dac_10bit_2ch_DC'
    case 3:
        Nb = 16  ## no. of bits
        circname = 'cs_dac_16bit_2ch_DC'

if 0:
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dc_analysis_for_all_bit_pattern_combinations(Nb, circname, timestamp)

if 1:
    timestamp = '20250126T131543'
    timestamp = '20250126T141847'
    timestamp = '20250206T144059'
    read_spice_output_and_save_to_npy(circname, timestamp)
