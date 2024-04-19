#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate INL values for a SPICE file

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

from matplotlib import pyplot as plt

def addtexttofile(filename, text):
    f = open(filename, 'w')
    f.write(text)
    f.close()


def generate_dc_input(Nb, v, tempdir='spice_temp', geninputfile='input_for_spice_sim.txt'):
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
    Run operating point analysis for all bit patterns
    """
    
    circdir = 'spice_circuits'
    
    outdir = os.path.join('spice_output_dc', circname + '_' + timestamp)

    if os.path.exists(outdir):
        print('Putting output files in existing directory: ' + outdir)
    else:
        os.mkdir(outdir) 

    
    circf = circname + '.cir'  # circuit description
    spicelogf = circname + '_batch.log'  # spice log file
    spicef = circname + '_batch.cir'  # complete spice input file
    
    with open(os.path.join(outdir, spicef), 'w') as fout:
        fins = [os.path.join(circdir, circf),
                os.path.join(tempdir, geninputfile)]
        fin = fileinput.input(fins)
        for line in fin:
            fout.write(line)
        fin.close()

    
    spice_path = '/home/eielsen/ngspice_files/bin/ngspice'  # newest ver., fastest (local)
    #spice_path = 'ngspice'  #
    cmd = [spice_path, '-o', os.path.join(outdir, spicelogf),
           '-b', os.path.join(outdir, spicef)]

    print(cmd)
    
    subprocess.run(cmd)
    #return 





#Nb = 6  ## no. of bits
#circname = 'cs_dac_06bit_2ch_DC'
Nb = 16  ## no. of bits
circname = 'cs_dac_16bit_2ch_DC'

timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

outdir = os.path.join('spice_output_dc', timestamp)

if os.path.exists(outdir):
    print('Putting output files in existing directory: ' + timestamp)
else:
    os.mkdir(outdir)

for v in range(0, 2**Nb):  # generate all bit pattern permutations 
    generate_dc_input(Nb, v)
    generate_and_run_dc_spice_batch_file(timestamp, circname)


actual_1 = []
actual_2 = []

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

ML = np.array([actual_1, actual_2])

outfile = circname + '_levels'

np.save(outfile, ML)

                









