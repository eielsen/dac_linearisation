#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File wrangling for SPICE simulations.

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
from scipy import interpolate
import pickle
from prefixed import Float

from lin_method_util import lm, dm
from figures_of_merit import FFT_SINAD, TS_SINAD
from quantiser_configurations import quantiser_word_size


class sinad_comp:
    FFT = 1  # FFT based
    CFIT = 2  # curve fit


class sim_config:
    def __init__(self, lin, dac, fs, t, fc, nf, carrier_scale, carrier_freq):
        self.lin = lin
        self.dac = dac
        self.fs = fs
        self.t = t
        self.fc = fc
        self.nf = nf
        self.carrier_scale = carrier_scale
        self.carrier_freq = carrier_freq
    
    def __str__(self):
        s = str(self.lin) + '\n'
        s = s + str(self.dac) + '\n'
        s = s + 'Fs=' + f'{Float(self.fs):.0h}' + '\n'
        s = s + 'Fc=' + f'{Float(self.fc):.0h}' + '\n'
        s = s + 'Nf=' + f'{Float(self.nf):.0h}' + '\n'
        s = s + 'Xs=' + f'{Float(self.carrier_scale):.0h}' + '\n'
        s = s + 'Fx=' + f'{Float(self.carrier_freq):.0h}' + '\n'

        return s + '\n'


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


def run_spice_sim(spicef, outputf, outdir='spice_output/', spice_path='ngspice'):
    """
    Run SPICE simulaton using provided filenames

    Arguments
        spicef - SPICE batch file
        outputf - Output files name
    """
    
    print(spicef)
    print(outputf)

    cmd = [spice_path, '-o', outdir + outputf + '.log',
                    # '-r', outdir + outputf + '.bin',
                    '-b', outdir + spicef]

    print(cmd)

    subprocess.run(cmd)


def run_spice_sim_parallel(spicef_list, outputf_list, outdir='spice_output/', spice_path='ngspice'):
    """
    Run SPICE simulaton using provided filenames

    Arguments
        spicef_list - SPICE batch files
        outputf_list - Output files names
    """
    
    cmd_list = []
    for k in range(0, len(spicef_list)):
        cmd = [spice_path, '-o', outdir + outputf_list[k] + '.log',
            #'-r', outdir + outputf_list[k] + '.bin',
            '-b', outdir + spicef_list[k]]
        print(cmd)
        cmd_list.append(cmd)

    procs_list = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in cmd_list]
    
    for proc in procs_list:
        print('Waiting for SPICE to return...')
        proc.wait()


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
    
    tempdir = 'spice_temp'
    circdir = 'spice_circuits'
    
    outdir = os.path.join('spice_output', timestamp)

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

            outputf = 'cs_dac_16bit_ngspice_batch_' + str(seq)
            
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

            # ctrl_str = '\n' + '.save v(out)' + '\n' + '.tran 10u ' + str(t[-1]) + '\n'

            outputf = 'cs_dac_16bit_ngspice_batch_' + str(seq)
            
            ctrl_str = ''

            if seed == 1:
                #ctrl_str = '\n.option method=trap XMU=0.495 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=1\n'
                ctrl_str = '\n.option method=trap TRTOL=5 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=1\n'
            elif seed == 2:
                #ctrl_str = '\n.option method=trap XMU=0.495 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=2\n'
                ctrl_str = '\n.option method=trap TRTOL=5 gmin=1e-19 reltol=200u abstol=100f vntol=100n seed=2\n'
            
            ctrl_str = ctrl_str + \
                '\n.control\n' + \
                'tran 10u ' + str(t[-1]) + '\n' + \
                'write $inputdir/' + outputf + '.bin' + ' v(out)\n' + \
                '.endc\n'

    addtexttofile(os.path.join(tempdir, 'spice_cmds.txt'), ctrl_str)

    addtexttofile(os.path.join(tempdir, waveformfile), t1 + t2)

    with open(os.path.join(outdir, spicef), 'w') as fout:
        fins = [os.path.join(circdir, circf),
                os.path.join(tempdir, 'spice_cmds.txt'),
                os.path.join(tempdir, waveformfile)]
        fin = fileinput.input(fins)
        for line in fin:
            fout.write(line)
        fin.close()

    print(circf)
    print(spicef)
    print(outputf)
    
    return spicef, outputf


def read_spice_bin_file(fdir, fname):
    """
    Read a given SPICE ouput file (assuming a certain format, i.e. not general)
    """

    fpath = os.path.join(fdir, fname)
    fid = open(fpath, 'rb')
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


def read_spice_bin_file_with_most_recent_timestamp(fdir):
    """
    Read SPICE ouput file (assuming a certain format, i.e. not general)
    """

    binfiles = [file for file in os.listdir(fdir) if file.endswith('.bin')]
    binfiles.sort()
    fname = binfiles[-1]
    
    t_spice, y_spice = read_spice_bin_file(fdir, fname)

    return t_spice, y_spice


def process_sim_output(t, y, Fc, Nf, TRANSOFF, SINAD_COMP_SEL, plot=False, descr=''):
    # Filter the output using a reconstruction (output) filter
    Wc = 2*np.pi*Fc
    b, a = signal.butter(Nf, Wc, 'lowpass', analog=True)  # filter coefficients
    Wlp = signal.lti(b, a)  # filter LTI system instance

    y = y.reshape(-1, 1)  # ensure the vector is a column vector
    y_avg_out = signal.lsim(Wlp, y, t, X0=None, interp=False) # filter the ideal output
    y_avg = y_avg_out[1] # extract the filtered data; lsim returns (T, y, x) tuple, want output y

    match SINAD_COMP_SEL:
        case sinad_comp.FFT:  # use FFT based method to detemine SINAD
            R = FFT_SINAD(y_avg[TRANSOFF:-TRANSOFF], Fs, plot, descr)
        case sinad_comp.CFIT:  # use time-series sine fitting based method to detemine SINAD
            R = TS_SINAD(y_avg[TRANSOFF:-TRANSOFF], t[TRANSOFF:-TRANSOFF], plot, descr)

    ENOB = (R - 1.76)/6.02

    # Print FOM
    print(descr + ' SINAD: {}'.format(R))
    print(descr + ' ENOB: {}'.format(ENOB))

    return y_avg, ENOB

def main():
    """
    Read results 
    """
    outdir = 'spice_output'

    rundirs = os.listdir(outdir)
    rundirs.sort()

    print('No. dirs.: ' + str(len(rundirs)))
    rundir = rundirs[7]  # pick run

    path = os.path.join(outdir, rundir)

    binfiles = [file for file in os.listdir(path) if file.endswith('.bin')]
    binfiles.sort()

    if True:
        with open(os.path.join(path, 'sim_config.pickle'), 'rb') as fin:
            SC = pickle.load(fin)
        
        print(SC.lin)

        Nch = len(binfiles)  # one file per channel

        t = SC.t
        Fs = SC.fs
        Fx = SC.carrier_freq

        t_end = 3/Fx  # time vector duration
        Fs_ = Fs*72  # 
        print(f'Fs: {Float(Fs):.0h}')
        t_ = np.arange(0, t_end, 1/Fs_)  # time vector

        YM = np.zeros([Nch, t_.size])

        for k in range(0,Nch):
            print(os.path.join(path, binfiles[k]))
            t_spice, y_spice = read_spice_bin_file(path, binfiles[k])
            min_t = np.min(np.diff(t_spice))
            print(f'Fs_max: {Float(1/min_t):.0h}')
            match 1:
                case 1:
                    y_resamp = np.interp(t_, t_spice, y_spice)  # re-sample
                case 2:
                    #y_resamp = interpolate.CubicSpline(t_spice, y_spice)(t_)
                    y_resamp = interpolate.Akima1DInterpolator(t_spice, y_spice)(t_)
                    #y_resamp = interpolate.PchipInterpolator(t_spice, y_spice)(t_)
            YM[k,:] = y_resamp

        # Summation stage
        if SC.lin.method == lm.DEM:
            K = np.ones((Nch,1))
        if SC.lin.method == lm.PHYSCAL:
            K = np.ones((Nch,1))
            K[1] = 1e-2
        else:
            K = 1/Nch

        ym_ = np.sum(K*YM, 0)

        if False:
            #ym = np.interp(t, t_, ym_)  # re-sample
            #ym = interpolate.Akima1DInterpolator(t_, ym_)(t)
            ym = interpolate.PchipInterpolator(t_, ym_)(t)
            #ym = signal.resample(ym_, t.size)
            TRANSOFF = np.floor(0.5*Fs/Fx).astype(int)  # remove transient effects from output
        else:
            ym = ym_
            t = t_
            TRANSOFF = np.floor(0.25*Fs_/Fx).astype(int)  # remove transient effects from output

        Fc = SC.fc
        Nf = SC.nf

        ym_avg, ENOB_M = process_sim_output(t, ym, Fc, Nf, TRANSOFF, sinad_comp.CFIT, False, 'SPICE')

        #plt.plot(t,ym)
        #plt.plot(t,ym_avg)

        from tabulate import tabulate

        results = [['Method', 'Model', 'Fs', 'Fc', 'Fx', 'ENOB'],
                [str(SC.lin), str(SC.dac), f'{Float(SC.fs):.0h}', f'{Float(SC.fc):.0h}', f'{Float(SC.carrier_freq):.1h}', f'{Float(ENOB_M):.3h}']]

        print(tabulate(results))

        #t_spice, y_spice = read_spice_bin_file_with_most_recent_timestamp(path)
        #YM = np.zeros([1,y_spice.size])
        #YM[0,:] = y_spice
        #plt.plot(y_spice)
        #print(YM)


if __name__ == "__main__":
    main()
