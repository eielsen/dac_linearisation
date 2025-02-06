#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read results from a given SPICE simulation and process the data.

@author: Arnfinn Eielsen
@date: 30.01.2025
@license: BSD 3-Clause
"""

%reload_ext autoreload
%autoreload 2

import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from prefixed import Float
from tabulate import tabulate

from utils.results import handle_results
from utils.spice_utils import run_spice_sim, run_spice_sim_parallel, gen_spice_sim_file, read_spice_bin_file, process_sim_output
from LM.lin_method_util import lm, dm
from utils.test_util import sim_config, sinad_comp, test_signal
from utils.inl_processing import get_physcal_gain

# choose method
METHOD_CHOICE = 6
match METHOD_CHOICE:
    case 1: RUN_LM = lm.BASELINE
    case 2: RUN_LM = lm.PHYSCAL
    case 3: RUN_LM = lm.NSDCAL
    case 4: RUN_LM = lm.PHFD
    case 5: RUN_LM = lm.SHPD
    case 6: RUN_LM = lm.DEM
    case 7: RUN_LM = lm.MPC # lm.MPC or lm.MHOQ
    case 8: RUN_LM = lm.ILC
    case 9: RUN_LM = lm.ILC_SIMP

top_d = 'generated_codes/'  # directory for generated codes and configuration info
method_d = os.path.join(top_d, str(lm(RUN_LM)))

codes_dirs = []
codes_dirs = os.listdir(method_d)

if not codes_dirs:  # list empty?
    raise SystemExit('No codes found.')

codes_d = codes_dirs[0]  # pick run

# read pickled (marshalled) state/config object
with open(os.path.join(method_d, codes_d, 'sim_config.pickle'), 'rb') as fin:
    SC = pickle.load(fin)

run_info = [['Method', 'Model', 'Fs', 'Fc', 'Fx', ],
            [str(SC.lin), str(SC.dac), f'{Float(SC.fs):.0h}', f'{Float(SC.fc):.0h}', f'{Float(SC.ref_freq):.1h}']]
print(tabulate(run_info))

hash_stamp = codes_d
spice_case_d = os.path.join('spice_sim', 'cases', str(SC.lin).replace(' ', '_'), hash_stamp)

binfiles = [file for file in os.listdir(spice_case_d) if file.endswith('.bin')]
binfiles.sort()

if not binfiles:  # list empty?
    raise SystemExit('No output found for case: {}'.format(str(lm(RUN_LM))))

Nbf = len(binfiles)  # number of bin (binary data) files

# Read some config. params.
QConfig = SC.qconfig
Nch = SC.nch
Fs = SC.fs
Ts = 1/Fs  # sampling time
Fx = SC.ref_freq

if Nbf == 1:  # may contain several channels in ngspice bin file
    print(os.path.join(spice_case_d, binfiles[0]))
    t_spice, y_spice = read_spice_bin_file(spice_case_d, binfiles[0])
    Nch = y_spice.shape[0]
    print('No. channels:')
    print(Nch)

    # Summation stage
    if SC.lin.method == lm.BASELINE:
        K = np.ones((Nch,1))
        K[1] = 0.0  # null one channel (want single channel resp.)
    elif SC.lin.method == lm.DEM:
        #K = np.ones((Nch,1))
        K = 1/Nch
    elif SC.lin.method == lm.PHYSCAL:
        K = np.ones((Nch,1))
        K[1] = get_physcal_gain(QConfig)
    elif SC.lin.method == lm.NSDCAL:
        K = np.ones((Nch,1))
        K[1] = 0.0  # null one channel (want single channel resp.)
    elif SC.lin.method == lm.ILC:
        K = np.ones((Nch,1))
        K[1] = 0.0  # null one channel (want single channel resp.)
    else:
        K = 1/Nch
        
    print('Summing gain:')
    print(K)

    t_end = t_spice[-1] #7/Fx  # time vector duration
    Fs_ = Fs*72  # semi-optimal factor for most sims with different non-uniform sampling per file
    #Fs_ = Fs
    #Fs_ = 1/np.mean(np.diff(t_spice))

    print(f'Fs: {Float(Fs):.0h}')
    t_ = np.arange(0, t_end, 1/Fs_)  # time vector

    y_spice_ = np.sum(K*y_spice, 0)
    ym_ = np.interp(t_, t_spice, y_spice_)  # re-sample


ym = ym_
t = t_
TRANSOFF = np.floor(1*Fs_/Fx).astype(int)  # remove transient effects from output

Fc = SC.fc
Nf = SC.nf

ym_avg, ENOB_M = process_sim_output(t, ym, Fc, Fs_, Nf, TRANSOFF, sinad_comp.CFIT, False, 'SPICE')

plt.plot(t[TRANSOFF:-TRANSOFF],ym[TRANSOFF:-TRANSOFF])
plt.plot(t[TRANSOFF:-TRANSOFF],ym_avg[TRANSOFF:-TRANSOFF])

SC.dac = dm(dm.SPICE)
handle_results(SC, ENOB_M)

#results_tab = [['DAC config', 'Method', 'Model', 'Fs', 'Fc', 'X scale', 'Fx', 'ENOB'],
#    [str(SC.qconfig), str(SC.lin), str(SC.dac), f'{Float(SC.fs):.2h}', f'{Float(SC.fc):.1h}', f'{Float(SC.ref_scale):.1h}%', f'{Float(SC.ref_freq):.1h}', f'{Float(ENOB_M):.3h}']]
#print(tabulate(results_tab))

if False:

    outdir = '../spice_sim/output'

    rundirs = os.listdir(outdir)
    rundirs.sort()

    print('No. dirs.: ' + str(len(rundirs)))

    method_str = 'baseline'
    #method_str = 'physical_level_calibration'
    #method_str = 'periodic_dither'
    #method_str = 'noise_dither'
    #method_str = 'digital_calibration'
    #method_str = 'dynamic_element_matching'
    #method_str = 'ilc'

    matching = [s for s in rundirs if method_str.upper() in s]

    if not matching:  # list empty?
        print("No matching simlation cases found for: {}".format(method_str))
        #return

    #rundir = rundirs[16]  # pick run
    rundir = matching[0]  # pick run

    bindir = os.path.join(outdir, rundir)

    # read pickled (marshalled) state/config object
    with open(os.path.join(bindir, 'sim_config.pickle'), 'rb') as fin:
        SC = pickle.load(fin)

    print(bindir)

    run_info = [['Method', 'Model', 'Fs', 'Fc', 'Fx'],
                [str(SC.lin), str(SC.dac), f'{Float(SC.fs):.0h}', f'{Float(SC.fc):.0h}', f'{Float(SC.carrier_freq):.1h}']]

    print(tabulate(run_info))

    binfiles = [file for file in os.listdir(bindir) if file.endswith('.bin')]
    binfiles.sort()

    if not binfiles:  # list empty?
        print("No output found for case: {}".format(method_str))
        #return

    if True:
        Nbf = len(binfiles)  # number of bin (binary data) files

        t = SC.t
        Fs = SC.fs
        Fx = SC.carrier_freq

        t_end = 3/Fx  # time vector duration
        Fs_ = Fs*72  # semi-optimal factor for most sims with different non-uniform sampling per file
        Fs_ = Fs
        print(f'Fs: {Float(Fs):.0h}')
        t_ = np.arange(0, t_end, 1/Fs_)  # time vector
        
        if Nbf == 1:  # may contain several channels in ngspice bin file
            print(os.path.join(bindir, binfiles[0]))
            t_spice, y_spice = read_spice_bin_file(bindir, binfiles[0])
            Nch = y_spice.shape[0]
            print('No. channels:')
            print(Nch)

            # Summation stage
            if SC.lin.method == lm.BASELINE or SC.lin.method == lm.ILC:
                K = np.ones((Nch,1))
                K[1] = 0.0  # null one channel (want single channel resp.)
            elif SC.lin.method == lm.DEM:
                K = np.ones((Nch,1))
            elif SC.lin.method == lm.PHYSCAL:
                K = np.ones((Nch,1))
                K[1] = 1e-2
            else:
                K = 1/Nch
            
            print('Summing gain:')
            print(K)

            y_spice_ = np.sum(K*y_spice, 0)
            ym_ = np.interp(t_, t_spice, y_spice_)  # re-sample

        else:  # assume one channel per bin file
            Nch = Nbf
            YM = np.zeros([Nch, t_.size])
            for k in range(0, Nbf):
                print(os.path.join(bindir, binfiles[k]))
                t_spice, y_spice = read_spice_bin_file(bindir, binfiles[k])
                y_resamp = np.interp(t_, t_spice, y_spice)  # re-sample
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

        ym_avg, ENOB_M = process_sim_output(t, ym, Fc, Fs_, Nf, TRANSOFF, sinad_comp.CFIT, False, 'SPICE')

        plt.plot(t,ym)
        plt.plot(t,ym_avg)
        
        results_tab = [['Config', 'Method', 'Model', 'Fs', 'Fc', 'Fx', 'ENOB'],
            [str(SC.qconfig), str(SC.lin), str(SC.dac), f'{Float(SC.fs):.2h}', f'{Float(SC.fc):.1h}', f'{Float(SC.carrier_freq):.1h}', f'{Float(ENOB_M):.3h}']]
        print(tabulate(results_tab))
        
        #t_spice, y_spice = read_spice_bin_file_with_most_recent_timestamp(path)
        #YM = np.zeros([1,y_spice.size])
        #YM[0,:] = y_spice
        #plt.plot(y_spice)
        #print(YM)