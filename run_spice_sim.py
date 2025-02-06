#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run DAC simulations via SPICE (ngspice)

Currently this script will generate the required input to ngspice
by sourcing a circuit description and adding the PWL input signal
description and a suitable control block for transient simulation.

Will produce the command line string for running ngspice,
which can be done manually. The shell escape provided by Python
prevents good debugging of ngspice execution.

@author: Arnfinn Eielsen
@date: 29.01.2025
@license: BSD 3-Clause
"""

%reload_ext autoreload
%autoreload 2

import os
import pickle
import numpy as np

from utils.test_util import sim_config
from utils.quantiser_configurations import quantiser_configurations, qs
from LM.lin_method_util import lm, dm
from utils.spice_utils import run_spice_sim, run_spice_sim_parallel, gen_spice_sim_file, read_spice_bin_file, process_sim_output
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

codes_dirs = os.listdir(method_d)

if not codes_dirs:  # list empty?
    raise SystemExit('No codes found.')

codes_d = codes_dirs[0]  ###################### pick run

# read pickled (marshalled) state/config object
with open(os.path.join(method_d, codes_d, 'sim_config.pickle'), 'rb') as fin:
    SC = pickle.load(fin)

hash_stamp = codes_d
spice_case_d = os.path.join('spice_sim', 'cases', str(SC.lin).replace(' ', '_'), hash_stamp)

if os.path.exists(spice_case_d):
    print('Putting output files in existing directory: ' + spice_case_d)
else:
    os.makedirs(spice_case_d)

spicef_list = []
outputf_list = []

# Read some config. params.

# the dac model flag appears to be superfluous, deprecated it?
#if not SC.dac.model == dm.SPICE:
#    raise SystemExit('Configuration error.')

QConfig = SC.qconfig
Nch = SC.nch
Ts = 1/SC.fs  # sampling time

# regenerate time vector
#t_end = SC.ncyc/SC.ref_freq  # time vector duration
#t = np.arange(0, t_end, Ts)  # time vector
# copy time vector
t = SC.t

Nb, Mq, Vmin, Vmax, Rng, Qstep, YQ, Qtype = quantiser_configurations(QConfig)

if QConfig in [qs.w_6bit_2ch_SPICE, qs.w_16bit_2ch_SPICE, qs.w_10bit_2ch_SPICE]:
    SEPARATE_FILE_PER_CHANNEL = False
else:
    SEPARATE_FILE_PER_CHANNEL = True

codes_fn = 'codes.npy'  # TODO: magic constant, name of codes file

if os.path.exists(os.path.join(method_d, codes_d, codes_fn)):  # codes exists
    C = np.load(os.path.join(method_d, codes_d, codes_fn))
else:
    raise SystemExit('No codes file found.')

if SEPARATE_FILE_PER_CHANNEL:
    for k in range(0,Nch):
        c = C[k,:]
        seed = k + 1
        spicef, outputf = gen_spice_sim_file(c, Nb, t, Ts, QConfig, spice_case_d, seed, k)
        spicef_list.append(spicef)
        outputf_list.append(outputf)
else:
    spicef, outputf = gen_spice_sim_file(C, Nb, t, Ts, QConfig, spice_case_d)
    spicef_list.append(spicef)  # list with 1 entry
    outputf_list.append(outputf)


spice_path = 'ngspice'  # 
#out_d = os.path.join('spice_sim', 'output')

#print('Running SPICE...')
if True:  # run ngspice sequentially for ech channel
    if SEPARATE_FILE_PER_CHANNEL:
        for k in range(0,Nch):
            run_spice_sim(spicef_list[k], outputf_list[k], spice_case_d, spice_path, run_spice=True)
    else:
        run_spice_sim(spicef_list[0], outputf_list[0], spice_case_d, spice_path, run_spice=True)
else:  # use shell escape interface to run ngspice as parallel processes
    run_spice_sim_parallel(spicef_list, outputf_list, out_d, spice_path)
    






if False:



    # Read results from a given SPICE simulation and process the data.

    outdir = 'spice_sim/output'

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




    from utils.spice_utils import run_spice_sim, run_spice_sim_parallel, gen_spice_sim_file, read_spice_bin_file, sim_config, process_sim_output, sinad_comp

    #spice_path = '/home/eielsen/ngspice_files/bin/ngspice'  # newest ver., fastest (local)
    spice_path = 'ngspice'  # 

    if False:
        for k in range(0,Nch):
            run_spice_sim(spicef_list[k], outputf_list[k], outdir, spice_path)
    else:  # use shell escape interface to run ngspice as parallel procs.
        print('Running SPICE...')
        run_spice_sim_parallel(spicef_list, outputf_list, outdir, spice_path)

    # re-sample SPICE output for uniform sampling (for FFT)
    YM = np.zeros([Nch, t.size])
    tm = t
    if SEPARATE_FILE_PER_CHANNEL:
        for k in range(0, Nch):
            t_spice, y_spice = read_spice_bin_file(outdir, outputf_list[k] + '.bin')
            y_resamp = np.interp(t, t_spice, y_spice)  # re-sample
            YM[k,:] = y_resamp
    else:
        t_spice, y_spice = read_spice_bin_file(outdir, outputf_list[0] + '.bin')
        for k in range(0, y_spice.shape[0]):
            y_resamp = np.interp(t, t_spice, y_spice[k,:])  # re-sample
            YM[k,:] = y_resamp


    if run_SPICE or SC.dac.model == dm.STATIC:
        # Summation stage TODO: Tidy up, this is case dependent
        if SC.lin.method == lm.BASELINE:
            if QConfig == qws.w_6bit_2ch_SPICE:
                K = np.ones((Nch,1))
                K[1] = 0.0  # null secondary channel (want single channel resp.)
            else:
                K = 1/Nch
        elif SC.lin.method in [lm.NSDCAL, lm.MPC, lm.MHOQ, lm.ILC]:
            if QConfig == qws.w_6bit_2ch_SPICE or QConfig == qws.w_16bit_2ch_SPICE:
                K = np.ones((2,1))
                K[1] = 0.0  # secondary channel will have zero input, null to remove any noise
            else:
                K = 1/Nch
        elif SC.lin.method == lm.DEM:
            K = np.ones((Nch,1))
        elif SC.lin.method == lm.PHYSCAL:
            K = np.ones((Nch,1))
            K[1] = get_physcal_gain(QConfig)
        else:
            K = 1/Nch

        ym = np.sum(K*YM, 0)

        # Remove transients and process the output
        TRANSOFF = np.floor(Npt*Fs/Xcs_FREQ).astype(int)  # remove transient effects from output
        # yu_avg, ENOB_U = process_sim_output(tu, yu, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, False, 'uniform')
        ym_avg, ENOB_M = process_sim_output(tm, ym, Fc_lp, Fs, N_lp, TRANSOFF, SINAD_COMP_SEL, PLOT_CURVE_FIT, 'non-linear')

        handle_results(SC, ENOB_M)
