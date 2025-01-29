#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run DAC simulations via SPICE (ngspice)

@author: Arnfinn Eielsen
@date: 29.01.2025
@license: BSD 3-Clause
"""

import os

# choose method
method_str = 'baseline'
#method_str = 'physical_level_calibration'
#method_str = 'periodic_dither'
#method_str = 'noise_dither'
#method_str = 'digital_calibration'
#method_str = 'dynamic_element_matching'
#method_str = 'ilc'


top_d = 'generated_codes/'  # directory for generated codes and configuration info
method_d = top_d + method_str.upper().replace(" ", "_") + '/'

out_dirs = os.listdir(method_d)

if not out_dirs:  # list empty?
    print('No codes found...')
    exit(-1)

out_dir = out_dirs[0]  # pick run

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
