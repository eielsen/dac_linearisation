#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gains with some moderate pains

@author: Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

# %%
import numpy as np
from matplotlib import pyplot as plt

# %% DAC  and sampling frequency
DAC_16bit_Fs_1MHz = 1
DAC_6bit_Fs_1MHz = 2
DAC_6bit_prop_Fs_65MHz = 3
DAC_16bit_prop_Fs_65MHz = 4
DAC_16bit_Fs_32MHz = 5
DAC_6bit_Fs_32MHz = 6

# %%
match 1:
    case 1: # 16-bit 1 MHz Trond
        tech = 'SkyWater'
        node = 'SKY130'
        method0 = 'static'
        method1 = 'SPICE'

        methods = f'{method0}-{method1}'

        Nb = 16
        Fs = 1.02
        static_baseline = 11.121
        static_physcal = 16.487
        static_nsdcal = 14.416
        static_phfd = 11.283
        static_shpd = 10.750 
        static_dem = 8.645
        #static_ilc= 15.551

        # Spice simulation results
        mos_model_baseline = 11.111
        spice_baseline = mos_model_baseline
        spice_physcal = 15.189
        spice_nsdcal = 14.249
        spice_phfd = 11.173
        spice_shpd = 10.672
        spice_dem = 8.640
        #spice_ilc= 15.004

        # Calculate gains
        #static_gains = np.array([static_physcal,  static_phfd, static_shpd, static_dem, static_nsdcal,static_ilc ])- static_baseline
        #spice_gains = np.array([spice_physcal,  spice_phfd, spice_shpd, spice_dem, spice_nsdcal,spice_ilc])- spice_baseline
        
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem]) - static_baseline
        spice_gains = np.array([spice_physcal, spice_nsdcal,  spice_phfd, spice_shpd, spice_dem]) - mos_model_baseline
        mos_model_gains = spice_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM']

    case 2: # 6-bit 1 MHz Trond
        tech = 'SkyWater'
        node = 'SKY130'
        method0 = 'static'
        method1 = 'SPICE'

        methods = f'{method0}-{method1}'

        Nb = 6
        Fs = 1.02

        static_baseline = 6.051
        static_physcal = 6.832
        static_nsdcal = 8.022
        static_phfd = 6.752
        static_shpd= 3.538
        static_dem = 4.924

        # Spice simulation results
        mos_model_baseline = 6.093
        spice_baseline = mos_model_baseline
        spice_physcal = 6.164
        spice_nsdcal = 8.343
        spice_phfd = 6.783
        spice_shpd= 3.522
        spice_dem = 4.874

        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem]) - static_baseline
        spice_gains = np.array([spice_physcal, spice_nsdcal,  spice_phfd, spice_shpd, spice_dem]) - mos_model_baseline
        mos_model_gains = spice_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM']

    case 3: # SPECTRE 6 bit,
        tech = 'proprietary'
        node = '130 nm'
        method0 = 'static'
        method1 = 'Spectre'

        methods = f'{method0}-{method1}'

        Nb = 6
        Fs = 65.47
        static_baseline = 4.45
        static_physcal = 8.54
        static_nsdcal = 16.7
        static_phfd = 9.22
        static_shpd = 6.58
        static_dem = 8.06

        mos_model_baseline = 5.37
        spectre_baseline = mos_model_baseline
        spectre_physcal = 5.64
        spectre_nsdcal = 5.44
        spectre_phfd = 9.19
        spectre_shpd = 5.91
        spectre_dem = 8.14
        # spice_ilc_16_1= 14.646
        # spice_mpc_16_1= 14.646

        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem]) - static_baseline
        spectre_gains = np.array([spectre_physcal, spectre_nsdcal, spectre_phfd, spectre_shpd, spectre_dem]) - mos_model_baseline
        mos_model_gains = spectre_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM']

    case 4: # SPECTRE 16 bit,
        tech = 'proprietary'
        node = '130 nm'
        method0 = 'static'
        method1 = 'Spectre'

        methods = f'{method0}-{method1}'

        Nb = 16
        Fs = 65.47
        static_baseline = 7.00
        static_physcal = 12.61
        static_nsdcal = 18.65
        static_phfd = 9.93
        static_shpd = 7.80
        static_dem = 9.22

        mos_model_baseline = 5.98
        spectre_baseline = mos_model_baseline
        spectre_physcal = 5.86
        spectre_nsdcal = 5.90
        spectre_phfd = 4.66
        spectre_shpd = 5.22
        spectre_dem = 6.25
        # spice_ilc_16_1= 14.646
        # spice_mpc_16_1= 14.646

        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem]) - static_baseline
        spectre_gains = np.array([spectre_physcal, spectre_nsdcal, spectre_phfd, spectre_shpd, spectre_dem]) - mos_model_baseline
        mos_model_gains = spectre_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM']
    
    case 5: # 16-bit DAC 2 Ch from Trond 33 MHz
        tech = 'SkyWater'
        node = 'SKY130'
        method0 = 'static'
        method1 = 'SPICE'

        methods = f'{method0}-{method1}'

        Nb = 16
        Fs = 32.7

        static_baseline = 11.514
        static_physcal = 16.940
        static_nsdcal = 18.332
        static_phfd = 15.886
        static_shpd= 11.303
        static_dem = 11.141 
        

        # Spice simulation results
        mos_model_baseline = 11.134
        spice_baseline = mos_model_baseline
        spice_physcal = 16.913
        spice_nsdcal = 16.986
        spice_phfd = 15.109
        spice_shpd= 11.172
        spice_dem = 11.184
        
        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem]) - static_baseline
        spice_gains = np.array([spice_physcal, spice_nsdcal,  spice_phfd, spice_shpd, spice_dem]) - mos_model_baseline
        mos_model_gains = spice_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM']
    
    case 6: # 6-bit DAC 2 Ch from Trond 33 MHz
        tech = 'SkyWater'
        node = 'SKY130'
        method0 = 'static'
        method1 = 'SPICE'

        methods = f'{method0}-{method1}'

        Nb = 6
        Fs = 32.7

        static_baseline = 6.389
        static_physcal = 8.206
        static_nsdcal = 16.476
        static_phfd = 8.828
        static_shpd= 7.461
        static_dem = 7.393

        # Spice simulation results
        mos_model_baseline = 6.379
        spice_baseline = mos_model_baseline
        spice_physcal = 6.483
        spice_nsdcal = 11.21
        spice_phfd = 8.85
        spice_shpd = 7.502
        spice_dem = 7.401

        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem]) - static_baseline
        spice_gains = np.array([spice_physcal, spice_nsdcal,  spice_phfd, spice_shpd, spice_dem]) - mos_model_baseline
        mos_model_gains = spice_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM']

# %% Plots
barWidth = 0.25
# set position of the bar on X axis
bar1 = np.arange(static_gains.size)
bar2 = [x + barWidth for x in bar1] 
bar3 = [x + barWidth for x in bar2]

# % Draw plot
fig, ax = plt.subplots(figsize = (7,5))
plt.axhline(y = 0, color = 'black', linestyle = '-')
b1 = plt.bar(bar2, static_gains,    width = barWidth, color = 'tab:blue',   edgecolor = 'white', label = method0)
b2 = plt.bar(bar3, mos_model_gains, width = barWidth, color = 'tab:orange', edgecolor = 'white', label = method1)
# match methods:
#     case "static-spice":
#         b2 = plt.bar(bar3, spice_gains, width = barWidth,  color = 'tab:orange',edgecolor = 'white', label = 'SPICE')
#     case "static-spectre":
#         b2 = plt.bar(bar3, spectre_gains, width = barWidth,  color = 'tab:orange',edgecolor = 'white', label = 'Spectre')

plt.xlabel('Linearisation method', fontweight ='bold', fontsize = 15) 
plt.ylabel('ENOB gain', fontsize = 13) 

pos_xticks = np.array([r + barWidth for r in range(len(static_gains))]) + barWidth/2
plt.xticks(pos_xticks, lin_methods , fontsize = 13)

ah = []
for rect in b1 + b2 :
    height = rect.get_height()
    ah.append(height)
    if height > 0 :
        plt.text(rect.get_x() + rect.get_width()/2.0 - 0.03, 0.3, f'{height:.2f} bits', rotation = 90, fontsize  = 13)        
    if height < 0 :
        plt.text(rect.get_x() + rect.get_width()/2.0 - 0.03, 0.3, f'{height:.2f} bits', rotation = 90, fontsize  = 13)        

# Adjust location of the value Fs    
# ax.text(1, -1.2,  f'Fs = {Fs} MHz',  ha='right', va='bottom', fontsize = "20")

# for rect in b1 + b2:
#     height = rect.get_height()
#     if height > 0 :
#         plt.text(rect.get_x() + rect.get_width() / 2.0 -0.03, -1, '1.02 MHz', rotation = 90, fontsize = 13)        
#     if height < 0 :
#         plt.text(rect.get_x() + rect.get_width() / 2.0 -0.03, 0.3, '1.02 MHz', rotation = 90, fontsize = 13)

# for rect in b3 + b4:
#     height = rect.get_height()
#     if height > 0 :
#         plt.text(rect.get_x() + rect.get_width() / 2.0 -0.03, -1.25, '1 MHz', rotation = 90, fontsize = 13)        
#     if height < 0 :
#         plt.text(rect.get_x() + rect.get_width() / 2.0 -0.03, 0.5, '1 MHz', rotation = 90, fontsize = 13)        

plt.title(f"{int(Nb)}-bit DAC | Technology: {tech} {node} | Fs: {Fs} MHz\n{method0} baseline: {static_baseline} bits | {method1} baseline: {mos_model_baseline} bits", fontsize = "13")
plt.legend(fontsize="13", loc='upper right')
ax.set_axisbelow(True)
ax.grid(zorder=0, axis = "y")
fig.tight_layout()
# plt.savefig(f"Gainplot-{Nb}bits.pdf")

# %%
fname = f"figures/Gainplot_{Nb}b_{tech}_{node}_{int(Fs)}MHz_{methods}".replace(" ", "_")
fname = str(fname)
fig.savefig(fname + ".svg", format='svg', bbox_inches='tight') # Practical for PowerPoint and other applications
fig.savefig(fname + ".pdf", format='pdf', bbox_inches='tight') # Best for LaTeX