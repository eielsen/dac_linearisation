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
DAC_6bit_SKY_Fs_102MHz = 1
DAC_6bit_SKY_Fs_327MHz = 2
DAC_10bit_SKY_Fs_102MHz = 3
DAC_10bit_SKY_Fs_327MHz = 4
DAC_6bit_PRO_Fs_209MHz = 5
DAC_10bit_PRO_Fs_209MHz = 6

# %%
match 6:
    case 1: # 6 bit SkyWater at 1.02 Mhz 
        tech = 'SkyWater'
        # node = 'SKY130'
        method0 = 'static'
        method1 = 'spice'

        methods = f'{method0}-{method1}'

        Nb = 6
        Fs = 1.02
        static_baseline = 11.121
        static_physcal = 16.487
        static_nsdcal = 14.416
        static_phfd = 11.283
        static_shpd = 10.750 
        static_dem = 8.645
        static_mhoq= 15.551

        # Spice simulation results
        mos_model_baseline = 11.111
        spice_baseline = mos_model_baseline
        spice_physcal = 15.189
        spice_nsdcal = 14.249
        spice_phfd = 11.173
        spice_shpd = 10.672
        spice_dem = 8.640
        spice_mhoq= 15.004

        
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem, static_mhoq]) - static_baseline
        spice_gains = np.array([spice_physcal, spice_nsdcal,  spice_phfd, spice_shpd, spice_dem, spice_mhoq]) - mos_model_baseline
        mos_model_gains = spice_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM','MHOQ']

    case 2: # 6-bit SkyWater at 32.07  
        tech = 'SkyWater'
        # node = 'SKY130'
        method0 = 'static'
        method1 = 'spice'

        methods = f'{method0}-{method1}'

        Nb = 6
        Fs = 32.07

        static_baseline = 6.051
        static_physcal = 6.832
        static_nsdcal = 8.022
        static_phfd = 6.752
        static_shpd= 3.538
        static_dem = 4.924  
        static_mhoq= 15.551

        # Spice simulation results
        mos_model_baseline = 6.093
        spice_baseline = mos_model_baseline
        spice_physcal = 6.164
        spice_nsdcal = 8.343
        spice_phfd = 6.783
        spice_shpd= 3.522
        spice_dem = 4.874
        spice_mhoq = 1
        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem, static_mhoq]) - static_baseline
        spice_gains = np.array([spice_physcal, spice_nsdcal,  spice_phfd, spice_shpd, spice_dem, spice_mhoq]) - mos_model_baseline
        mos_model_gains = spice_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM', 'MHOQ']

    case 3: # 10 bit SkyWater at 1.02 Mhz 
        tech = 'SkyWater'
        # node = '130 nm'
        method0 = 'static'
        method1 = 'spice'

        methods = f'{method0}-{method1}'

        Nb = 10
        Fs = 1.02
        static_baseline = 4.45
        static_physcal = 8.54
        static_nsdcal = 16.7
        static_phfd = 9.22
        static_shpd = 6.58
        static_dem = 8.06
        static_mhoq= 15.551


        # Spice simulation results
        mos_model_baseline = 6.093
        spice_baseline = mos_model_baseline
        spice_physcal = 6.164
        spice_nsdcal = 8.343
        spice_phfd = 6.783
        spice_shpd= 3.522
        spice_dem = 4.874
        spice_mhoq = 1

        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem, static_mhoq]) - static_baseline
        spice_gains = np.array([spice_physcal, spice_nsdcal,  spice_phfd, spice_shpd, spice_dem, spice_mhoq]) - mos_model_baseline
        mos_model_gains = spice_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM', 'MHOQ']

    case 4: #10 bit SkyWater at 32.07 MHz,
        tech = 'SkyWater'
        # node = '130 nm'
        method0 = 'static'
        method1 = 'spice'

        methods = f'{method0}-{method1}'

        Nb = 10
        Fs = 32.07

        static_baseline = 7.00
        static_physcal = 12.61
        static_nsdcal = 18.65
        static_phfd = 9.93
        static_shpd = 7.80
        static_dem = 9.22
        static_mhoq= 15.551

        # Spice simulation results
        mos_model_baseline = 6.093
        spice_baseline = mos_model_baseline
        spice_physcal = 6.164
        spice_nsdcal = 8.343
        spice_phfd = 6.783
        spice_shpd= 3.522
        spice_dem = 4.874
        spice_mhoq = 1
        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem, static_mhoq]) - static_baseline
        spice_gains = np.array([spice_physcal, spice_nsdcal,  spice_phfd, spice_shpd, spice_dem, spice_mhoq]) - mos_model_baseline
        mos_model_gains = spice_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM', 'MHOQ']
    
    case 5: ## 6 bit ZTC ARTI
        tech = 'ZTCARTI'
        # node = 'SKY130'
        method0 = 'static'
        method1 = 'spectre'

        methods = f'{method0}_{method1}'

        Nb = 6
        Fs = 209.72 

        static_baseline = 11.00
        static_physcal = 10.93
        static_nsdcal = 11.06
        static_phfd = 10.40
        static_shpd= 10.82
        static_dem = 10.92
        static_mhoq= 12.44
        

        # Spectre simulation results
        mos_model_baseline = 10.22
        spectre_baseline = mos_model_baseline
        spectre_physcal = 9.90
        spectre_nsdcal = 9.80
        spectre_phfd = 10.36
        spectre_shpd = 9.96
        spectre_dem = 9.69
        spectre_mhoq = 9.67
        
        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem, static_mhoq]) - static_baseline
        spectre_gains = np.array([spectre_physcal, spectre_nsdcal, spectre_phfd, spectre_shpd, spectre_dem, spectre_mhoq]) - mos_model_baseline
        mos_model_gains = spectre_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM','MHOQ']
    
    case 6: # 10 bit ZTC ARTI
        tech = 'ZTCARTI '
        # node = 'SKY130'
        method0 = 'static'
        method1 = 'sprectre'

        methods = f'{method0}-{method1}'

        Nb = 10
        Fs = 209.72 

        static_baseline = 10.99 
        static_physcal = 14.96 
        static_nsdcal = 18.78 
        static_phfd = 14.35 
        static_shpd= 11.66 
        static_dem = 12.32 
        static_mhoq= 9.99 


        # Spectre simulation results
        mos_model_baseline = 8.92
        spectre_baseline = mos_model_baseline
        spectre_physcal = 8.66
        spectre_nsdcal = 8.50
        spectre_phfd = 9.43
        spectre_shpd = 8.92
        spectre_dem = 8.05
        spectre_mhoq = 9.23
        

        # Calculate gains
        static_gains = np.array([static_physcal, static_nsdcal, static_phfd, static_shpd, static_dem, static_mhoq]) - static_baseline
        spectre_gains = np.array([spectre_physcal, spectre_nsdcal, spectre_phfd, spectre_shpd, spectre_dem, spectre_mhoq]) - mos_model_baseline
        mos_model_gains = spectre_gains

        lin_methods =  ['PHYSCAL', 'NSDCAL', 'PHFD', 'SHPD', 'DEM','MHOQ']

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

plt.title(f"{int(Nb)}-bit DAC | Technology: {tech} | Fs: {Fs} MHz", fontsize = "13")

# plt.title(f"{int(Nb)}-bit DAC | Technology: {tech} {node} | Fs: {Fs} MHz\n{method0} baseline: {static_baseline} bits | {method1} baseline: {mos_model_baseline} bits", fontsize = "13")
plt.legend(fontsize="13", loc='upper right')
ax.set_axisbelow(True)
ax.grid(zorder=0, axis = "y")
fig.tight_layout()
# plt.savefig(f"Gainplot-{Nb}bits.pdf")

# %%
# fname = f"figures/Gainplot_{Nb}b_{tech}_{node}_{int(Fs)}MHz_{methods}".replace(" ", "_")
# fname = str(fname)
# fig.savefig(fname + ".svg", format='svg', bbox_inches='tight') # Practical for PowerPoint and other applications
# fig.savefig(fname + ".pdf", format='pdf', bbox_inches='tight') # Best for LaTeX

# %%

# fname = f"Gainplot_{Nb}b_{tech}_{node}_{int(Fs)}MHz_{methods}".replace(" ", "_")
fname = f"Gainplot_{Nb}b_{tech}_{int(Fs)}MHz_{methods}".replace(" ", "_")
fname = str(fname)

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Figures/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
fig.savefig(results_dir + fname + ".svg", format ='svg', bbox_inches ='tight')
fig.savefig(results_dir + fname + ".pdf", format ='pdf', bbox_inches ='tight')

# fig.savefig(fname + ".svg", format='svg', bbox_inches='tight') # Practical for PowerPoint and other applications
# fig.savefig(fname + ".pdf", format='pdf', bbox_inches='tight') # Best for LaTeX
# %%
