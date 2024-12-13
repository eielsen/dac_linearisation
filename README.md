# Comparison of various linearisation techniques for Digital-to-Analogue Converters 

1. [Introduction](#introduction)
2. [Linearisation Methods](#linearisation-methods)
3. [Results](#results)
4. [Dependencies](#dependencies)
5. [Simulation](#simulation)

## Introduction
In this repository you can find various implementations of linearisation methods (LM) that are being investigated with the aim of improving the accuracy of digital-to-analogue converters (DACs). The methods are first and foremost evaluated for the application to a custom integrated circuit (IC), attempting to co-optimise the performance of the linearsiation method and the IC design simultaneuously. More about this project can be found on the project group's website: https://pinacl.ux.uis.no/.

The latest results have been published in the [paper](publications/Methods_for_Improving_the_Accuracy_of_Digital_to_Analog_Converters.pdf) titled "Improving the Accuracy of Digital-to-Analogue Converters (DACs). The paper was accepted for publication as an open access publication in [Measurement: Sensors](https://www.sciencedirect.com/journal/measurement-sensors). The paper was presented at the [Technical Committee, 4 of IMEKO](https://www.imeko.org/index.php/tc4-homepage) (IMEKO TC4 - Measurement of Electrical Quantities) at [The XXIV IMEKO World Congress 2024](https://www.imeko2024.org/home), organised by the PTB, the Physikalisch-Technische Bundesanstalt, held in Hamburg, Germany, on 26 - 29 August 2024. For details regarding the modelling of the DACs and implementation of the linearisation methods (LM), please refer to the [paper](publications/Methods_for_Improving_the_Accuracy_of_Digital_to_Analog_Converters.pdf).

## Linearisation methods 
This repository contains the implementation of 7 linearisation methods (LM) which are as follows:
1. [PHYSCAL - Physical level Calibration](https://pubs.aip.org/aip/rsi/article-abstract/36/7/1062/462480/Double-Precision-Bidirectional-Self-Calibrating?redirectedFrom=fulltext)
2. [NSDCAL - Noise shaping with Digital Calibration](https://ieeexplore.ieee.org/document/100434)
3. [PHFD - Periodic High-Frequency Dithering](https://ieeexplore.ieee.org/document/823976)
4. [SHPD - Stochastic High-Pass Dithering](https://link.springer.com/article/10.1023/A:1008850101197)
5. [DEM - Dynamic Element Matching](https://ieeexplore.ieee.org/document/5420027)
6. [MHOQ - Moving Horizon Optimal Quantiser](https://www.sciencedirect.com/science/article/pii/S2405896324013946), or Model Predictive Control (MPC)
7. [ILC - Iterative Learning Control](https://ieeexplore.ieee.org/abstract/document/10252330)

## Results
A summary of the latest results can be found in this repository at [results/results.md](results/results.md).

## Dependencies
The code in this repository is dependent on the libraries listed below:
```
numpy
scipy
matplotlib
statistics
itertools
math    
```
Optimization solver
```  
gurobi
```

## Simulation
To run simulations, open ```run_me.py``` and:
1. Choose quantiser configuration
2. Choose linearisation methods
