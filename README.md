# Comparison of various linearisation techniques for Digital-to-Analog Converters 

1. [Introduction](#introduction)
2. [Linearisation Methods](#moving-horizon-optimal-quantiser)
3. [Implementation Details](#implementation-details)
4. [Simulation](#Simulation)

## Introduction
In this repository you find various implementations of linearisation methods that are being investigated with the aim of improving the accuracy of digital-to-analog converters (DACs). More about this project can be found on the project group's website: https://pinacl.ux.uis.no/.

The latest results have been published in the [paper](publications/Methods_for_Improving_the_Accuracy_of_Digital_to_Analog_Converters.pdf) titled "Improving the Accuracy of Digital-to-Analogue Converters (DACs). The paper was accepted for publication as an open access publication in [Measurement: Sensors](https://www.sciencedirect.com/journal/measurement-sensors). The paper was presented orally at the [Technical Committee, 4 of IMEKO](https://www.imeko.org/index.php/tc4-homepage) (IMEKO TC4 - Measurement of Electrical Quantities) at [The XXIV IMEKO World Congress 2024](https://www.imeko2024.org/home), organized by the PTB, the Physikalisch-Technische Bundesanstalt, held in Hamburg, Germany, on 26 - 29 August 2024. For details regarding the DAC modelling and algorithm implementation, please refer to the [paper](publications/Methods_for_Improving_the_Accuracy_of_Digital_to_Analog_Converters.pdf).

## Linearisation methods 
This repository contains the implementation of 7 linearisation methods (LM) which are as follows:
1. [Physical Calibration](https://pubs.aip.org/aip/rsi/article-abstract/36/7/1062/462480/Double-Precision-Bidirectional-Self-Calibrating?redirectedFrom=fulltext)
2. [Noise shaping with Digital Calibration](https://ieeexplore.ieee.org/document/4061014)
3. [Periodic High Frequency Dithering](https://ieeexplore.ieee.org/document/823976)
4. [Stochastic High Pass Dithering](https://link.springer.com/article/10.1023/A:1008850101197)
5. [Dynamic Element Matching](https://ieeexplore.ieee.org/document/5420027)
6. [Moving Horizon Optimal Quantiser](https://ieeexplore.ieee.org/document/5420027)
7. [Iterative Learning Control](https://ieeexplore.ieee.org/abstract/document/10252330) 

## Results
A summary of the current results can be found [here](results/results.md).

## Implementation Details
The implementation is based on a small set of libraries mentioned as follows
```
numpy
scipy
matplotlib
statistics
itertools
math    
gurobi
```

## Simulation
To start the simulation, go to ```run_me.py``` and 
    1. Choose quantiser configuration,
    2. Choose linearisation methods