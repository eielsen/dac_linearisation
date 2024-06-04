# Comparison of various linearisation techniques for Digital-to-Analog Converters 

1. [Introduction](#introduction)
2. [Linearisation Methods](#moving-horizon-optimal-quantiser)
3. [Implementation Details](#implementation-details)
4. [Simulation](#Simulation)

## Introduction

In this repository you find an implementation of various linearisation methods to improve the accuracy of the Digital-to-Analog Converters (DACs). The results are published in the paper title "Improving the Accuracy of the Digital ot Analog Converters(DAC)s". The paper is accepted for  publication in [IMEKO 2024](https://www.imeko2024.org/home) conference. The details regarding problem formulation, simulation results and test results of the alogrithm can be found in the mentioned paper. The paper is also included in the repository.  


## Linearisation methods 
This repository contains the implementation of 7 linearisation methods which are as follows:
1. [Physical Calibration](https://pubs.aip.org/aip/rsi/article-abstract/36/7/1062/462480/Double-Precision-Bidirectional-Self-Calibrating?redirectedFrom=fulltext)
2. [Noise shaping with Digital Calibration](https://ieeexplore.ieee.org/document/4061014)
3. [Periodic High Frequency Dithering](https://ieeexplore.ieee.org/document/823976)
4. [Stochastic High Pass Dithering](https://link.springer.com/article/10.1023/A:1008850101197)
5. [Dynamic Element Matching](https://ieeexplore.ieee.org/document/5420027)
6. [Moving Horizon Optimal Quantiser](https://ieeexplore.ieee.org/document/5420027)
7. [Iterative Learning Control](https://ieeexplore.ieee.org/abstract/document/10252330) 

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
To start the simulation, go to ```main.py``` and 
    1. Choose quantiser configuration,
    2. Choose linearisation methods


