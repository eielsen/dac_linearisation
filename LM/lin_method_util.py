#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utility functions

@author: Arnfinn Aas Eielsen
@date: 04.04.2024
@license: BSD 3-Clause
"""

class lm:  # linearisation method
    BASELINE = 1  # baseline
    PHYSCAL = 2  # Physical level Calibration
    DEM = 3  # Dynamic Element Matching
    NSDCAL = 4  # Noise shaping with Digital Calibration (INL model)
    SHPD = 5  # Stochastic High-Pass Dithering
    PHFD = 6  # Periodic High-Frequency Dithering
    MPC = 7  # Model Predictive Control (with INL model)
    MHOQ = 7  # Moving Horizon Optimal Quantiser (The same as MPC)
    ILC = 8  # iterative learning control (with INL model, periodic signals)
    ILC_SIMP = 9  # iterative learning control, basic implementation

    def __init__(self, method):
        self.method = method

    def __str__(self):
        match self.method:
            case lm.BASELINE:
                return 'BASELINE'
            case lm.PHYSCAL:
                return 'PHYSCAL'
            case lm.DEM:
                return 'DEM'
            case lm.NSDCAL:
                return 'NSDCAL'
            case lm.SHPD:
                return 'SHPD'
            case lm.PHFD:
                return 'PHFD'
            case lm.MPC | lm.MHOQ:
                return 'MHOQ / MPC'
            case lm.ILC:
                return 'ILC'
            case lm.ILC_SIMP:
                return 'ILC simple'
            case _:
                return '-'
            


class dm:  # DAC model
    STATIC = 1  # static model
    SPICE = 2  # spice model

    def __init__(self, model):
        self.model = model

    def __str__(self):
        match self.model:
            case dm.STATIC:
                return 'static'
            case dm.SPICE:
                return 'spice'
            case _:
                return '-'


def main():
    """
    Test
    """
    lmethod = lm(lm.BASELINE)

    

if __name__ == "__main__":
    main()
