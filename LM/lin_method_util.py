#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utility functions

@author: Arnfinn Aas Eielsen
@date: 04.04.2024
@license: BSD 3-Clause
"""

class lm:  # linearisation method
    BASELINE = 1  # baseline
    PHYSCAL = 2  # physical level calibration
    DEM = 3  # dynamic element matching
    NSDCAL = 4  # noise shaping with digital calibration (INL model)
    SHPD = 5  # stochastic high-pass noise dither
    PHFD = 6  # periodic high-frequency dither
    MPC = 7  # model predictive control (with INL model)
    ILC = 8  # iterative learning control (with INL model, periodic signals)
    ILC_SIMP = 9  # iterative learning control, basic implementation

    def __init__(self, method):
        self.method = method

    def __str__(self):
        match self.method:
            case lm.BASELINE:
                return 'baseline'
            case lm.PHYSCAL:
                return 'physical level calibration'
            case lm.DEM:
                return 'dynamic element matching'
            case lm.NSDCAL:
                return 'digital calibration'
            case lm.SHPD:
                return 'noise dither'
            case lm.PHFD:
                return 'periodic dither'
            case lm.MPC:
                return 'mpc'
            case lm.ILC:
                return 'ilc'
            case lm.ILC_SIMP:
                return 'ilc simple'
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
