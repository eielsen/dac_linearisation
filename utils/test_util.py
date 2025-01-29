#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for DAC test stimuli

@author: Arnfinn Eielsen
@date: 29.01.2025
@license: BSD 3-Clause
"""

import numpy as np


def test_signal(SCALE, MAXAMP, FREQ, OFFSET, t):
    """
    Generate a test signal (reference)
    
    Arguments
        SCALE - percentage of maximum amplitude
        MAXAMP - maximum amplitude
        FREQ - signal frequency in hertz
        OFFSET - signal offset
        t - time vector
    
    Returns
        x - sinusoidal test signal
    """
    return (SCALE/100)*MAXAMP*np.cos(2*np.pi*FREQ*t) + OFFSET
