#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run INL processing

@author: Arnfinn Eielsen
@date: 28.02.2024
@license: BSD 3-Clause
"""

from INL_processing import generate_physical_level_calibration_look_up_table
from INL_processing import generate_random_output_levels

generate_physical_level_calibration_look_up_table(SAVE_LUT=1)
#generate_random_output_levels(QuantizerConfig=4)
