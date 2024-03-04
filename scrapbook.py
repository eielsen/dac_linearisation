#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

M = np.random.normal(0, 2.5, size=(4,10))

x = np.array([1, 2, 3, 4, 3, 2, 1, 0, 1, 1, 1], np.int32)

for k in range(0,M.shape[0]):
    print(k)
    print(M[k,x])


#M[x]

