# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:37:24 2017

@author: HEWEI
"""

"""1. A = [1, 3, 2, 4; 2, 2, 3, 1; 3, 2, 4, 5; 4, 2, 0, 1], B = [1, 2, 3, 4; 2, 1, 3, 0; 4, 1, 3, 4; 2, 4,
3, 4]. Please compute the convolution between A and B by hand. They try to verify your
answer by Python code."""

import numpy as np

A = np.array([[1, 3, 2, 4], [2, 2, 3, 1],[3,2,4,5],[4,2,0,1]])
B = np.array([[1,2,3,4], [2,1,3,0],[4,1,3,4],[2,4,3,4]])

matrix = [[0 for i in range(3)] for i in range(3)]
print(']\nMatrix A:\n')
print(A)
print('\nMatrix B (before flip):\n')
print(B)

from scipy import signal
filtered=signal.convolve(A, B, 'same')
print('\nConvolution Result(the same size version):\n')
print(filtered)
