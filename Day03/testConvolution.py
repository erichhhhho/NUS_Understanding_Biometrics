#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

A = np.array([[1, 3, 2, 4], [2, 2, 3, 1],[3,2,4,5],[4,2,0,1]])
B = np.array([[1,2,3,4], [2,1,3,0],[4,1,3,4],[2,4,3,4]])
C = np.array([[1, 2,3], [4,5,6],[7,8,9]])
D = np.array([[-1,-2,-1], [0,0,0],[1,2,1]])
matrix = [[0 for i in range(3)] for i in range(3)]
print(A)
print(B)
#print(C)
#print(D)
from scipy import signal
filtered=signal.convolve(C, D, 'full')
print(filtered)


