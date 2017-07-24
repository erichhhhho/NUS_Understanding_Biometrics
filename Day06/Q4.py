#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Wed Jul 19 17:37:24 2017

@author: HEWEI
"""

"""Using Python, solve Ax = b when A =[100 100; 100 100.01], and b = [2 ;2].
The Numpy functions linalg.det, linalg.svd, linalg.solve may be useful here."""

import numpy as np
from scipy.linalg import solve
a = np.array([[100,100], [100,100.01]])
b = np.array([2,2]).transpose()
c=np.array([2,2.0001]).transpose()
x1 = solve(a, b)
x2 = solve(a, c)
print(x1)
print(x2)
cond=np.linalg.cond(a)

z=np.linalg.svd(a)
det=np.linalg.det(a)


print(z)
print(det)
print(cond)