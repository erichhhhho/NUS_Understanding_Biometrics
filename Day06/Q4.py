#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Wed Jul 19 17:37:24 2017

@author: HEWEI
"""

import numpy as np
from scipy.linalg import solve
a = np.array([[100,100], [100,100.1]])
b = np.array([2,2])
c=np.array([2,2.0001])
x1 = solve(a, b)
x2 = solve(a, c)
print(x1)
print(x2)
cond=np.linalg.cond(a,'fro')

z=np.linalg.svd(a)
det=np.linalg.det(a)


print(z)
print(det)
print(cond)