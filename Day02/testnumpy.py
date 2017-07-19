#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import sys


M=np.array([1,1,1])
L=list(M)

M=np.random.randint(1,100,(100,100))



M_transpose=M.transpose()
print(M_transpose)
print(M)
print(sys.version_info)
