#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import math
from math import *
os.chdir("F:/Desktop/NUS Summer/HW2/hw2")
print(os.listdir("."))

from PIL import Image
img=Image.open ('lena.png')
print(img.format,img.size,img.mode)
img.show()

from PIL import ImageEnhance
enhancer=ImageEnhance.Contrast(img)
enhanced_img=enhancer.enhance(2.0)
enhanced_img.show()

import numpy as np
img_array=np.asarray(img)
img=Image.fromarray(img_array)
img.show()

print(img_array)
print(math.pi)
