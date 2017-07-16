#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("F:\Desktop\WeChat Image_20170708121829.jpg", cv2.IMREAD_COLOR)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()