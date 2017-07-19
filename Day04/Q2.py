#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Wed Jul 19 17:37:24 2017

@author: HEWEI
"""

import cv2

from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter

"""1.rotate a rectangular region by 45 degree counter-clockwise, whose ver-
tices are (100,100), (100,400),(400,100),(400,400)."""

"""Used PIL"""
import os
os.chdir("F:/Desktop/NUS SUMMER/HW2/hw2")
img_PIL=Image.open('lena.png')

img2_PIL = img_PIL.crop((100, 100, 400, 400))
img2_PIL=img2_PIL.rotate(45)
img_PIL.paste(img2_PIL,(100,100))
img_PIL.show()
img_PIL.save('Q2_1.jpg')

"""2.Perform histogram equalization on lena.png. Use matplotlib to plot the histogram
figure for both original image and processed image."""
"""Used Opencv"""

img_cv = cv2.imread('lena.png',0)
equ_cv = cv2.equalizeHist(img_cv)

cv2.imshow('image1',img_cv)
cv2.imshow('image2',equ_cv)

#res = np.hstack((img,equ)) #stacking images side-by-side
#cv2.imwrite('res.png',res)

plt.hist(img_cv.flatten(),256,[0,256], color = 'r');
plt.legend(('histogram_beforeE'), loc = 'upper left')
plt.show()

plt.hist(equ_cv.flatten(),256,[0,256], color = 'r');
plt.legend(('histogram_afterE'), loc = 'upper left')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

"""3.Perform Max Filtering, Min Filtering, and Median Filter on lena.png.
"""
"""Used PIL"""

img=Image.open('lena.png')

img_MaxFiltered = img.filter(ImageFilter.MaxFilter)
img_MinFiltered = img.filter(ImageFilter.MinFilter)
img_MedianFiltered = img.filter(ImageFilter.MedianFilter)

#img.show(title='origin')
#img_MaxFiltered.show(title='Max Filter')
#img_MinFiltered.show(title='Min Filter')
#img_MedianFiltered.show(title='Median Filter')

img_MaxFiltered.save('Q2_3_MaxFiltered.jpg')
img_MinFiltered.save('Q2_3_MinFiltered.jpg')
img_MedianFiltered.save('Q2_3_MedianFiltered.jpg')

"""4.Perform Gaussian Blur with sigma equal to 3 and 5.
"""
"""Used Opencv"""

img_GaussianBlur = cv2.imread('lena.png',0)
img_blurby3=cv2.GaussianBlur(img_GaussianBlur,(3,3),3,3)
img_blurby5=cv2.GaussianBlur(img_GaussianBlur,(5,5),5,5)

cv2.imshow('image1',img_GaussianBlur)
cv2.imshow('image2',img_blurby3)
cv2.imshow('image3',img_blurby5)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Q2_4_GaussianBlur_byCV3.jpg',img_blurby3)
cv2.imwrite('Q2_4_GaussianBlur_byCV5.jpg',img_blurby5)

"""Used PIL"""

img_GaussianBlurPIL = Image.open('lena.png')

img_blurby3PIL = img_GaussianBlurPIL.filter(ImageFilter.GaussianBlur(radius=3))
img_blurby5PIL = img_GaussianBlurPIL.filter(ImageFilter.GaussianBlur(radius=5))

img_blurby3PIL.save('Q2_4_GaussianBlur_byPIL3.jpg')
img_blurby5PIL.save('Q2_4_GaussianBlur_byPIL5.jpg')
