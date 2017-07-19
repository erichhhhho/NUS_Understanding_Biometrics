#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
os.chdir("F:/Desktop/NUS SUMMER/HW2/hw2")
print(os.listdir("."))
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter

"""1.rotate a rectangular region by 45 degree counter-clockwise, whose ver-
tices are (100,100), (100,400),(400,100),(400,400)."""
img_PIL=Image.open('lena.png')

img2_PIL = img_PIL.crop((100, 100, 400, 400))
img2_PIL=img2_PIL.rotate(45)
img_PIL.paste(img2_PIL,(100,100))
#img.show()
#img.show()

"""2.Perform histogram equalization on lena.png. Use matplotlib to plot the histogram
figure for both original image and processed image."""
img_cv = cv2.imread('lena.png',0)
equ_cv = cv2.equalizeHist(img_cv)

#cv2.imshow('image1',img_cv)
#cv2.imshow('image2',equ_cv)

#res = np.hstack((img,equ)) #stacking images side-by-side
#cv2.imwrite('res.png',res)

plt.hist(img_cv.flatten(),256,[0,256], color = 'r');
plt.legend(('histogram_beforeE'), loc = 'upper left')
#plt.show()

plt.hist(equ_cv.flatten(),256,[0,256], color = 'r');
plt.legend(('histogram_afterE'), loc = 'upper left')
#plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

"""3.Perform Max Filtering, Min Filtering, and Median Filter on lena.png.
"""
img=Image.open('lena.png')
img_MaxFiltered = img.filter(ImageFilter.MaxFilter)
img_MinFiltered = img.filter(ImageFilter.MinFilter)
img_MedianFiltered = img.filter(ImageFilter.MedianFilter)

img_MaxFiltered.save('img_MaxFiltered.png')
#img.show(title='origin')
#img_MaxFiltered.show(title='Max Filter')
#img_MinFiltered.show(title='Min Filter')
#img_MedianFiltered.show(title='Median Filter')

"""4.Perform Gaussian Blur with sigma equal to 3 and 5.
"""
img_GaussianBlur = cv2.imread('lena.png',0)
img_blurby3=cv2.GaussianBlur(img_GaussianBlur,(3,3),3,3)
img_blurby5=cv2.GaussianBlur(img_GaussianBlur,(5,5),5,5)
cv2.imshow('image1',img_GaussianBlur)
cv2.imshow('image2',img_blurby3)
cv2.imshow('image3',img_blurby5)
cv2.waitKey(0)
cv2.destroyAllWindows()