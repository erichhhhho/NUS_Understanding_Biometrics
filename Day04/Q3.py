# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:38:57 2017

@author: HEWEI
"""

"""3. Color space conversion. Use Python OpenCV functions to perform following operations
on 'bee.png' and save the images."""



import cv2

img_BGR= cv2.imread('bee.png')
img_HSV=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)

h,s,v=cv2.split(img_HSV)
v_eq=cv2.equalizeHist(v)

img_HSV2 = cv2.merge([h,s,v_eq])
img_BGR2=cv2.cvtColor(img_HSV2,cv2.COLOR_HSV2BGR)

cv2.imshow('BGR Before HSV Histogram Equalization',img_BGR)
cv2.imshow('BGR After HSV Histogram Equalization',img_BGR2)
cv2.imwrite('Q3.jpg',img_BGR2)

cv2.waitKey(0)
cv2.destroyAllWindows()
