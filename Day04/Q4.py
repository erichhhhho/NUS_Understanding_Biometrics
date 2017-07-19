# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:34:12 2017

@author: HEWEI
"""

"""4. Make fun of Color. In this task, you will be instructed to change color for certain object
in an image. Learn Python OpenCV and HSV color space, and then understand how to
change color for object."""
import cv2


img_BGR= cv2.imread('bee.png')
img_HSV=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img_HSV, (25,0,0),(35,255,255))
cv2.imshow('mask',mask)
cv2.imwrite('Q4_Mask.jpg',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

H,S,V=cv2.split(img_HSV)

H_bg = cv2.bitwise_and(H, 255-mask)
H_roi = cv2.bitwise_and(H+150-30, mask)
H = cv2.bitwise_or(H_bg, H_roi)

mask_Merged = cv2.merge([H,S,V])
mask_BGR=cv2.cvtColor(mask_Merged,cv2.COLOR_HSV2BGR)

cv2.imshow('mask_BGR',mask_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Q4_MaskAfterMerged.jpg',mask_BGR)

"""My own pic"""
img_me_BGR= cv2.imread('me.jpg')
img_me_HSV=cv2.cvtColor(img_me_BGR,cv2.COLOR_BGR2HSV)
mask_me = cv2.inRange(img_me_HSV, (0,60,0),(150,170,170))
cv2.imshow('mask',mask_me)
cv2.imwrite('Q4_Mask_me.jpg',mask_me)

H_me,S_me,V_me=cv2.split(img_me_HSV)
H_bg_me = cv2.bitwise_and(H_me, 255-mask_me)
H_roi_me = cv2.bitwise_and(H_me +150-30, mask_me)
H_me = cv2.bitwise_or(H_bg_me, H_roi_me)

mask_Merged_me = cv2.merge([H_me,S_me,V_me])
mask_BGR_me=cv2.cvtColor(mask_Merged_me,cv2.COLOR_HSV2BGR)
cv2.imshow('mask_BGR_me',mask_BGR_me)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Q4_MaskAfterMerged_me.jpg',mask_BGR_me)
