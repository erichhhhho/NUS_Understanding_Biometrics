#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import cv2

cap=cv2.VideoCapture("F:\Desktop\DDD.avi")
while(cap.isOpened()):
    ret,frame=cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)

    if cv2.waitKey(100) & 0xFF ==ord('q'):
        break
    elif cv2.waitKey(100) & 0xFF ==ord('s'):
        cv2.imwrite('test.jpg',frame)


cap.release()
cv2.destroyAllWindows()
