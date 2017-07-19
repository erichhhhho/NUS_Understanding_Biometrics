#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(100) & 0xFF ==ord('s'):
        cv2.imwrite('test.jpg',frame)
        break
# When everything done, release the capture
print(cap.get(3))
cap.release()
cv2.destroyAllWindows()