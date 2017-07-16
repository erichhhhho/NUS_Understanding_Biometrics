import cv2
import numpy as np

# import pdb
# pdb.set_trace()#turn on the pdb prompt
print(cv2.__version__);
# read image
img = cv2.imread("F:\Desktop\WeChat Image_20170708121829.jpg", cv2.IMREAD_COLOR)

cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27: # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
