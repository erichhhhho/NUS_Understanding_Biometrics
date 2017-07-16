import cv2
import numpy as np

# import pdb
# pdb.set_trace()#turn on the pdb prompt
print(cv2.__version__);
# read image
img = cv2.imread("F:\Desktop\WeChat Image_20170708121829.jpg", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("Image")

cv2.imshow("Image", img)
cv2.waitKey (0)
cv2.destroyAllWindows(  )
