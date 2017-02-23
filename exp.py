import numpy as np
import cv2
from matplotlib import pyplot as plt

# imgL = cv2.imread('./data/frame9900.png',0)
# imgR = cv2.imread('./data/frame9903.png',0)
imgL = cv2.imread('./data/desk0.png',0)
imgR = cv2.imread('./data/desk1.png',0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()