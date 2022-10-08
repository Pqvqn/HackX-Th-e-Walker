import cv2
import numpy as np

img = cv2.imread('ReferenceFloor.jpg')

camPts = np.float32([[10, 3024], [1552, 1411], [2480, 1411], [4022, 3024]])
flatPts = np.float32([[0, 3024], [0, 0], [4032, 0], [4032, 3024]])

mat = cv2.getPerspectiveTransform(camPts, flatPts)
wimg = cv2.warpPerspective(img, mat, (1008*4, 756*4))

wimg = cv2.resize(wimg, (1008, 756))

cv2.imshow('Adrian', wimg)
cv2.waitKey(0)
cv2.imwrite('ReferenceWarp.jpg', wimg)
cv2.destroyAllWindows()