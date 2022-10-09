from cmath import pi
from math import tan
import cv2

image = cv2.imread("blob.jpg")
dimensions = image.shape
width = dimensions[1]
height = dimensions[0]
iAngle = 45
iDist = 31.25
iHeight = 31.25
fAngle = 60
fHeight = 32

print("Width:  ", width)
print("Height: ", height)

hwidth = int(width/2)
for i in range(height):
    yVal = height - 1 - i
    color = image[yVal, hwidth]
    #print(color)
    if color[0] == 0:
        print("Obstruction Detected:")
        print("Pixel x: ", hwidth)
        print("Pixel y: ", yVal)
        fDist = fHeight * tan((fAngle / 180) * pi)
        print(fDist)
        exit()

print("No Obstruction Detected")