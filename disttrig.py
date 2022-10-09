import cv2
image = cv2.imread("Color_icon_red.png")
dimensions = image.shape
width = dimensions[1]
height = dimensions[0]
print("Width:  ", width)
print("Height: ", height)

hwidth = int(width/2)
for i in range(int(height/300)):
    color = image[i * 300, hwidth]
    print (color)