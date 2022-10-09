from math import cos, pi, tan
import cv2

def degToRad(angle):
    angle = (angle / 180) * pi
    return angle

def readImage(filename):
    image = cv2.imread(filename)
    return image

def distanceToObstruction(image, coords, angle, height, fov):
    totalDist = height / tan(angle)
    contactAngle = 90 - (angle + (fov / 2))
    contactDist = height * tan(degToRad(contactAngle))
    projectedPixelDist = totalDist - contactDist
    projectionAngle = 90 + (fov / 2) + angle
    absolutePixelDist = projectedPixelDist / cos(degToRad(projectionAngle))

    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
    lowest = imageHeight
    
    for c in coords:
        