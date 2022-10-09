import cv2

def detectObstacle(filename):
    currFrame = cv2.imread(filename)
    dimensions = currFrame.shape
    height = dimensions[0]
    width = dimensions[1]

    yLimit = int(height * (2 / 3))
    