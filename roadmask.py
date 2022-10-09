import cv2
import numpy as np


img = cv2.imread('ReferenceFloor.jpg')
img = cv2.resize(img, [int(img.shape[1]/4), int(img.shape[0]/4)])

channel_ranges = [[cv2.COLOR_BGR2GRAY, (0, 255)],
                  [-1, (0, 255), (0, 255), (0, 255)],
                  [cv2.COLOR_BGR2HSV, (0, 255), (0, 255), (0, 255)]]
channel_outputs = []

for space in channel_ranges:
    space_img = img if space[0] < 0 else cv2.cvtColor(img, space[0])
    if space[0] == cv2.COLOR_BGR2GRAY:
        print(np.amin(space_img), np.amax(space_img))
    lower_ranges = space[1][0] if len(space) == 2 else np.array([x[0] for x in space[1:]])
    upper_ranges = space[1][1] if len(space) == 2 else np.array([x[1] for x in space[1:]])
    channel_outputs.append(cv2.inRange(img, lower_ranges, upper_ranges))
    cv2.imshow(str(space[0]), channel_outputs[-1])
    cv2.waitKey(0)
