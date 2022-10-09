import cv2
import numpy as np


img = cv2.imread('ReferenceFloor.jpg')

sample = img[int(img.shape[0] * 6/7):, int(img.shape[1] * 2/7):int(img.shape[1] * 5/7)]
cv2.imshow("sample", cv2.resize(sample, [int(sample.shape[1]/4), int(sample.shape[0]/4)]))

channel_ranges = [[-1, (0, 255), (0, 255), (0, 255)],
                  [cv2.COLOR_BGR2HSV, (0, 255), (0, 255), (0, 255)],
                  [cv2.COLOR_BGR2HLS, (0, 255), (0, 255), (0, 255)],
                  [cv2.COLOR_BGR2YCrCb, (0, 255), (0, 255), (0, 255)],
                  [cv2.COLOR_BGR2YUV, (0, 255), (0, 255), (0, 255)],
                  [cv2.COLOR_BGR2Lab, (0, 255), (0, 255), (0, 255)]]

for space in channel_ranges:
    space_sample = sample.copy() if space[0] < 0 else cv2.cvtColor(sample, space[0])
    for c, chan in enumerate(space[1:]):
        channel_sample = space_sample[:, :, c]
        mean = np.mean(channel_sample)
        std = np.std(channel_sample)
        #mini = np.amin(channel_sample)
        #maxi = np.amax(channel_sample)
        space[c+1] = (mean-std*3, mean+std*3)
        #space[c + 1] = (mini, maxi)

channel_outputs = []

maskAnd = np.ones_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) * 255
maskAvg = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

for space in channel_ranges:
    space_img = img.copy() if space[0] < 0 else cv2.cvtColor(img, space[0])
    lower_ranges = np.array([x[0] for x in space[1:]])
    upper_ranges = np.array([x[1] for x in space[1:]])
    ranged_img = cv2.inRange(space_img, lower_ranges, upper_ranges)
    channel_outputs.append(ranged_img)
    cv2.imshow(str(space[0]), cv2.resize(ranged_img, [int(ranged_img.shape[1]/4), int(ranged_img.shape[0]/4)]))
    cv2.waitKey(0)
    maskAnd = cv2.bitwise_and(maskAnd, ranged_img)
    maskAvg = np.add(maskAvg, np.divide(ranged_img, len(channel_ranges)))

    #cv2.imshow("average", cv2.resize(maskAvg, [int(maskAvg.shape[1] / 4), int(maskAvg.shape[0] / 4)]))
    #cv2.waitKey(0)

cv2.imshow("and", cv2.resize(maskAnd, [int(maskAnd.shape[1]/4), int(maskAnd.shape[0]/4)]))
cv2.waitKey(0)
cv2.imshow("average", cv2.resize(maskAvg.astype("uint8"), [int(maskAvg.shape[1]/4), int(maskAvg.shape[0]/4)]))
cv2.waitKey(0)
