import cv2
import numpy as np


def make_mask(in_img):

    sample = in_img[int(in_img.shape[0] * 6/7):, int(in_img.shape[1] * 2/7):int(in_img.shape[1] * 5/7)]
    #cv2.imshow("sample", cv2.resize(sample, [int(sample.shape[1]/4), int(sample.shape[0]/4)]))

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

    #maskAnd = np.ones_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) * 255
    maskAvg = np.zeros_like(cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY))

    for space in channel_ranges:
        space_img = in_img.copy() if space[0] < 0 else cv2.cvtColor(in_img, space[0])
        lower_ranges = np.array([x[0] for x in space[1:]])
        upper_ranges = np.array([x[1] for x in space[1:]])
        ranged_img = cv2.inRange(space_img, lower_ranges, upper_ranges)
        channel_outputs.append(ranged_img)

        #cv2.imshow(str(space[0]), cv2.resize(ranged_img, [int(ranged_img.shape[1]/4), int(ranged_img.shape[0]/4)]))
        #cv2.waitKey(0)

        #maskAnd = cv2.bitwise_and(maskAnd, ranged_img)
        maskAvg = np.add(maskAvg, np.divide(ranged_img, len(channel_ranges)))

    #cv2.imshow("and", cv2.resize(maskAnd, [int(maskAnd.shape[1]/4), int(maskAnd.shape[0]/4)]))
    #cv2.waitKey(0)
    maskAvg = maskAvg.astype("uint8")
    maskAvg = cv2.threshold(maskAvg, 200, 255, 0)[1]

    kernel = np.ones((int(in_img.shape[1]/500), int(in_img.shape[1]/500)))
    maskAvg = maskAvg

    maskAvg = cv2.erode(maskAvg, kernel)
    maskAvg = cv2.dilate(maskAvg, kernel)

    maskAvg = cv2.blur(maskAvg, (int(in_img.shape[1]/250), int(in_img.shape[1]/250)))

    maskAvg = cv2.threshold(maskAvg, 125, 255, 0)[1]

    cont, hier = cv2.findContours(maskAvg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont2 = []
    for contour in cont:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = cv2.contourArea(contour)
            if cy > in_img.shape[0] / 3 and area > in_img.shape[0] * in_img.shape[1] * 1/1000:
                cont2.append(contour)

    filledMask = cv2.drawContours(np.zeros_like(maskAvg), cont2, -1, 255, -1)

    return filledMask, cont2, hier


def find_hazards(filledMask, contourList, hierarchy):
    outMask = cv2.cvtColor(filledMask, cv2.COLOR_GRAY2BGR).astype("uint8")
    blankimg = np.zeros(outMask.shape[:-1])
    for c, cnt in enumerate(contourList):
        hull = cv2.convexHull(cnt)
        blankimg = cv2.drawContours(blankimg, [hull], 0, 255, -1)
        blankimg = cv2.drawContours(blankimg, [cnt], 0, 0, -1)

    kernel = np.ones((int(outMask.shape[1] / 250), int(outMask.shape[1] / 250)))
    blankimg = cv2.erode(blankimg, kernel)
    blankimg = cv2.dilate(blankimg, kernel)

    hazards, hier = cv2.findContours(blankimg.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #blankimg = cv2.drawContours(np.zeros_like(blankimg), hazards, -1, 255, 2)

    #cv2.imshow("holes", cv2.resize(blankimg.astype("uint8"), [int(blankimg.shape[1]/4), int(blankimg.shape[0]/4)]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    lowests = [h[np.argmax(h[:, :, 1])][0] for h in hazards]

    return outMask, lowests


if __name__ == "__main__":
    img = cv2.imread('DIST4.jpg')
    mask, cnt, hiera = make_mask(img)
    omask, lows = find_hazards(mask, cnt, hiera)
    for l in lows:
        cv2.circle(omask, l, 5, (255,0,255), 5)
    cv2.imshow("average", cv2.resize(omask.astype("uint8"), [int(omask.shape[1]/4), int(omask.shape[0]/4)]))
    cv2.waitKey(0)
