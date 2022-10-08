import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2 as cv
import numpy as np
import copy
# import PySpin
import time
from skimage import feature, exposure


class Window(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.setWindowTitle("Barrel2")

        self.update_speed = 100

        control_layout = QVBoxLayout()
        add_button = QPushButton("Add Filter")
        add_button.clicked.connect(self.add_filter)
        control_layout.addWidget(add_button)
        remove_button = QPushButton("Remove Filter")
        remove_button.clicked.connect(self.remove_filter)
        control_layout.addWidget(remove_button)
        clear_button = QPushButton("Clear Filters")
        clear_button.clicked.connect(self.clear_filters)
        control_layout.addWidget(clear_button)

        filter_boxes = []
        filter_boxes.append(QGroupBox("Shape"))
        filter_boxes.append(QGroupBox("Segment"))
        filter_boxes.append(QGroupBox("Morph"))
        filter_boxes.append(QGroupBox("Color"))
        filter_boxes.append(QGroupBox("Blur"))
        filter_boxes.append(QGroupBox("Edge"))
        filter_boxes.append(QGroupBox("Gradient"))
        filter_boxes.append(QGroupBox("Marking"))
        filter_boxes.append(QGroupBox("Composite"))
        filter_box_layouts = {}
        for fbox in filter_boxes:
            grid = QGridLayout()
            filter_box_layouts[fbox.title()] = grid
            fbox.setLayout(grid)
            control_layout.addWidget(fbox)
        self.radio_button = None

        def convertColor(color):
            return (int(color / (256 ** 2)), int((color % (256 ** 2)) / (256)), int(color % (256)))

        def crop(img, top, bot, lef, rig):
            w = img.shape[1]
            h = img.shape[0]
            t = min(int((top / 100) * h), h - 1)
            b = max(int(h - (bot / 100) * h), t + 1)
            l = min(int((lef / 100) * w), w - 1)
            r = max(int(w - (rig / 100) * w), l + 1)
            return img[t:b, l:r]

        def contours(img, color=0, thickness=3, index=-1):
            img2 = img.copy()
            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.threshold(img, 125, 255, 0)[1]
            cont = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
            if index >= len(cont):
                index = -1
            img = cv.drawContours(img2, cont, index, convertColor(color), thickness)
            return img

        def blobs(img, color, thresh_l, thresh_h, thresh_s, area_l, area_h, circle_l, circle_h, convex_l, convex_h,
                  inertia_l, inertia_h, dist_l, repeat_l):
            img2 = img.copy()
            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = cv.threshold(img, 125, 255, 0)[1]

            params = cv.SimpleBlobDetector_Params()
            params.minThreshold = thresh_l
            params.maxThreshold = thresh_h
            params.thresholdStep = thresh_s
            params.filterByArea = area_h >= area_l >= 0
            params.minArea = area_l
            params.maxArea = area_h
            params.filterByCircularity = circle_h >= circle_l >= 0
            params.minCircularity = circle_l / 100
            params.maxCircularity = circle_h / 100
            params.filterByConvexity = convex_h >= convex_l >= 0
            params.minConvexity = convex_l / 100
            params.maxConvexity = convex_h / 100
            params.filterByInertia = inertia_h >= inertia_l >= 0
            params.minInertiaRatio = inertia_l / 100
            params.maxInertiaRatio = inertia_h / 100
            params.minDistBetweenBlobs = dist_l
            params.minRepeatability = repeat_l
            params.blobColor = color
            detector = cv.SimpleBlobDetector_create(params)
            kpoints = detector.detect(img)

            img = cv.drawKeypoints(img2, kpoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return img

        def mask(img, pos_index, neg_index, mask_index):
            this_index = 0
            for i in range(self.filter_layout.count()):
                filter = self.filter_layout.itemAt(i).widget()
                if np.all(filter.image == img):
                    this_index = i
            if pos_index > this_index:
                pos_index = this_index
            if neg_index > this_index:
                neg_index = this_index
            if mask_index > this_index:
                mask_index = this_index
            # bw_not, and/or, order of img
            pos_img = self.get_filter(pos_index).image
            neg_img = self.get_filter(neg_index).image
            mask_img = self.get_filter(mask_index).image

            pos_img = cv.resize(pos_img, (mask_img.shape[1], mask_img.shape[0]))
            neg_img = cv.resize(neg_img, (mask_img.shape[1], mask_img.shape[0]))

            pos_img = cv.bitwise_or(pos_img, pos_img, mask=cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY))
            neg_img = cv.bitwise_or(neg_img, neg_img, mask=cv.bitwise_not(cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY)))

            return cv.bitwise_or(pos_img, neg_img)

        def watershed(img):
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv.dilate(opening, kernel, iterations=3)
            dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
            ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv.subtract(sure_bg, sure_fg)
            ret, markers = cv.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv.watershed(img, markers)
            img[markers == -1] = [255, 0, 0]
            return img

        def houghline(img, color, thickness, threshold, minlen, maxgap):
            img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            lines = cv.HoughLinesP(img2, 1, np.pi / 180, threshold, minlen, maxgap)
            if lines is None:
                return img
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if color < 0:
                    col = tuple(np.random.choice(range(256), size=3))
                    col = (int(col[0]), int(col[1]), int(col[2]))
                else:
                    col = convertColor(color)
                cv.line(img, (x1, y1), (x2, y2), col, thickness)
            return img

        def houghcirc(img, color, thickness, dp, mindist, p1, p2, minrad, maxrad):
            img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            circs = cv.HoughCircles(img2, cv.HOUGH_GRADIENT, dp=dp / 10, minDist=mindist,
                                    param1=p1, param2=p2, minRadius=minrad, maxRadius=maxrad)
            if circs is None:
                return img

            for (x, y, r) in circs[0]:
                if color < 0:
                    col = tuple(np.random.choice(range(256), size=3))
                    col = (int(col[0]), int(col[1]), int(col[2]))
                else:
                    col = convertColor(color)
                cv.circle(img, (int(x), int(y)), int(r), col, thickness)
            return img

        def houghcircauto(img, minsize, maxsize, circnum, guessaccthresh):
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            circles = None
            minimum_circle_size = minsize
            maximum_circle_size = maxsize
            guess_dp = 1.0
            number_of_circles_expected = circnum
            breakout = False
            max_guess_accumulator_array_threshold = guessaccthresh
            circleLog = []
            paramsLog = []
            guess_accumulator_array_threshold = max_guess_accumulator_array_threshold

            while guess_accumulator_array_threshold > 1 and breakout == False:
                guess_dp = 1.0
                # print("resetting guess_dp:" + str(guess_dp))
                while guess_dp < 9 and breakout == False:
                    guess_radius = maximum_circle_size
                    # print("setting guess_radius: " + str(guess_radius))
                    # print(circles is None)
                    while True:
                        # print("guessing radius: " + str(guess_radius) +
                        # " and dp: " + str(guess_dp) + " vote threshold: " +
                        # str(guess_accumulator_array_threshold))
                        circles = cv.HoughCircles(gray,
                                                  cv.HOUGH_GRADIENT,
                                                  dp=guess_dp,  # resolution of accumulator array.
                                                  minDist=100,
                                                  # number of pixels center of circles should be from each other, hardcode
                                                  param1=50,
                                                  param2=guess_accumulator_array_threshold,
                                                  minRadius=(guess_radius - 3),
                                                  # HoughCircles will look for circles at minimum this size
                                                  maxRadius=(guess_radius + 3)
                                                  # HoughCircles will look for circles at maximum this size
                                                  )

                        if circles is not None:
                            if len(circles[0]) == number_of_circles_expected:
                                # print("len of circles: " + str(len(circles)))
                                circleLog.append(copy.copy(circles))
                                paramsLog.append((guess_dp * 10, 100, 50, guess_accumulator_array_threshold,
                                                  guess_radius - 3, guess_radius + 3))
                                # print("k1")
                            break
                            circles = None
                        guess_radius -= 5
                        if guess_radius < 40:
                            break;
                    guess_dp += 1.5
                guess_accumulator_array_threshold -= 2

            for param in paramsLog:
                print(param)
            for cir in circleLog:
                # print(cir[0, :])
                cir = np.round(cir[0, :]).astype("int")
                for (x, y, r) in cir:
                    cv.circle(img, (x, y), r, (0, 0, 255), 2)
                    cv.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            return img

        def channelmult(img, space, *args):
            cvtinto = convertspaces(img, True, space)
            channels = cv.split(cvtinto)
            for i, c in enumerate([*args]):
                channels[i] = channels[i] * (c / 255)
            merged = cv.merge(channels).astype('uint8')
            cvtout = convertspaces(merged, False, space)
            return cvtout

        def channeladd(img, space, *args):
            cvtinto = convertspaces(img, True, space)
            channels = cv.split(cvtinto)
            for i, c in enumerate([*args]):
                channels[i] = cv.add(channels[i], c);
            merged = cv.merge(channels).astype('uint8')
            cvtout = convertspaces(merged, False, space)
            return cvtout

        def channelisolate(img, space, *args):
            cvtinto = convertspaces(img, True, space)
            ranges = [*args]
            mask = cv.inRange(cvtinto, np.array([ranges[0], ranges[2], ranges[4]]),
                              np.array([ranges[1], ranges[3], ranges[5]]))
            res = cv.bitwise_and(cvtinto, cvtinto, mask=mask)
            cvtout = convertspaces(res, False, space)
            return cvtout

        # BGR = -1, HLS = 0, HSV = 1, Lab = 2, Luv = 3, YUV = 4
        spaceconversions = [[cv.COLOR_HLS2BGR, cv.COLOR_HSV2BGR, cv.COLOR_Lab2BGR, cv.COLOR_Luv2BGR, cv.COLOR_YUV2BGR],
                            [cv.COLOR_BGR2HLS, cv.COLOR_BGR2HSV, cv.COLOR_BGR2Lab, cv.COLOR_BGR2Luv, cv.COLOR_BGR2YUV]]

        def convertspaces(img, isbgr, cvttype):
            if cvttype < 0:
                return img
            return cv.cvtColor(img, spaceconversions[isbgr][cvttype])

        def equalizecolored(img):
            imgyuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
            imgyuv[:, :, 0] = cv.equalizeHist(imgyuv[:, :, 0])
            return cv.cvtColor(imgyuv, cv.COLOR_YUV2BGR)

        def hog(img, orients, pixels, blocks):
            image = img
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            (H, hogImage) = feature.hog(gray, orientations=orients, pixels_per_cell=pixels,
                                        cells_per_block=blocks, transform_sqrt=True, block_norm="L1", visualize=True)
            hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
            hogImage = hogImage.astype("uint8")
            hogImage = cv.cvtColor(hogImage, cv.COLOR_GRAY2BGR)
            return hogImage

        def lbp(img, pts, rad):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(img, pts, rad, method="uniform")
            img = lbp.astype("uint8")
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            return img

        def lbp_hist(img, pts, rad, dx):
            outimg = np.zeros_like(img)
            for i in range(0, img.shape[1], dx):
                imgslice = img[:, i:i + dx, :]
                grayslice = cv.cvtColor(imgslice, cv.COLOR_BGR2GRAY)
                lbp = feature.local_binary_pattern(grayslice, pts, rad, method="uniform")
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, pts + 3), range=(0, pts + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                chunkh = img.shape[0] / len(hist)
                print(hist)
                for y, h in enumerate(hist):
                    value = h * 255
                    print((i, int(chunkh * y)), (i + dx, int(chunkh * (y + 1))), value)
                    cv.rectangle(outimg, (i, int(chunkh * y)), (i + dx, int(chunkh * (y + 1))), (value, value, value),
                                 -1)
            return outimg

        def ridges(img, direc, sigma):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            helems = feature.hessian_matrix(img, sigma=sigma, order='rc')
            i1, i2 = feature.hessian_matrix_eigvals(helems)
            if direc < 0:
                img2 = i2
            else:
                img2 = i1
            img2 = exposure.rescale_intensity(img2, out_range=(0, 255))
            img2 = img2.astype("uint8")
            return cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

        filters = {
            "Grayscale": {
                "function": lambda img: cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR),
                "name": "Grayscale",
                "ranges": [],
                "group": "Color"
            },

            "Threshold": {
                "function": lambda *args: cv.threshold(*args)[1],
                "name": "Threshold",
                "ranges": [("Thresh", "i", 0, 255, (1, 0), 125), ("Max", "i", 0, 510, (1, 0), 255),
                           ("Type", "i", 0, 4, (1, 0), 0)],
                "group": "Segment"
            },

            "ThresholdA": {

                "function": lambda img, met, reg, c: cv.cvtColor(
                    cv.adaptiveThreshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 255, met, cv.THRESH_BINARY_INV, reg, c),
                    cv.COLOR_GRAY2BGR),
                "name": "Adaptive Threshold",
                "ranges": [("Method", "i", 0, 1, (1, 0), 0), ("Region", "i", 3, 999, (2, 1), 21),
                           ("C", "i", -255, 255, (1, 0), 0)],
                "group": "Segment"
            },

            "Resize": {
                "function": lambda img, mult: cv.resize(img, tuple(
                    int(mult * x / 100) for f, x in enumerate(reversed(img.shape)) if len(img.shape) - f <= 2)),
                "name": "Resize",
                "ranges": [("Percent", "i", 1, 1000, (1, 0), 100)],
                "group": "Shape"
            },

            "Gaussian": {
                "function": cv.GaussianBlur,
                "name": "Gaussian Blur",
                "ranges": [("K Size", "t", 3, 21, (2, 1), 11), ("Step", "i", 1, 30, (1, 0), 10),
                           ("Border", "borderType", 0, 4, (1, 0), 1)],
                "group": "Blur"
            },

            "Median": {
                "function": cv.medianBlur,
                "name": "Median Blur",
                "ranges": [("Size", "i", 1, 50, (2, 1), 5)],
                "group": "Blur"
            },

            "Bilateral": {
                "function": cv.bilateralFilter,
                "name": "Bilateral Filter",
                "ranges": [("Diameter", "i", 1, 50, (1, 0), 5), ("Color", "i", 1, 200, (1, 0), 75),
                           ("Space", "i", 1, 200, (1, 0), 75)],
                "group": "Blur"
            },

            "Canny": {
                "function": lambda *args: cv.cvtColor(cv.Canny(*args), cv.COLOR_GRAY2BGR),
                "name": "Canny Edge Detect",
                "ranges": [("Min", "i", 1, 1000, (1, 0), 0), ("Max", "i", 1, 1000, (1, 0), 400)],
                "group": "Edge"
            },

            "Laplace": {
                "function": lambda img, kernel: cv.Laplacian(img, cv.CV_8U, ksize=kernel),
                "name": "Laplacian Operator",
                "ranges": [("K Size", "i", 1, 21, (2, 1), 1)],  # ("Border", "borderType", 0, 4, (1, 0), 1)]
                "group": "Edge"
            },

            "Sobel": {
                "function": lambda img, dx, dy, kernel: cv.Sobel(img, cv.CV_8U, dx, dy, ksize=kernel),
                "name": "Sobel Operator",
                "ranges": [("dx", "i", 0, 2, (1, 0), 1), ("dy", "i", 0, 2, (1, 0), 0),
                           ("K Size", "i", 1, 7, (2, 1), 3)],
                "group": "Edge"
            },

            "Erode": {
                "function": cv.erode,
                "name": "Erode",
                "ranges": [("K Size", "o", 1, 21, (2, 1), 3)],
                "group": "Morph"
            },

            "Dilate": {
                "function": cv.dilate,
                "name": "Dilate",
                "ranges": [("K Size", "o", 1, 21, (2, 1), 3)],
                "group": "Morph"
            },

            "Gradient": {
                "function": lambda img, ks: cv.morphologyEx(img, cv.MORPH_GRADIENT, ks),
                "name": "Morphological Gradient",
                "ranges": [("K Size", "o", 1, 21, (2, 1), 3)],
                "group": "Morph"
            },

            "TopH": {
                "function": lambda img, ks: cv.morphologyEx(img, cv.MORPH_TOPHAT, ks),
                "name": "Top Hat",
                "ranges": [("K Size", "o", 1, 21, (2, 1), 3)],
                "group": "Morph"
            },

            "BlackH": {
                "function": lambda img, ks: cv.morphologyEx(img, cv.MORPH_BLACKHAT, ks),
                "name": "Black Hat",
                "ranges": [("K Size", "o", 1, 21, (2, 1), 3)],
                "group": "Morph"
            },

            "Contours": {
                "function": contours,
                "name": "Draw Contours",
                "ranges": [("Color", "color", 0, 16777215, (1, 0), 65280), ("Width", "thickness", -1, 50, (1, 0), 3),
                           ("Index", "index", -1, 100, (1, 0), -1)],
                "group": "Marking"
            },

            "Blob": {
                "function": blobs,
                "name": "Simple Blob Detector",
                "ranges": [("Color", "i", 0, 255, (1, 0), 0),
                           ("Thresh L", "i", 0, 255, (1, 0), 50), ("Thresh H", "i", 0, 255, (1, 0), 220),
                           ("Thresh S", "i", 0, 255, (1, 0), 10),
                           ("Area L", "i", -1, 500000, (1, 0), -1), ("Area H", "i", -1, 500000, (1, 0), -1),
                           ("Circle L", "i", -1, 100, (1, 0), -1), ("Circle H", "i", -1, 100, (1, 0), -1),
                           ("Convex L", "i", -1, 100, (1, 0), -1), ("Convex H", "i", -1, 100, (1, 0), -1),
                           ("Inertia L", "i", -1, 100, (1, 0), -1), ("Inertia H", "i", -1, 100, (1, 0), -1),
                           ("Dist L", "i", 0, 1000, (1, 0), 10), ("Repeat H", "i", 1, 10, (1, 0), 2)],
                "group": "Marking"
            },

            "Watershed": {
                "function": watershed,
                "name": "Watershed Algorithm",
                "ranges": [],
                "group": "Marking"
            },

            "Mask": {
                "function": mask,
                "name": "Mask Outputs",
                "ranges": [("+ Img", "i", 0, 100, (1, 0), 100), ("- Img", "i", 0, 100, (1, 0), 0),
                           ("Mask", "i", 0, 100, (1, 0), 100)],
                "group": "Composite"
            },

            "Invert": {
                "function": cv.bitwise_not,
                "name": "Invert Color",
                "ranges": [],
                "group": "Color"
            },

            "HoughL": {
                "function": houghline,
                "name": "Hough Line Transform",
                "ranges": [("Color", "i", -1, 16777215, (1, 0), -1), ("Width", "i", 1, 50, (1, 0), 3),
                           ("Thresh", "i", 0, 100, (1, 0), 50), ("MinLen", "i", 0, 1000, (1, 0), 50),
                           ("MaxGap", "i", 0, 1000, (1, 0), 100)],
                "group": "Marking"
            },

            "HoughC": {
                "function": houghcirc,
                "name": "Hough Circle Transform",
                "ranges": [("Color", "i", -1, 16777215, (1, 0), -1), ("Width", "i", 1, 50, (1, 0), 3),
                           ("Res", "i", 1, 100, (1, 0), 10), ("Dist", "i", 0, 500, (1, 0), 20),
                           ("P1", "i", 0, 100, (1, 0), 50), ("P2", "i", 0, 100, (1, 0), 30),
                           ("MinRad", "i", 0, 500, (1, 0), 0), ("MaxRad", "i", 0, 500, (1, 0), 0)],
                "group": "Marking"
            },

            "AutoHoughC": {
                "function": houghcircauto,
                "name": "Auto Hough Circle",
                "ranges": [("Min", "i", 0, 1000, (1, 0), 100), ("Max", "i", 0, 1000, (1, 0), 150),
                           ("Num", "i", 1, 3, (1, 0), 1), ("AccThresh", "i", 1, 300, (1, 0), 100)],
                "group": "Marking"
            },

            "Equalize": {
                "function": lambda img: cv.cvtColor(cv.equalizeHist(cv.cvtColor(img, cv.COLOR_BGR2GRAY)),
                                                    cv.COLOR_GRAY2BGR),
                "name": "Equalize Histogram",
                "ranges": [],
                "group": "Color"
            },

            "EqualizeA": {
                "function": lambda img: cv.cvtColor(
                    cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv.cvtColor(img, cv.COLOR_BGR2GRAY)),
                    cv.COLOR_GRAY2BGR),
                "name": "Adaptive Equilization",
                "ranges": [],
                "group": "Color"
            },

            "EqualizeC": {
                "function": equalizecolored,
                "name": "Colorized Equalization",
                "ranges": [],
                "group": "Color"

            },

            "Channel+": {
                "function": channeladd,
                "name": "Adjust Channels Add",
                "ranges": [("Space", "i", -1, 4, (1, 0), -1), ("c1", "i", -255, 255, (1, 0), 0),
                           ("c2", "i", -255, 255, (1, 0), 0), ("c3", "i", -255, 255, (1, 0), 0)],
                "group": "Color"
            },

            "Channel*": {
                "function": channelmult,
                "name": "Adjust Channels Multiply",
                "ranges": [("Space", "i", -1, 4, (1, 0), -1), ("c1", "i", 0, 255, (1, 0), 255),
                           ("c2", "i", 0, 255, (1, 0), 255), ("c3", "i", 0, 255, (1, 0), 255)],
                "group": "Color"
            },

            "Isolate": {
                "function": channelisolate,
                "name": "Isolate Color Ranges",
                "ranges": [("Space", "i", -1, 4, (1, 0), -1), ("c1Min", "i", 0, 255, (1, 0), 0),
                           ("c1Max", "i", 0, 255, (1, 0), 255), ("c2Min", "i", 0, 255, (1, 0), 0),
                           ("c2Max", "i", 0, 255, (1, 0), 255), ("c3Min", "i", 0, 255, (1, 0), 0),
                           ("c3Max", "i", 0, 255, (1, 0), 255)],
                "group": "Segment"
            },

            "Crop": {
                "function": crop,
                "name": "Crop",
                "ranges": [("Top", "i", 0, 100, (1, 0), 0), ("Bottom", "i", 0, 100, (1, 0), 0),
                           ("Left", "i", 0, 100, (1, 0), 0), ("Right", "i", 0, 100, (1, 0), 0)],
                "group": "Shape"
            },

            "Transpose": {
                "function": cv.transpose,
                "name": "Transpose",
                "ranges": [],
                "group": "Shape"
            },

            "Pad": {
                "function": lambda img, t, b, l, r: cv.copyMakeBorder(img, int(img.shape[1] * t / 100),
                                                                      int(img.shape[1] * b / 100),
                                                                      int(img.shape[0] * l / 100),
                                                                      int(img.shape[0] * r / 100),
                                                                      cv.BORDER_CONSTANT),
                "name": "Pad",
                "ranges": [("Top", "i", 0, 100, (1, 0), 0), ("Bottom", "i", 0, 100, (1, 0), 0),
                           ("Left", "i", 0, 100, (1, 0), 0), ("Right", "i", 0, 100, (1, 0), 0)],
                "group": "Shape"
            },

            "Flip": {
                "function": cv.flip,
                "name": "Flip",
                "ranges": [("Direction", "i", -1, 1, (1, 0), -1)],
                "group": "Shape"
            },

            "Hog": {
                "function": hog,
                "name": "Histogram of Oriented Gradients",
                "ranges": [("Orientations", "i", 1, 16, (1, 0), 9), ("CellSz", "t", 1, 20, (1, 0), 10),
                           ("BlockSz", "t", 1, 20, (1, 0), 2)],
                "group": "Gradient"
            },

            "Lbp": {
                "function": lbp,
                "name": "Local Binary Patterns",
                "ranges": [("Points", "i", 1, 128, (1, 0), 24), ("Radius", "i", 1, 32, (1, 0), 8)],
                "group": "Gradient"
            },

            "LbpHist": {
                "function": lbp_hist,
                "name": "Local Binary Patterns Histogram",
                "ranges": [("Points", "i", 1, 128, (1, 0), 24), ("Radius", "i", 1, 32, (1, 0), 8),
                           ("dx", "i", 5, 100, (5, 0), 30)],
                "group": "Gradient"
            },

            "Ridge": {
                "function": ridges,
                "name": "Detect Ridges",
                "ranges": [("Dir", "i", -1, 1, (2, 1), 1), ("Sigma", "i", 1, 5, (1, 0), 3)],
                "group": "Edge"
            }
        }

        for f, key in enumerate(filters):
            radio_button = QRadioButton(key)
            radio_button.filter_name = key
            radio_button.filter_settings = filters[key]
            radio_button.toggled.connect(self.select_filter)
            layout = filter_box_layouts[filters[key]["group"]];
            k = layout.count()
            layout.addWidget(radio_button, int(k / 2), k % 2)
        self.selected_filter = filters["Grayscale"]

        print_button = QPushButton("Print Sequence")
        print_button.clicked.connect(self.print_sequence)
        control_layout.addWidget(print_button)
        save_img_button = QPushButton("Save Image")
        save_img_button.clicked.connect(self.save_image)
        control_layout.addWidget(save_img_button)

        container_widget = QGroupBox()
        self.filter_layout = QVBoxLayout()
        container_widget.setLayout(self.filter_layout)
        self.filter_layout.setAlignment(Qt.AlignTop)
        container_scroll = QScrollArea()
        container_scroll.setWidget(container_widget)
        container_scroll.setWidgetResizable(True)

        self.original = InputImageWidget(self, "Original")
        self.filter_layout.addWidget(self.original)
        self.disp_img = QLabel("Output")
        # self.orig_img = QLabel("Original")
        disp_scroll = QScrollArea()
        disp_scroll.setWidget(self.disp_img)
        disp_scroll.setWidgetResizable(True)
        # orig_scroll = QScrollArea()
        # orig_scroll.setWidget(self.orig_img)
        # orig_scroll.setWidgetResizable(True)
        self.update_display()

        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(container_scroll)
        main_layout.addWidget(disp_scroll)
        # main_layout.addWidget(orig_scroll)
        self.widget = QWidget()
        self.widget.setLayout(main_layout)

        self.setCentralWidget(self.widget)
        self.show()

    def update_display(self):
        frame = None
        imageout = None
        for i in range(self.filter_layout.count()):
            filter = self.get_filter(i)
            if self.original.image is not None:
                frame = filter.apply_to(frame)
                filter.image = frame.copy()
                imageout = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                q_img = QImage(imageout, imageout.shape[1], imageout.shape[0],
                               imageout.strides[0], QImage.Format_RGB888)
                pmap = QPixmap(q_img)
                scaled = pmap.scaledToHeight(100) if imageout.shape[0] > imageout.shape[1] else pmap.scaledToWidth(100)
                filter.demo.setPixmap(scaled)
            else:
                filter.image = None
                filter.demo.setText(filter.text)
        self.imageout = imageout
        # oframe = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        # oq_img = QImage(oframe, oframe.shape[1], oframe.shape[0], oframe.strides[0], QImage.Format_RGB888)
        if self.original.image is not None:
            self.disp_img.setPixmap(QPixmap(q_img))
        else:
            self.disp_img.setText("Output")
        # self.orig_img.setPixmap(QPixmap(oq_img))

    def get_filter(self, index):
        if index >= self.filter_layout.count():
            index = self.filter_layout.count() - 2
        return self.filter_layout.itemAt(index).widget()

    def add_filter(self):
        new_filter = FilterWidget(self, self.selected_filter)
        self.filter_layout.addWidget(new_filter)
        self.update_display()

    def select_filter(self):
        if self.sender().isChecked():
            last = self.radio_button
            self.radio_button = self.sender()
            if last is not None:
                last.setAutoExclusive(False)
                last.setChecked(False)
                last.setAutoExclusive(True)
            self.selected_filter = self.radio_button.filter_settings

    def print_sequence(self):
        for i in range(self.filter_layout.count()):
            print(self.get_filter(i).get_text(), end=" ")
        print()

    def save_image(self):
        filename = QFileDialog.getSaveFileName(self, "Save Output")[0]
        if not "." in filename:
            filename += ".jpg"
        cv.imwrite(filename, cv.cvtColor(self.imageout, cv.COLOR_BGR2RGB))

    def remove_filter(self):
        item = self.filter_layout.itemAt(self.filter_layout.count() - 1)
        if item == None:
            return
        removed_filter = item.widget()
        removed_filter.setParent(None)
        self.update_display()

    def clear_filters(self):
        while self.filter_layout.count() > 1:
            item = self.filter_layout.itemAt(1)
            if item == None:
                break
            removed_filter = item.widget()
            removed_filter.setParent(None)

        self.update_display()


class InputImageWidget(QGroupBox):
    def __init__(self, window, name, *args, **kwargs):
        super(QWidget, self).__init__(name, *args, **kwargs)

        self.window = window
        self.enabled = True
        self.text = name
        self.image = None
        self.isvid = False
        self.streamer = None
        self.cam = None
        self.cam_list = None

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        self.text_layout = QVBoxLayout()
        main_layout.addLayout(self.text_layout)
        self.slider_layout = QHBoxLayout()
        main_layout.addLayout(self.slider_layout)
        self.num_layout = QVBoxLayout()
        main_layout.addLayout(self.num_layout)
        main_layout.addStretch()

        position_options = QGridLayout()
        main_layout.addLayout(position_options);
        self.open_button = QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        position_options.addWidget(self.open_button, 0, 0)
        self.clear_button = QPushButton("Clear Image")
        self.clear_button.clicked.connect(self.clear_image)
        position_options.addWidget(self.clear_button, 0, 1)
        self.input_type = QLabel("image")
        position_options.addWidget(self.input_type, 0, 2)
        video_box = QCheckBox()
        video_box.setChecked(Qt.Unchecked)
        video_box.stateChanged.connect(self.vid_toggle)
        position_options.addWidget(video_box, 0, 3)
        speed_slider = QSlider(Qt.Horizontal)
        speed_slider.setMinimum(50)
        speed_slider.setMaximum(3000)
        speed_slider.setSingleStep(10)
        speed_slider.setValue(100)
        self.slider_layout.addWidget(QLabel("Update Speed"))
        self.speed_text = QLabel(str(speed_slider.value()))
        speed_slider.valueChanged.connect(lambda: self.change_speed(speed_slider))
        self.slider_layout.addWidget(speed_slider)
        self.slider_layout.addWidget(self.speed_text)

        self.demo = QLabel(self.text)
        main_layout.addWidget(self.demo)

    def apply_to(self, nothing):
        return self.image

    def open_image(self):
        if self.isvid:
            self.load_video()
        else:
            self.image = cv.imread(QFileDialog.getOpenFileName(self, "Choose Image")[0])
        self.window.update_display()

    def clear_image(self):
        if self.isvid:
            self.end_video()
        else:
            self.image = None
        self.window.update_display()

    def get_text(self):
        return self.text

    def vid_toggle(self, s):
        self.isvid = s == Qt.Checked
        self.open_button.setText("Start Video" if s else "Open Image")
        self.clear_button.setText("End Video" if s else "Clear Image")
        self.input_type.setText("video" if s else "image")
        if (self.streamer and not s):
            self.end_video()

    def change_speed(self, s):
        speed = s.value()
        self.speed_text.setText(str(speed))
        self.window.update_speed = speed
        if self.streamer:
            self.streamer.change_speed(speed)

    def load_video(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()
        self.cam = self.cam_list[0]

        self.cam.Init()
        nodemap = self.cam.GetNodeMap()

        sNodemap = self.cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        node_newestonly_mode = node_newestonly.GetValue()
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))

        node_pixel_format_RGB8 = node_pixel_format.GetEntryByName('RGB8')

        pixel_format_RGB8 = node_pixel_format_RGB8.GetValue()
        node_pixel_format.SetIntValue(pixel_format_RGB8)

        self.cam.BeginAcquisition()
        self.ready = True;

        self.streamThread = QThread()
        self.streamer = StreamObject()
        self.streamer.moveToThread(self.streamThread)
        self.streamThread.started.connect(self.streamer.stream_video)
        self.streamer.captured.connect(self.capture_frame)
        self.streamThread.start()

    def capture_frame(self):
        if self.cam:
            self.image_result = self.cam.GetNextImage(100)

            converted_image = self.image_result.GetNDArray()
            converted_image = cv.cvtColor(converted_image, cv.COLOR_BGR2RGB)
            self.image = converted_image

            self.image_result.Release()
            self.window.update_display()

    def end_video(self):
        self.streamer.end()
        self.streamer = None
        self.streamThread.quit()
        # self.streamThread = None

        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.cam = None
        self.cam_list.Clear()

        self.system.ReleaseInstance()


class StreamObject(QObject):
    finished = pyqtSignal()
    captured = pyqtSignal()
    speed = 100

    def end(self):
        self.running = False

    def stream_video(self):
        self.running = True
        while (self.running):
            time.sleep(self.speed / 1000)
            self.captured.emit()

    def change_speed(self, s):
        self.speed = s


class FilterWidget(QGroupBox):

    def __init__(self, window, filt, *args, **kwargs):
        super(QWidget, self).__init__(filt["name"], *args, **kwargs)

        self.window = window
        self.enabled = True
        self.locked = False
        self.text = filt["name"]
        self.filter_method = filt["function"]
        self.sliders = []
        self.arguments = []
        self.keywords = []
        self.image = self.window.original.image

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        self.text_layout = QVBoxLayout()
        main_layout.addLayout(self.text_layout)
        self.slider_layout = QVBoxLayout()
        main_layout.addLayout(self.slider_layout)
        self.num_layout = QVBoxLayout()
        main_layout.addLayout(self.num_layout)

        for arg in filt["ranges"]:
            i = 0
            if len(arg[1]) > 1:
                self.keywords.append((arg[1], self.get_parameter(arg[1], arg[5])))
                i = len(self.keywords) - 1
            else:
                self.arguments.append(self.get_parameter(arg[1], arg[5]))
                i = len(self.arguments) - 1
            self.add_slider(arg, i)

        if not self.sliders:
            main_layout.addStretch()

        position_options = QGridLayout()
        main_layout.addLayout(position_options);
        remove_button = QPushButton("X")
        remove_button.clicked.connect(self.close)
        position_options.addWidget(remove_button, 0, 1)

        self.enable_box = QCheckBox()
        self.enable_box.setChecked(Qt.Checked)
        self.enable_box.stateChanged.connect(self.toggle)
        position_options.addWidget(self.enable_box, 1, 1)

        self.lock_box = QCheckBox()
        self.lock_box.setChecked(Qt.Unchecked)
        self.lock_box.stateChanged.connect(self.lock)
        position_options.addWidget(self.lock_box, 2, 1)

        shift_up = QPushButton("^")
        shift_up.clicked.connect(self.move_up)
        position_options.addWidget(shift_up, 0, 0)
        shift_down = QPushButton("v")
        shift_down.clicked.connect(self.move_down)
        position_options.addWidget(shift_down, 1, 0)

        self.demo = QLabel(self.text)
        main_layout.addWidget(self.demo)

    def apply_to(self, img):
        if self.enabled:
            if self.locked:
                return self.image
            else:
                curr_time = time.time()
                ret_img = self.filter_method(img, *self.arguments, **dict(self.keywords))
                if ((time.time() - curr_time) * 1000 > self.window.update_speed):
                    self.enable_box.setChecked(Qt.Unchecked)
                return ret_img
        else:
            return img

    def toggle(self, s):
        self.enabled = s == Qt.Checked
        self.window.update_display()

    def lock(self, s):
        self.locked = s == Qt.Checked
        self.window.update_display()

    def move_up(self):
        index = self.window.filter_layout.indexOf(self) - 1
        if index < 1:
            index = window.filter_layout.count() - 1
        self.window.filter_layout.insertWidget(index, self)
        self.window.update_display()

    def move_down(self):
        index = self.window.filter_layout.indexOf(self) + 1
        if index > window.filter_layout.count() - 1:
            index = 1
        self.window.filter_layout.insertWidget(index, self)
        self.window.update_display()

    def get_text(self):
        if self.enabled:
            return self.text
        else:
            return ""

    def close(self):
        self.setParent(None)
        self.window.update_display()

    def add_slider(self, ranges, i):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(ranges[2])
        slider.setMaximum(ranges[3])
        slider.setSingleStep(ranges[4][0])
        slider.setValue(ranges[5])
        self.sliders.append(slider)
        self.num_layout.addWidget(QLabel(str(self.get_parameter(ranges[1], ranges[5]))))
        self.text_layout.addWidget(QLabel(ranges[0]))
        slider.valueChanged.connect(lambda: self.moved(slider, ranges, i))
        self.slider_layout.addWidget(slider)

    def moved(self, slider, ranges, ix):
        ptype = ranges[1]
        step = ranges[4]
        value = slider.value()
        if (value + step[1]) % step[0] != 0:
            value = round(value / step[0]) * step[0] + step[1]

        newval = self.get_parameter(ptype, value)
        if len(ranges[1]) > 1:
            self.keywords[ix] = (ranges[1], newval)
        else:
            self.arguments[ix] = newval

        self.num_layout.itemAt(self.sliders.index(slider)).widget().setText(str(newval))
        self.window.update_display()

    def get_parameter(self, ptype, value):
        if ptype == 'i':
            return value
        elif ptype == 't':
            return (value, value)
        elif ptype == 'o':
            return np.ones((value, value))
        else:
            return value


app = QApplication(sys.argv)
window = Window()
window.show()
app.exec_()
