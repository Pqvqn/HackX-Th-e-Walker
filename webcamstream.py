import cv2
import roadmask
import os
import disttrig
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)
i = 0

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")



while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #cv2.imshow('Input', frame)
    cv2.imwrite('frame.png', frame)
    frame = cv2.imread(r"frame.png")
    #image = image.convert("L")
    #image = image.filter(ImageFilter.FIND_EDGES)
    #image.save(r"Edge_Sample.png")
    #edgeim = cv2.imread(r"Edge_Sample.png")
    #cv2.imshow('Edge', edgeim)
    roadimg, cnt, hiera = roadmask.make_mask(frame)
    omask, lows = roadmask.find_hazards(roadimg, cnt, hiera)

    #disttrig.readImage(roadimg)
    disttrig.distanceToObstruction(roadimg, lows, 60, 7, 69.4)
    cv2.imshow("O",omask)
    #plt.show()
    #cv2.imshow('Road Mask', omask)    

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()