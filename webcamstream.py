import cv2
from PIL import Image, ImageFilter

cap = cv2.VideoCapture(0)
i = 0

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")



while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    cv2.imwrite('frame.png', frame)
    image = Image.open(r"frame.png")
    image = image.convert("L")
    image = image.filter(ImageFilter.FIND_EDGES)
    image.save(r"Edge_Sample.png")
    edgeim = cv2.imread(r"Edge_Sample.png")
    cv2.imshow('Edge', edgeim)

    c = cv2.waitKey(1)
    if c == 27:
        cv2.imwrite('edge'+str(i)+'.png', edgeim)
        i = i + 1



cap.release()
cv2.destroyAllWindows()