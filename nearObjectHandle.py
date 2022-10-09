import time
import warn_user
from gtts import gTTS
from playsound import playsound

# warn_user code modified fromhttps://www.geeksforgeeks.org/convert-text-speech-python/

# This module is imported so that we can
# play the converted audio
import os

# Handles the cases where the object is near the user
def nearObjectHandle(x, y, imageSize):
    
    xLMBound = imageSize[1] * (1.0/3.0) # left bound for x val
    xRMBound = imageSize[1] * (2.0/3.0) # right boudnd for x val
    yBound = imageSize[0] * (2.0/3.0)   # 'top' bound for y val

    

    # Determine if object is close by; say corresponding
    # audio cue if close
    fd = None
    if (y >= yBound):
        if (x <= xLMBound):
            #fd = os.open(r"left.mp3", os.O_RDONLY)
            playsound(r"left.mp3")
            #print("Object Left")
        elif (x > xLMBound and x < xRMBound):
            #fd = os.open(r"center.mp3", os.O_RDONLY)
            playsound(r"center.mp3")
            #print("Object Center")
        elif (x >= xRMBound):
            #fd = os.open(r"right.mp3", os.O_RDONLY)
            playsound(r"right.mp3")
            #print("Object Right")
    
    if fd is not None:
        os.close(fd)

    # slight delay to reduce spam
    #time.sleep(3)

if __name__ == "__main__":
    nearObjectHandle(0, 0, (500,500), 50)
    