import time
import warn_user

# warn_user code modified fromhttps://www.geeksforgeeks.org/convert-text-speech-python/

# This module is imported so that we can
# play the converted audio
import os

# Handles the cases where the object is near the user
def nearObjectHandle(x, y, imageSize, relDist):
    xLMBound = imageSize[0] * (1.0/3.0) # left bound for x val
    xRMBound = imageSize[0] * (2.0/3.0) # right boudnd for x val
    yBound = imageSize[1] * (2.0/3.0)   # 'top' bound for y val

    # Determine if object is close by; say corresponding
    # audio cue if close
    if (y >= yBound):
        if (x <= xLMBound):
            warn_user("OBJECT LEFT IN " + relDist + "inches")
        elif (x > xLMBound & x < xRMBound):
            warn_user("OBJECT CENTER IN " + relDist + "inches")
        elif (x >= xRMBound):
            warn_user("OBJECT RIGHT IN " + relDist + "inches")

    # slight delay to reduce spam
    time.wait(1.5)

if __name__ == "__main__":
    nearObjectHandle(0, 0, (500,500), 50)
    