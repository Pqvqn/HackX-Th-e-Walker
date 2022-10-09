# Import the required module for text
# to speech conversion
from gtts import gTTS
import playsound

# code modified fromhttps://www.geeksforgeeks.org/convert-text-speech-python/

# This module is imported so that we can
# play the converted audio
import os
from playsound import playsound

def warn_user(warnMessage):
    # The text that you want to convert to audio
    mytext = "right.mp3"

    # Language in which you want to convert
    language = 'en'

    # Passing the text and language to the engine,
    # here we have marked slow=False. Which tells
    # the module that the converted audio should
    # have a high speed
    myobj = gTTS(text=warnMessage, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome
    myobj.save(mytext)

    # Playing the converted file
    #os.system("mpg321 " + mytext)
    playsound.playsound(mytext)

if __name__ == "__main__":
    warn_user("Object right")