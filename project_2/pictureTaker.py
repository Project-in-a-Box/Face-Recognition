"""
ECE196 Face Recognition Project
Author: Chris Blazej

Adapted from:
http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
"""


# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
import os.path

def main():
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (800, 608)
    camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=(800, 608))

    if(not os.path.exists("./images/")):
        os.mkdir("images")

    print("Taking pictures in:")

    for i in list(reversed(range(1,6))):
        print(i)
        time.sleep(1)
    
    frameCount = 0
    pictureNum = 0
    
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array

        if (frameCount >= 6):
            cv2.imwrite("./images/" + str(pictureNum) + '.jpg', image)
            print("./images/" + str(pictureNum) + ".jpg written")
            pictureNum = pictureNum + 1
            frameCount = 0
        else:
            frameCount = frameCount + 1
        
        # show the frame
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
                
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
                
if (__name__ == "__main__"):
    main()
