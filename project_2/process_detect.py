"""
ECE196 Face Recognition Project
Author: W Chen

Adapted from:
http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

Use this code as a template to process images in real time, using the same techniques as the last challenge.
You need to display a gray scale video with 320x240 dimensions, with box at the center
"""


# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

def process_image(img, newWidth, newHeight):
    # Converts the image to grayscale.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resizes our image to our newly scaled dimensions
    img = cv2.resize(img, (newWidth, newHeight), interpolation = cv2.INTER_AREA)
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(img, 1.15, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
                    
    return img

def main():
  # initialize the camera and grab a reference to the raw camera capture
  camera = PiCamera()
  camera.resolution = (160, 120)
  camera.framerate = 12
  rawCapture = PiRGBArray(camera, size=(160, 120))
  
  # allow the camera to warmup
  time.sleep(0.1)
  
  # capture frames from the camera
  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
      # grab the raw NumPy array representing the image, then initialize the timestamp
      # and occupied/unoccupied text
      image = frame.array
      
      image = process_image(image, 160, 120)
  
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
