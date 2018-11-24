"""
Author: Chris Blazej
November 23, 2018

Takes a group of images which are assumed to be stored as /images/##/*.jpg.
Checks for faces and crops the area around them. Afterwards stores the images
in...
"""
import cv2
import numpy

def cropFace(img):
    # Converts the image to grayscale.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(img, 1.1, 3)
    
    (x,y,w,h) = faces[0]
    
    # Crop Legnth
    cL = max(w,h)
    
    newY = max(y-10,0)
    newX = max(x-10,0)
    
    crop_img = img[newY:y+cL+10, newX:x+cL+10]
    
    crop_img = cv2.resize(crop_img, (224, 224), interpolation = cv2.INTER_AREA)
            
    return crop_img

def main():
    img = cv2.imread('people.jpg')
    img = cropFace(img)
    cv2.imwrite('peopleNew.jpg',img)


if (__name__ == '__main__'):
    main()