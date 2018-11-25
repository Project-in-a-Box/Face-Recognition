"""
Author: Chris Blazej
November 23, 2018

Takes a group of images which are assumed to be stored as /images/##/*.jpg.
Checks for faces and crops the area around them. Afterwards stores the images
in...
"""
import cv2
import numpy
import glob
import os.path
import shutil
import os
import array
import random

def formatNum(number):
  # Determines if we should append a zero to the left of the string
  # or not to allow for correct formatting. If we're outside the range
  # use '00' as a default.
  if (0 < number < 10):
    numStr = '0' + str(number)
  elif ( 10 <= number < 100):
    numStr = str(number)
  else:
    numStr = '00'

  return numStr

def getImgNames(path, number):
  # Returns all the file names in images/##/ in the format of .jpg.
  return glob.glob(path + formatNum(number) + '/*.jpg')


def cropFace(img):
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

  faces = face_cascade.detectMultiScale(img, 1.3, 5)
  
  # If we found no faces, just return nothing/a zero.
  if len(faces) is 0:
    return None
  
  (x,y,w,h) = faces[0]
  
  # Crop Legnth
  cL = max(w,h)
  
  newY = max(y-10,0)
  newX = max(x-10,0)
  
  crop_img = img[newY:y+cL+10, newX:x+cL+10]
  
  crop_img = cv2.resize(crop_img, (224, 224), interpolation = cv2.INTER_AREA)
   
  return crop_img


def main():
  dirCount = 0
  numStr = formatNum(dirCount)
  
  imagesDir = os.getcwd() + '/images/'
  dataDir = imagesDir + 'Data/'
  
  fileCount = array.array('i')
  
  trainDir = dataDir + 'Train/'
  validationDir = dataDir + 'Validation/'
  testDir = dataDir + 'Test/'
  
  # If none of these folders exist, create them.
  if (not os.path.exists(dataDir)):
    os.mkdir(dataDir)
  
  if (not os.path.exists(trainDir)):
    os.mkdir(trainDir)
  
  if (not os.path.exists(validationDir)):
    os.mkdir(validationDir)
  
  if (not os.path.exists(testDir)):
    os.mkdir(testDir)
  
  
  # While the next ## + 1 directory exists, we will open it and crop the faces out, transfering the editted photos to cleaned.   
  while (os.path.exists(imagesDir + numStr + '/')):
    fileNames = getImgNames(imagesDir, dirCount)

    if(not os.path.exists(dataDir + numStr)):
      os.mkdir(dataDir + numStr)
    
    fileI = 0
    for name in fileNames:
      img = cv2.imread(name, 0)
      img = cropFace(img)
    
      # If this image isn't a face, move on to the next file name.
      if img is None:
        continue
      else:
        baseName = numStr + '_' + str(fileI) + '.jpg'
        cv2.imwrite(dataDir + numStr + '/' + baseName, img)
        fileI = fileI + 1
    
    fileCount.append(fileI)
    dirCount = dirCount+1
    numStr = formatNum(dirCount)

  validCount = 0;
  testCount = 0;
  trainCount = 0;

  for i in range(0, dirCount):
    numStr = formatNum(i)
   
    if (not os.path.exists(trainDir + numStr)):
      os.mkdir(trainDir + numStr)
             
    if (not os.path.exists(validationDir + numStr)):
      os.mkdir(validationDir + numStr)
     
    if (not os.path.exists(testDir + numStr)):
      os.mkdir(testDir + numStr)         
         
    for j in range(0, fileCount[i]):
      randChoice = random.randint(1,11)
      curFile = dataDir + numStr + '/' + numStr + '_' + str(j) + '.jpg'
         
      if (1 <= randChoice <= 7):
        shutil.move(curFile, trainDir + numStr)
        trainCount = trainCount + 1
      elif (randChoice is 8):
        shutil.move(curFile, validationDir + numStr)
        validCount = validCount + 1
      else:
        shutil.move(curFile, testDir + numStr)
        testCount = testCount + 1
        
    os.rmdir(dataDir + numStr)
    
  totalCount = validCount + testCount + trainCount
  validPercent = (float)(validCount)/(totalCount) * 100
  testPercent = (float)(testCount)/(totalCount) * 100
  trainPercent = (float)(trainCount)/(totalCount) * 100
  print("Total images: %d" % totalCount)
  print("Validation Percent: %.1f" % validPercent) 
  print("Test Percent: %.1f" % testPercent)
  print("Train Percent: %.1f" % trainPercent)


if (__name__ == '__main__'):
  main()