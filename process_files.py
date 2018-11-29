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
  
def createFolders(paths):
  # If none of these folders exist, create them.
  for path in paths:
    if (not os.path.exists(path)):
      os.mkdir(path)

# We want to know if a directory in the format path/## exists.
def dirNumExist(path, directory):
  return os.path.exists(path + directory)
  
# Returns a value of the first directory that doesn't exist in dataPath/##
def newestDirectory(dataPath):
  i = 0;
  while (i < 100):
    numStr = formatNum(i)
    if (not dirNumExist(dataPath, numStr)):
      return i
    
    i = i + 1
    
  return -1
    
def getPartition(fileNames):
  validPercent = 0.1
  testPercent = 0.2
  trainPercent = 0.7
  
  totalCnt = len(fileNames)
  validCnt = int(totalCnt*validPercent)
  testCnt = int(totalCnt*testPercent)
  trainCnt = int(totalCnt*trainPercent)
  sum = trainCnt + validCnt + testCnt
  # If our sum is too big, then subtract the difference from trainCnt (Because its the biggest partition).
  # Else if we're below, add the difference to the smallest paritition. Otherwise return the normal count.
  if ( sum > totalCnt ):
    return [validCnt, testCnt, trainCnt - (sum - totalCnt)]
  elif ( sum < totalCnt):
    return [validCnt + (totalCnt - sum), testCnt, trainCnt]
  else:
    return [validCnt, testCnt, trainCnt]
    
def writeFiles(fileNames, path, numStr):
  croppedNames = []
  fileI = 0
  for name in fileNames:
    img = cv2.imread(name, 0)
    img = cropFace(img)
    
    # If this image isn't a face, move on to the next file name.
    if img is None:
      continue
    else:
      baseName = numStr + '_' + str(fileI) + '.jpg'
      croppedNames.append(baseName)
      cv2.imwrite(path + baseName, img)
      fileI = fileI + 1
      
  return croppedNames
    
def partitionFiles(fileNames, path, partitionPaths):
  cur = 0
  counter = 0
  partitionNum = numpy.array([0,0,0])
  dataCnt = getPartition(fileNames)
  for name in fileNames:
    if(counter < dataCnt[cur]):
      shutil.move(path + name, partitionPaths[cur])
      counter = counter + 1
    else:
      partitionNum[cur] = counter
      cur = cur + 1
      shutil.move(path + name, partitionPaths[cur])
      counter = 0

  partitionNum[cur] = counter + 2
  return partitionNum

def main():
  imagesDir = os.getcwd() + '/images/'
  dataDir = imagesDir + 'Data/'  
  trainDir = dataDir + 'Train/'
  validationDir = dataDir + 'Validation/'
  testDir = dataDir + 'Test/'
  tempDir = os.getcwd() + '/temp/'
  
  # Creates the directories that will be used for data partitioning.
  createFolders([dataDir, trainDir, validationDir, testDir, tempDir])
  
  # If process_files.py has already ran, we only want to update if images contains new folders. Check data/train/ for the lowest
  # non-existing folder. 
  dirCount = newestDirectory(trainDir)
  numStr = formatNum(dirCount)
  parNum = numpy.array([0,0,0])
  
  # While the next ## directory exists, we will open it and crop the faces out, transfering the editted photos to cleaned.   
  while (dirNumExist(imagesDir, numStr)):
    print("Person " + numStr + ": Getting file names.")
    fileNames = getImgNames(imagesDir, dirCount)
    
    print("Person " + numStr + ": Creating folders in our partition.")
    createFolders([trainDir+numStr, validationDir+numStr, testDir+numStr])
    
    print("Person " + numStr + ": Writing the cropped files to /temp/.")
    croppedNames = writeFiles(fileNames, tempDir, numStr)
    
    writeDir = [validationDir+numStr, testDir+numStr, trainDir+numStr]
    
    print("Person " + numStr + ": Parititioning files in /temp/")
    parNum = parNum + partitionFiles(croppedNames, tempDir, writeDir)    
        
    print(" ")    
    dirCount = dirCount+1
    numStr = formatNum(dirCount)

  os.rmdir(tempDir)
  totalNum = numpy.sum(parNum)
  validNum = parNum[0]
  testNum = parNum[1]
  trainNum = parNum[2]
  
  validPercent = (validNum/float(totalNum))*100
  testPercent = (testNum/float(totalNum))*100
  trainPercent = (trainNum/float(totalNum))*100
  
  print("Validation Percent: %.2f" % validPercent)
  print("Test Percent: %.2f" % testPercent)
  print("Train Percent: %.2f" % trainPercent)

if (__name__ == '__main__'):
  main()