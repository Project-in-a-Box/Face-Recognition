"""
ECE196 Face Recognition Project
Author: Will Chen

Prerequisite: You need to install OpenCV before running this code
The code here is an example of what you can write to print out 'Hello World!'
Now modify this code to process a local image and do the following:
1. Read geisel.jpg
2. Convert color to gray scale
3. Resize to half of its original dimensions
4. Draw a box at the center the image with size 100x100
5. Save image with the name, "geisel-bw-rectangle.jpg" to the local directory
All the above steps should be in one function called process_image()
"""

# TODO: Import OpenCV
import cv2

# Processes a given image by converting it to grayscale, using scale to scale its size, and
# draws a rectangle (more like square) with length rect and a certain color. 
def process_image(imgName = 'geisel.jpg',scale = 0.5, rect = 100, color = (int(255),int(255),int(255))):
	# Reads the image in grayscale, which is the zero.
	img = cv2.imread(imgName,0)
	
	# Scales down (or up!) the image that was read
	width = int(img.shape[1] * scale)
	height = int(img.shape[0] * scale)

	dim = (width, height)

	# Resizes our image to our newly scaled dimensions
	img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

	# Defines the top left and bottom right corners of a rectangle centered in the middle of the picture.
	pt1 = (int(width*scale - rect/2), int(height*.5 - rect/2))
	pt2 = (int(width*scale + rect/2), int(height*.5 + rect/2))

	# Draws the rectangle ontop of the picture with the given color with a width of 5 by default.
	cv2.rectangle(img, pt1, pt2, color, 5)
	
	# Creates a new image file of course.
	writeWord = '-bw-rectangle.jpg'
	writeImgName = imgName.split('.')[0] + writeWord

	cv2.imwrite(writeImgName,img)

	return

# Just prints 'Hello World! to screen.
def hello_world():
    print('Hello World!')
    return

# TODO: Call process_image function.
def main():
    process_image()
    return


if(__name__ == '__main__'):
    main()
