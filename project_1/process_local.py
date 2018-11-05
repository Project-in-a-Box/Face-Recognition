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

# TODO: Import CV2
import cv2


# TODO: Edit this function
def process_image():   
	CV_GRAYSCALE = 0;
	img = cv2.imread('geisel.jpg', CV_GRAYSCALE)
	width = int(img.shape[1]*(0.5))
	height = int(img.shape[0]*(0.5))
	recwidth = 100;
	recheight = 100;
	recpt1 = ((int(width)/2)-(int(recwidth)/2), (int(height)/2)-(int(recheight)/2));
	recpt2 = ((int(width)/2)+(int(recwidth)/2), (int(height)/2) + (int(recheight)/2));
	img = cv2.resize(img, (width ,height), interpolation=cv2.INTER_AREA)
	outputimg = cv2.rectangle(img, recpt1, recpt2, (255, 255, 255), 5) 
	cv2.imwrite('geisel-bw-rectangle.jpg', outputimg)
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
