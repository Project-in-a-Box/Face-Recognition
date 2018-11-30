"""
ECE196 Face Recognition Project
Author: W Chen

Adapted from: https://keras.io/getting-started/functional-api-guide/

Modify this code to write a LeNet with the following requirements:
* Input dimensions: 32x32x1 - Input
* C1: Convolutional Layer - Conv2D
    number of filters: 6
    kernel size: 5x5
    strides: 1 both horizontally and vertically (Set by default.)
    activation function: sigmoid
    output: 6 layers of 28x28 feature maps (Do not need to specify in function.)
* S2: Max Pooling Layer - MaxPooling2D
    pooling size: 2x2
    strides: 2 both horizontally and vertically
    output: 6 layers of 14x14 feature maps (Do not need to specify in function.)
* C3: Convolutional Layer - Conv2D
    number of filters: 16
    kernel size: 5x5
    strides: 1 both horizontally and vertically
    activation function: sigmoid
    output: 16 layers of 10x10 feature maps(Do not need to specify in function.)
* S4: Max Pooling Layer - MaxPooling2D
    pooling size: 2x2
    strides: 2 both horizontally and vertically
    output: 16 layers of 5x5 feature maps (Do not need to specify in function.)
* C5: Convolutional Layer - Conv2D
    number of filters: 120
    kernel size: 5x5
    strides: 1 both horizontally and vertically
    activation function: sigmoid
    output: 120 layers of 1x1 feature maps(Do not need to specify in function.)
* F6: Fully Connected Layer - Dense
    units: 84
    activation function: tanh
    output 84-dimensional vector (This is specified through units.)
* F7: Fully Connected Layer - Dense
    units: 10
    activation function: softmax
    output 10-dimensional vector (This is specified through units.)
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import Model



def main():
	inputs = Input(shape=(32,32,1))

	x = Conv2D(6, (5, 5), strides=(1, 1), activation='sigmoid')(inputs)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x)
	x = Conv2D(16, (5, 5), strides=(1, 1), activation='sigmoid')(x)
	x = MaxPooling2D((2,2), strides=(2, 2))(x)
	x = Conv2D(120, (5, 5), strides=(1, 1), activation='sigmoid')(x)
	x = Dense(84, activation='tanh')(x)
	
	outputs = Dense(10, activation='softmax')(x)

	model = Model(inputs=inputs, outputs=outputs)

	# Prints model architecture
	model.summary()
	return

if(__name__ == '__main__'):
	main()
