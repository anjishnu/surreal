from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

def convolutional_dnn(channels, input_patch, output_patch):
    ''' Builds a CNN model for super-resolution'''
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape= (channels, input_patch, input_patch)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(channels, 3, 3))
    model.add(Flatten())
    model.add(Dense(channels * output_patch * output_patch))    
    model.compile(loss='mse', optimizer=Adagrad())    
    return model

def create_model():
    return convolutional_dnn(channels=3, input_patch=16, output_patch=32)
    
