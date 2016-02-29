from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad

def convolutional_dnn(channels, img_rows, img_cols):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape= (channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution(32,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    mode.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(channels, 3, 3))
    model.add(Dense(channels * img_rows * img_cols))    
    return model

def create_model():
    return convolutional_dnn(channels=3, img_rows=32, img_cols=32)
    
