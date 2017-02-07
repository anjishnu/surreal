from __future__ import print_function
import numpy as np
import math
import logging 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from PIL import Image 
import scipy.misc

logger = logging.getLogger()

class DCGAN(object):

    # Deep Convolutional Generative Adversarial Network

    def __init__(self, input_size=100, input_shape=(1, 28, 28)):
        self.BATCH_SIZE=128
        self.input_shape = input_shape

        image_dimension = 1
        for dimension in input_shape:
            image_dimension = image_dimension*dimension            
        self.input_size = image_dimension/4 # We are downscaling by a factor of 4

        self.generator = self.get_generator()
        self.discriminator = self.get_discriminator()        
        # E2E model
        self.discriminator_on_generator = self.get_discriminator_on_generator(self.generator, 
                                                                              self.discriminator)        
                
        # Model compilation start
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True) # discriminator optimizer
        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True) # generator optimizer        
        self.generator.compile(loss='binary_crossentropy', optimizer="SGD")
        self.discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
        # Complete

    
    def get_generator(self):
        '''
        Generative part of the network
        '''
        model = Sequential()
        model.add(Dense(input_dim=self.input_size, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        return model

    
    def get_discriminator(self):
        '''
        Discriminator model
        '''
        model = Sequential()
        model.add(Convolution2D(64, 5, 5, border_mode='same',
                                input_shape=self.input_shape))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 5, 5))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model


    def get_discriminator_on_generator(self, generator, discriminator):
        '''
        Composite model - generator and discriminator 
        '''
        model = Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        return model

    
    def combine_images(self, generated_images):
        '''
        Output the images in an easy to read representation
        '''
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
 
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[0, :, :]
            
        image = image*127.5+127.5 # Cast the image back into 0-255 range
        return image

    
    def train(self):
        '''
        Start training the DCGAN
        '''

        # Initialize data 
        # For now load data from here - we should be able to modify this API soon. 
        (X_train, y_train), (X_test, y_test) = mnist.load_data()        

        # We don't care about labels right now
        self.X_train = (X_train.astype(np.float32) - 127.5)/127.5 # Normalize to (-1, 1) range
        self.X_train = self.X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])        
        print ("Training images", self.X_train.shape)

        self.train_model()
        
        
    def train_model(self, num_epochs=100):
        for self.epoch in range(num_epochs):
            print("Epoch is", self.epoch)
            logging.info("training epoch {}".format(str(self.epoch)))
            self.train_epoch()
            
            
    def get_random_noise(self): # Change to get inputs

        #noise = np.zeros((self.BATCH_SIZE, self.input_size))
        #for i in range(self.BATCH_SIZE):
        #    # Uniform noise 
        #    noise[i, :] = np.random.uniform(-1, 1, self.input_size)

        input_pixels = np.zeros((self.BATCH_SIZE, self.input_size))

        for i, image in enumerate(self.image_batch):
            # Keras to Scipy
            # print ('1:', image.shape)
            # 1: (1, 28, 28)
            #image = np.rollaxis(image, 0, 3)
            # 2D_image 
            img_2d = image[0]
            #2: (28, 28, 1)
            # Downscaling

            img_2d = scipy.misc.imresize(img_2d, 0.5, 'bicubic') 
            # print ('3:', image.shape)
            # Scipy to keras
            input_pixels[i, :] = img_2d.flatten()
           
        return input_pixels

    
    def generate_images(self, image_batch, batch_size=None):
        if not batch_size: batch_size = self.BATCH_SIZE
        noise = self.get_random_noise()
        generated_images = self.generator.predict(noise, verbose=0)        
        return generated_images

    
    def train_epoch(self):

        number_of_batches = int(self.X_train.shape[0]/self.BATCH_SIZE)        
        print("Number of batches", str(number_of_batches))         

        for self.index in range(number_of_batches):                

            # Random noise to start off with, replace with latent vectors when you get time

            batch_start, batch_end = (self.index)*self.BATCH_SIZE, (self.index+1)*self.BATCH_SIZE
            image_batch = self.X_train[batch_start : batch_end]           
            self.image_batch = image_batch
            generated_images = self.generate_images(image_batch) 
            
            if self.index % 20 == 0:
                # Monitor the progress of the model
                gen_image = self.combine_images(generated_images)
                ref_image = self.combine_images(self.image_batch)
                output_img="debug/epoch_{epoch}_img_{index}.png".format(epoch=str(self.epoch), 
                                                                        index=str(self.index))
                ref_img = "debug/reference_epoch_{epoch}_img_{index}.png".format(epoch=str(self.epoch),
                                                                        index=str(self.index))
                Image.fromarray(gen_image.astype(np.uint8)).save(output_img)
                Image.fromarray(ref_image.astype(np.uint8)).save(ref_img)
             
            logging.info('Training discriminator...')
            self.train_discriminator(image_batch, generated_images)            
            logging.info('Training generator...')
            self.train_generator()
                        
            if self.index % 10 == 9:
                # Save weights every now and then
                self.generator.save_weights('checkpoint/latest_generator.h5', True)
                self.discriminator.save_weights('checkpoint/latest_discriminator.h5', True)
                print('Saved weights...epoch:{} {}'.format(str(self.epoch), str(self.index)))


    def train_discriminator(self, real_image_batch, generated_images_batch):
        # Train discriminator
        self.discriminator.trainable = True
        # Putting the training data into the right format - class 1 is 'real' class 0 if 'fake'

        X = np.concatenate((real_image_batch, generated_images_batch))
        y = [1] * len(real_image_batch) + [0] * len(generated_images_batch)
        d_loss = self.discriminator.train_on_batch(X, y)
        print("batch {index} distriminative_loss : {loss}".format(index=str(self.index), loss=str(d_loss)))
        
    
    def train_generator(self):
        # Train generator 
        noise = self.get_random_noise()            
        self.discriminator.trainable = False            
        g_loss = self.discriminator_on_generator.train_on_batch(noise, [1] * self.BATCH_SIZE)            
        print("batch {index} generative_loss : {loss}".format(index=str(self.index), loss=str(g_loss)))
        

    @classmethod
    def generate(self, nice=False):
        '''
        Generate images with a given model
        '''
        
        generator = self.get_generator()
        generator.compile(loss='binary_crossentropy', optimizer="SGD")
        generator.load_weights('generator')

        if nice:
            discriminator = self.get_discriminator()
            discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
            discriminator.load_weights('discriminator')            
                
            noise = self.get_random_noise()
            generated_images = generator.predict(noise, verbose=1)
            d_pret = discriminator.predict(generated_images, verbose=1)
            
            index = np.arange(0, self.BATCH_SIZE*20)
            index.resize((self.BATCH_SIZE*20, 1))
            pre_with_index = list(np.append(d_pret, index, axis=1))

            pre_with_index.sort(key=lambda x: x[0], reverse=True)
            nice_images = np.zeros((self.BATCH_SIZE, 1) + (generated_images.shape[2:]), dtype=np.float32)
            
            for i in range(int(BATCH_SIZE)):
                idx = int(pre_with_index[i][1])
                nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
            image = self.combine_images(nice_images)
            
        else:
            noise = self.get_random_noise()
            generated_images = generator.predict(noise, verbose=1)
            image = self.combine_images(generated_images)
        
        Image.fromarray(image.astype(np.uint8)).save("generated_image.png")
        
        

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help='train or generate')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        dcgan = DCGAN()
        dcgan.train()
    elif args.mode == "generate":
        DCGAN.generate(BATCH_SIZE=args.batch_size, nice=args.nice)
