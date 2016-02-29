from __future__ import print_function
import keras
from PIL import Image
import os
import numpy as np
from argparse import ArgumentParser 
from scipy.ndimage import imread 
from models import create_model
from sklearn.feature_extraction.image import PatchExtractor


def build_parser():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument('-i', '--image_folder', required=True)

    # Arguments with default values
    parser.add_argument('-e', '--epochs', default=5)
    parser.add_argument('-b', '--batch_size', default=32)
    parser.add_argument('-c', '--checkpoint', default=10000)
    parser.add_argument('-cd', '--checkpoint_directory',
                        default='checkpoint_directory')    

    return parser # Return the parser 


def get_img_filepaths(directory):
    # Traverse a directory and generate images
    for dirpath, _, filenames in os.walk(directory):
        for fname in filenames:
            yield os.path.abspath(os.path.join(dirpath, fname))

            
def generate_data(img_folder):

    patch_extractor = PatchExtractor(patch_size=(32,32),
                                     max_patches=0.01)
    
    def resize(img_tensor, scale=0.5):
        # Generate a half size image tensor
        channels, max_x, max_y = img_tensor.shape
        img_tensor = np.rollaxis(img_tensor, axis=0, start=3)
        image = Image.fromarray(np.asarray(img_tensor), mode='RGB')
        small_img = np.array(image.resize(size=(int(max_x * scale),
                                                int(max_y * scale))))
        x_small, y_small, channels = small_img.shape
        return np.rollaxis(small_img, axis=2)
        
    for fpath in get_img_filepaths(img_folder):        
        print ('Reading image', fpath)
        img_tensor = imread(fpath, mode='RGB')
        input_matrix = np.array([img_tensor])
        patches = patch_extractor.transform(input_matrix)                
        patches = np.rollaxis(patches, axis=3, start=1)
        small_patches = np.array([resize(patch) for patch in patches])        
        patches = np.array([p.reshape(p.shape[0]*p.shape[1]*p.shape[2]) for p in patches])
        print ("Shapes of tensors", small_patches.shape, patches.shape)
        yield small_patches, patches


def train(args=None):
    model = create_model()
    print ('created model')
    total_iterations = 0
    for epoch in range(args.epochs):
        print ('Training epoch', epoch)
        for iteration, (train_x, train_y) in enumerate(generate_data(args.image_folder)):
            progbar = keras.utils.generic_utils.Progbar(train_x.shape[0])    
            loss = model.train_on_batch(train_x, train_y)
            progbar.add(train_x.shape[0], values=[('train loss', loss[0])])
            total_iterations += 1
            if (total_iterations % args.checkpoint == 0):
                print ('Reached checkpoint at iteration', iteration)
                #checkpoint reached, save model and validate performance
                # val_x, val_y = get_validation(args.batchsize)
                # model.predict(val_x, val_y)                
                # test_progbar = keras.utils.generic_utils.Progbar(val_x.shape[0])
                # score = model.test_on_batch(val_x, val_y)
                # progbar.add(val_x.shape[0], values=[('test loss', score[0])])
                
if __name__ == "__main__":
   args = build_parser().parse_args()
   train(args)
