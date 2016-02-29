from __future__ import print_function
import keras
import waifu2x
import jsonpickle
import Image as img
import os
from argparse import ArgumentParser 
from scipy.ndimage import imread as img_reader

def build_parser():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument('-i', '--image_folder', required=True)

    # Arguments with default values
    parser.add_argument('-e', '--epochs', default=5)
    parser.add_argument('-b', '--batch_size', default=32)
    parser.add_argument('-c', '--checkpoint', default=10000)
    parser.add_argument('-cd', '--checkpoint_directory', default='checkpoint_directory')
    
    return parser


def get_img_filepaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for fname in filenames:
            yield os.path.abspath(os.path.join(dirpath, fname))
            

def generate_data(imgs_folder):
    for fpath in get_img_filepaths(directory):        
        print ('Reading image', fname)
        image_tensor = img_reader(fname, mode='RGB')        
        yield patch_list 


def train(args=None):
    model = waifu2x.model.create()
    progbar = keras.utils.generic_utils.Progbar(X_train.shape[0])
    for epoch in range(args.epochs):
        for iteration, (train_x, train_y) in enumerate(generate_data(args.img_folder)):
            loss = model.train_on_batch(train_x, train_y)
            progbar.add(X_batch.shape[0], values=[('train loss', loss[0])])

            if (iteration % args.checkpoint == 0):
                #checkpoint reached, save model and validate performance
                val_x, val_y = get_validation(args.batchsize)
                model.predict(val_x, val_y)										
                test_progbar = keras.utils.generic_utils.Progbar(val_x.shape[0])
                score = model.test_on_batch(val_x, val_y)
                progbar.add(val_x.shape[0], values=[('test loss', score[0])])



                
if __name__ == "__main__":
   args = build_parser().parse_args()
   train(args)
