from __future__ import print_function
import keras
from PIL import Image
import os
import numpy as np
from argparse import ArgumentParser
from scipy.ndimage import imread
from scipy.misc import imsave, toimage
from models import create_model
from sklearn.feature_extraction.image import PatchExtractor

def build_parser():

    parser = ArgumentParser()

    # Required arguments
    parser.add_argument('-i', '--image_folder', default="images")

    # Arguments with default values
    parser.add_argument('-e', '--epochs', default=5000)
    parser.add_argument('-b', '--batch_size', default=4)
    parser.add_argument('-c', '--checkpoint', default=1000)
    parser.add_argument('-cd', '--checkpoint_directory',
                        default='checkpoint_directory')
    parser.add_argument('--patch_size', '-p', default=32)
    parser.add_argument('--debug', '-d', action='store_true', default=False)
    parser.add_argument('--train', '-t',  action='store_true', default=False)

    return parser # Return the parser


def get_img_filepaths(directory):
    # Traverse a directory and generate images
    for dirpath, _, filenames in os.walk(directory):
        for fname in filenames:
            yield os.path.abspath(os.path.join(dirpath, fname))


def resize(img_tensor, scale=0.5, i=0):
    # Generate a half size image tensor
    channels, max_x, max_y = img_tensor.shape

    # img_tensor.shape :  (max_x, max_y, channels)
    img_tensor = np.rollaxis(img_tensor, axis=0, start=3)

    # small_img.shape : (max_x, max_y, channels)
    small_img = np.array(toimage(img_tensor).resize(size=(int(max_x * scale),
                                                          int(max_y * scale))))
    x_small, y_small, args.channels = small_img.shape
    output = np.rollaxis(small_img, axis=2) # (channels, max_x, max_y)
    return output

def generate_data(img_folder, max_patches=0.001):
    for fpath in get_img_filepaths(img_folder):
        print ('Reading image', fpath)
        patch_extractor = PatchExtractor(patch_size=(32,32),
                                             max_patches=max_patches)

        img_tensor = imread(fpath, mode='RGB')
        # shape : (row, col, channels)

        input_matrix = np.array([img_tensor])
        # shape : (1, row, col, channels)

        input_matrix = input_matrix/255.0 # Casting into 0 to 1 space which DNN models learn faster
        
        patches = patch_extractor.transform(input_matrix)
        # shape : (n_samples, row, col, channels)

        patches = np.rollaxis(patches, axis=3, start=1)
        # shape : (n_samples, channels, row, col)

        small_patches = np.array([resize(patch) for patch in patches])
        # shape : (n_samples, channels, max_x, max_y)

        patches = np.array([p.reshape(p.shape[0] * p.shape[1] * p.shape[2])
                            for p in patches])
        # shape : (n_samples, output_vector_size)

        if False:
            # Print out values to debug
            print ("Shapes of tensors", small_patches.shape, patches.shape)
            for i, (small, big) in enumerate(zip(small_patches, patches)):
                small_img = np.rollaxis(small, axis=0, start=3)
                if not os.path.exists('debug'):
                    os.makedirs('debug')
                imsave('debug/small_patch_{}.jpg'.format(i), small_img)
                imsave('debug/big_patch_{}.jpg'.format(i), vec2img(big))

        yield small_patches, patches

def vec2img(vector, n_channels=3):
    ''' Convert softmax output to images
    vector = (n_channels * n_rows * n_cols)
    '''
    tensor = vector.reshape(n_channels, args.patch_size, args.patch_size)
    row_col_channel_tensor = np.rollaxis(tensor, axis=0, start=3) # shape : (rows, cols, channels)
    return row_col_channel_tensor * 255 # Putting back into rgb format

def img_to_input(image):
    '''
    image.shape : (row, col, channels)
    input_img.shape :  (channels, row, col)

    return: (1, channels, row, col)
    '''
    input_img = np.rollaxis(image, axis=2, start=0)
    return np.array([input_img])


def load_model(model_path):
    model = create_model()
    model.load_weights(model_path)
    return model


def decode(model, image):
    model_X = img_to_input(image)
    output = model.predict(model_X)
    img = vec2img(output_vector)
    return True

def test():
    model = load_model('/Users/anjikum/github_projects/surreal/checkpoint_directory/150500_model.h5py')
    val_x, val_y = next(generate_data('/Users/anjikum/github_projects/surreal/images', max_patches = 0.0001))

    score = model.test_on_batch(val_x, val_y)
    print (score)

    if args.debug:
        pred_y = model.predict(val_x)
        for index, (orig, real, pred) in enumerate(zip(val_x, val_y, pred_y)):
            print ('Saving index', index)
            print (orig.shape)
            in_patch = np.rollaxis(orig, axis=0, start=3)
            print (in_patch.shape)
            imsave('debug/real_patch_{}.jpg'.format(index), vec2img(real))
            imsave('debug/pred_patch_{}.jpg'.format(index), vec2img(pred))
            imsave('debug/input_patch_{}.jpg'.format(index), in_patch)

    print ('Done')


def train(args=None):
    model = create_model()
    print ('Created model...')
    total_iterations = 0
    sum_iterations = 0
    checkpoint_num = 0
    for epoch in range(args.epochs):
        print ('Training epoch', epoch)
        print ('total iterations', total_iterations)

        for iteration, (train_x, train_y) in enumerate(generate_data(args.image_folder,
                                                                     max_patches=0.05)):
            val_x, val_y = next(generate_data(args.image_folder, max_patches = 0.001)) 
            print (len(val_x), len(val_y))
            model.fit(train_x, train_y,
                      validation_data = (val_x, val_y),
                      batch_size=args.batch_size,
                      nb_epoch=1, show_accuracy=True)

            checkpoint_filepath = os.path.join(args.checkpoint_directory,
                                               '{}_model.h5py'.format(total_iterations))
            if not os.path.exists(args.checkpoint_directory):
                os.makedirs(args.checkpoint_directory)

            # Saving model
            model.save_weights(checkpoint_filepath, overwrite=True)

            # Scoring model
            val_x, val_y = next(generate_data(args.image_folder, max_patches = 0.0001))
            score = model.test_on_batch(val_x, val_y)

            # Save some images to see how well the model is training
            pred_y = model.predict(val_x)

            for i, (orig, real, pred) in enumerate(zip(val_x, val_y, pred_y)):
                in_patch = np.rollaxis(orig, axis=0, start=3)
                imsave('debug/real_patch_{0}_{1}.jpg'.format(i,checkpoint_num),
                       vec2img(real))
                imsave('debug/pred_patch_{0}_{1}.jpg'.format(i, checkpoint_num),
                       vec2img(pred))
                imsave('debug/input_patch_{0}_{1}.jpg'.format(i, checkpoint_num),
                       in_patch)
            print (i, 'images saved for debugging')
            print ('test loss', score[0])

if __name__ == "__main__":
   args = build_parser().parse_args()
   train(args)
