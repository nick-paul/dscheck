import sys
import cv2
import os
import numpy as np
from time import time

import matplotlib.pyplot as plt

from random import randint

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from keras.preprocessing.image import img_to_array

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp

plt.ion()  # interactive mode
print('deep learning modules imported')


##################
# Image Cropping #
##################


def cropImage(img, w, h, kind='ul'):
    #print(img.shape)

    # (C,h,w) => (h,w,C)
    img = np.rollaxis(img, 0, 3)

    W = img.shape[1]
    H = img.shape[0]

    if kind == 'ul':
        cropped = img[0:0+h, 0:0+w]
    elif kind == 'll':
        cropped = img[H-h:H, 0:0+w]
    elif kind == 'ur':
        cropped = img[0:0+h, W-w:W]
    elif kind == 'lr':
        cropped = img[H-h:H, W-w:W]
    elif kind == 'rand':
        x = randint(0, H-h)
        y = randint(0, W-w)
        cropped = img[x:x+h, y:y+w]
    else:
        raise ValueError("Invalid argument for 'kind': %s" + str(kind))

    # (h,w,C) => (C,h,w)
    cropped = np.rollaxis(cropped, 0, 3)
    cropped = np.rollaxis(cropped, 0, 3)

    return cropped

#################
# Image Loading #
#################


def imagePaths(root):
    imgs = {}

    for dir in os.listdir(root):
        path = os.path.join(root, dir)
        if os.path.isdir(path):
            imgs[dir] = []
            for f in os.listdir(path):
                imgs[dir].append(os.path.join(path, f))
    return imgs


def loadImage(path, output=sys.stdout, crop_size=None):
    """
    Load an image into a numpy array, if crop_size is proveded, crop the image
    crop_size: Tuple (w,h)
    """
    image = cv2.imread(path)
    if image is None:
        print("Unable to load image '%s'" % path, file=output)
        return None
    image = image.astype("float") / 255.0
    image = img_to_array(image)

    if crop_size is not None:
        image = cropImage(image, crop_size[0], crop_size[1], kind='ul')

    return image

################################
# Data Preview / Visualization #
################################


def previewImage(image, title=None, save='preview.png', show=False):
    """
    Preview an image using matplotlib
    """
    plt.figure()
    im = np.transpose(image, (1,2,0))
    #im = image
    rgb = np.fliplr(im.reshape(-1,3)).reshape(im.shape)
    plt.imshow(rgb)
    if title is not None:
        plt.title(title)

    if save is not None:
        plt.savefig(save)

    if show:
        plt.show()


def previewData(name, X_train, y_train, class_names, width=5, height=5, save=None, show=False):
    unique_imgs = 3
    offset = randint(0,len(y_train[0]) - (1 + unique_imgs))
    count = width * height
    fig = plt.figure(figsize=(width*1.7,height*2))

    def nextimg():
        label = offset + randint(0,unique_imgs-1)
        n = randint(0, len(y_train)-1)
        return label, n

    for i in range(count):
        label, n = nextimg()
        while int(np.argmax(y_train[n])) != label:
            label, n = nextimg()

        ax = fig.add_subplot(height,width,i+1, xticks=[], yticks=[])
        im = X_train[n,::]
        im = np.transpose(im, (1,2,0))
        rgb = np.fliplr(im.reshape(-1,3)).reshape(im.shape)
        plt.imshow(rgb)
        ax.set_title('%s : %d' % (class_names[int(np.argmax(y_train[n]))], n))

    plt.suptitle(name)
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()


##########################
# Load and organize data #
##########################



def loadAllImages(
        root,
        crop_size=None,
        do_shuffle=False,
        split=False,
        random_state=0,
        output=sys.stdout):

    """
    X, y = loadAllImages()
    X: image data
    y: labels
    """
    data = {}

    image_paths = imagePaths(root)

    labels = []
    imgs = []
    for k,paths in image_paths.items():
        for p in paths:
            labels.append(k)
            imgs.append(loadImage(p, crop_size=crop_size))

    X, y = np.array(imgs), labels

    if do_shuffle:
        X, y = shuffle(X, y, random_state=random_state)

    class_names = sorted(list(set(labels)))
    data['class_names'] = class_names
    data['num_classes'] = len(data['class_names'])

    # shape is (C,h,w) => (w,h)
    data['image_size'] = (X[0].shape[2], X[0].shape[1])

    # class names -> indices
    y = [class_names.index(name) for name in y]

    y = np_utils.to_categorical(y, data['num_classes'])


    if split:
        # Split data
        # 75% Train
        # 12.5% Validation (Dev)
        # 12.5% Test

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing and validataion
        (X_train, X_test_val, y_train, y_test_val) =\
            train_test_split(X, y, test_size=0.25, random_state=0)

        # Split test_val into testing and validation sets 50/50
        (X_test, X_val, y_test, y_val) = train_test_split(
            X_test_val, y_test_val, test_size=0.5, random_state=0)

        print('Training set size: %d' % len(y_train), file=output)
        print('Test set size: %d' % len(y_test), file=output)
        print('Validation set size: %d' % len(y_val), file=output)

        data['X_train'] = X_train
        data['X_test'] = X_test
        data['X_val'] = X_val
        data['y_train'] = y_train
        data['y_test'] = y_test
        data['y_val'] = y_val
    else:
        data['X'] = X
        data['y'] = y

    return data



