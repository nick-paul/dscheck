import sys
import cv2
import os
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from random import randint

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from .LeNet import LeNet

# Not including these will cause a dimension mismatch
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from keras import backend as K
if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")


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

    image_count = sum([len(v) for k,v in image_paths.items()])
    with tqdm(total=image_count) as pbar:
        for k,paths in image_paths.items():
            for p in paths:
                labels.append(k)
                imgs.append(loadImage(p, crop_size=crop_size))
                pbar.update()

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


######################
# Training The Model #
######################


def trainModel(name,
               trainX,
               trainY,
               valX,
               valY,
               num_classes,
               img_size,
               epochs,
               batch_size=45,
               init_lr=1e-3,
               output=sys.stdout):

    aug = ImageDataGenerator(rotation_range=30, 
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")

    # initialize the model
    print("[INFO] building model '%s' ..." % name, file=output)
    model = LeNet.build(width=img_size[0],
                        height=img_size[1],
                        depth=3,
                        classes=num_classes)

    print(model.summary())

    loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

    print("[INFO] compiling model '%s' ..." % name, file=output)
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=["accuracy"])

    # trainY_idx = np.arange(len(trainY[0]))

    # train the network
    steps_per_epoch = len(trainX) // batch_size
    print('Steps per epoch:', steps_per_epoch)
    print("[INFO] training network...", file=output)
    H = model.fit_generator(aug.flow(trainX, trainY),
                            validation_data=(valX, valY),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...", file=output)
    save_path = 'models/%s.model' % name
    print("Saving model as ./%s" % save_path, file=output)
    model.save(save_path)

    return H, model


def runTest(data, model, f=sys.stdout):
    score = model.evaluate(data['X_test'], data['y_test'])

    print('Loss on test data: %s' % score[0], file=f)
    data['pred_loss'] = score[0]

    print('Accuracy on test data: %s' % score[1], file=f)
    data['pred_acc'] = score[1]

    y_pred = model.predict(data['X_test'])
    data['y_pred'] = y_pred

    # prediction data
    num_correct = np.sum(np.argmax(data['y_test'], axis=1) == np.argmax(y_pred, axis=1))
    #print(data['y_test'])
    #print(y_pred)
    print('Correct / Total: %d/%d' % (num_correct, len(data['y_test'])), file=f)

    data['pred_num_correct'] = num_correct


##############
# Evaluation #
##############

# plot the training loss and accuracy
def plotTrainingHistory(name, H, epochs, save='validation_history.png', show=False):
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on classes")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()


def previewPredictedData(
        name,
        X_test,
        y_truth,
        y_pred,
        class_names,
        width=5,
        height=5,
        save='preview_predicted.png',
        show=False):

    count = width * height
    if len(X_test) < count:
        count = X_test
        height = 2
        width = count // 2

    fig = plt.figure(figsize=(width*1.7,height*2))

    for i in range(count):
        n = randint(0,len(y_truth)-1)
        ax = fig.add_subplot(height,width,i+1, xticks=[], yticks=[])
        im = X_test[n,::]
        im = np.transpose(im, (1,2,0))
        rgb = np.fliplr(im.reshape(-1,3)).reshape(im.shape)
        plt.imshow(rgb)
        #if y_truth[n] == y_pred[n]:
        #    title = class_names[int(np.argmax(y_truth[n]))]
        #else:
        title = '%s/%s' % (
                class_names[int(np.argmax(y_pred[n]))],
                class_names[int(np.argmax(y_truth[n]))])

        ax.set_title(title)

    plt.suptitle('%s: predicted / correct' % name)
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
