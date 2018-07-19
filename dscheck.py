#!/usr/bin/env python

import dataset_check as dc
from time import time
import numpy as np
import argparse
import os
import sys


def percentChange(random_guess, classifier):
    return (classifier-random_guess)/random_guess*100


def printHeader(out, name, path, crop_size, epochs):
    print('# Results for %s' % name, file=out)
    print('Directory: %s' % path, file=out)

    if crop_size is not None:
        print('Crop size: (%d,%d)' % crop_size, file=out)
    else:
        print('Crop size: None')

    print('Epochs: %d' % epochs, file=out)


def quickTest(name, data, epochs, output=sys.stdout):
    history, model = dc.trainModel(
        name,
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['num_classes'],
        data['image_size'],
        epochs,
        output=output)

    dc.plotTrainingHistory(name,
                           history,
                           epochs,
                           save='%s/%s_training_hist.png' % (results_dir, name))

    # creates data['y_pred']
    dc.runTest(data, model, f=output)

    dc.previewPredictedData('test',
                            data['X_test'],
                            data['y_test'],
                            data['y_pred'],
                            data['class_names'],
                            save='%s/%s_predicted_data.png' % (results_dir, name))


def runBatch(name, path, crops, epochs_list, nocrop_epochs=0, output=sys.stdout):

    if len(crops) != len(epochs_list):
        raise ValueError('crops and epochs must be the same length')


    if nocrop_epochs > 0:
        runSingle(name + '_x0', path, None, nocrop_epochs, output=output)

    for crop, epochs in zip(crops, epochs_list):
        print('\n\n\n', file=out)

        runSingle(name + '_x' + str(crop[0]), path, crop, epochs, output=output)


def runSingle(name, path, crop, epochs, output=sys.stdout, data=None):

    printHeader(out, name, path, crop, epochs)

    # Load all data
    data = dc.loadAllImages(path,
                            crop_size=crop,
                            do_shuffle=True,
                            random_state=int(time()),
                            split=True,
                            output=out)

    # Sample training data
    X = data['X_train']
    y = data['y_train']

    # Preview a single image
    dc.previewImage(X[2],
                    title=np.argmax(y[2]),
                    save='%s/%s_preview_image.png' % (results_dir, name))

    # Print information
    for k in ['num_classes', 'image_size', 'class_names']:
        print('%s: %s' % (k, data[k]), file=out)

    dc.previewData('sanity check',
                   X, y, data['class_names'],
                   save='%s/%s_sanity_check.png' % (results_dir, name))

    out.flush()

    quickTest(name, data, epochs, output=out)

    random_guess = 1.0 / data['num_classes']
    acc = data['pred_acc']
    print('Random guess accuracy:   %.8f' % random_guess, file=output)
    print('Classifier accuracy:     %.8f' % acc, file=output)
    print('Improvement over chance: %.8f' % percentChange(random_guess, acc), file=output)

    return data



if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('-size', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=4)
    args = parser.parse_args()
    args.dir = os.path.abspath(args.dir)

    # dataset name
    name = args.dir.split('/')[-1] # + '_x' + str(args.size)

    # create a directory
    global results_dir
    results_dir = 'results/%s' % name
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    out = open(os.path.join(results_dir, '%s_info.txt' % name), 'w')


    # crop = (args.size, args.size)
    # if args.size <= 0:
        # crop = None

    # runSingle(name, args.dir, crop, args.epochs, output=out)

    crops = [(x,x) for x in [50, 20, 10]]
    epochs = [40, 70, 90]

    runBatch(name, args.dir, crops, epochs, nocrop_epochs=20, output=out)


