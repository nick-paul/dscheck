#!/usr/bin/env python

import dataset_check as dc
from time import time
import numpy as np
import argparse
import os
import sys
import random


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

    parser.add_argument('--crop', nargs='+', type=int, default=[10],
                        help='The size of the crop in pixels')
    parser.add_argument('--epochs', nargs='+', type=int, default=[4],
                        help='Number of epochs to test on')
    parser.add_argument('--nocrop-epochs', type=int, default=0,
                        help='If provided, run the network on the uncropped dataset')
    parser.add_argument('--name', type=str,
                        help='Provide a name for the results directory')

    args = parser.parse_args()
    args.dir = os.path.abspath(args.dir)

    # Check size and epochs list lengths
    if len(args.crop) != len(args.epochs):
        print("Arguments '--crop' and '--epochs' must have same number of elements")
        print("Got:")
        print("\tsize: %s" % args.crop)
        print("\tepochs: %s" % args.epochs)
        exit(0)

    # dataset name
    name = args.dir.split('/')[-1]  # + '_x' + str(args.size)

    # results dir name
    if args.name is None:
        args.name = ''.join([random.choice('0123456789abcde') for x in range(6)])

    # create a directory
    global results_dir
    results_dir = 'results/%s/%s' % (name, args.name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    out = open(os.path.join(results_dir, '%s_info.txt' % name), 'w')

    # Use same size width and height
    crop_dims = [(c,c) for c in args.crop]

    if len(crop_dims) == 1:
        runSingle(name, args.dir, crop_dims[0], args.epochs[0], output=out)
    else:
        runBatch(name, args.dir, crop_dims, args.epochs, nocrop_epochs=args.nocrop_epochs, output=out)

    print("\nDone! Result files created in './%s'" % results_dir)
