#!/usr/bin/env python

import dataset_check as dc
from time import time
import numpy as np
import argparse
import os

if __name__ == '__main__':

    # Clear a couple lines after printed text from import statements
    print('\n\n\n')

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('-size', default=20)
    args = parser.parse_args()
    args.dir = os.path.abspath(args.dir)

    # dataset name
    name = args.dir.split('/')[-1] + '_x' + str(args.size)

    # Load all data
    data = dc.loadAllImages(args.dir,
                            crop_size=(args.size, args.size),
                            do_shuffle=True,
                            random_state=int(time()),
                            split=True)

    # Sample training data
    X = data['X_train']
    y = data['y_train']

    # Preview a single image
    dc.previewImage(X[2], title=np.argmax(y[2]), save='plots/%s_preview_image.png' % name)

    # Print information
    for k in ['num_classes', 'image_size', 'class_names']:
        print('%s: %s' % (k, data[k]))

    dc.previewData('sanity check',
                   X, y, data['class_names'],
                   save='plots/%s_sanity_check.png' % name)
