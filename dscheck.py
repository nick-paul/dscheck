#!/usr/bin/env python

import dataset_check as dc
from time import time
import numpy as np

if __name__ == '__main__':
    print('dataset checker loaded!')

    data = dc.loadAllImages(
        '/Users/npaul/Documents/git/face-dl/data/orl',
        crop_size=(50,50), do_shuffle=True, random_state=int(time()), split=True)

    X = data['X_train']
    y = data['y_train']

    dc.previewImage(X[2], title=np.argmax(y[2]), save='image.png')

    for k in ['num_classes', 'image_size', 'class_names']:
        print('%s: %s' % (k, data[k]))

    dc.previewData('sanity check', X, y, data['class_names'], save='sanity_check.png')
