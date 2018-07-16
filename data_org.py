# coil-100

import os

keys = list(sorted(list(set([s[0:s.index('_')+1] for s in os.listdir() if '_' in s]))))

for k in keys:
    imgs = [s for s in os.listdir() if k in s]
    for i in imgs:
        os.rename(i, os.path.join(k[0:-1], i))



