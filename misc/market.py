import os
import glob
import numpy as np
for root, dirs, files in os.walk("/Users/zhulf/data/market/train_all", topdown=False):
    #for name in files:
    #    print(os.path.join(root, name))
    a = []
    for name in dirs:
        #print(os.path.join(root, name))
        imgs = glob.glob(os.path.join(root, name, '*.jpg'))
        a.append(len(imgs))

b = sum(a)
c = np.sort(np.array(a))
print(b)
print(c)