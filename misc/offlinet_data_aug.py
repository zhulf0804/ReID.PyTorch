import os
import numpy as np
import random
from collections import defaultdict
from shutil import copyfile


raw_data_dir = '/Users/zhulf/data/reid_match/raw'
raw_train_dir = os.path.join(raw_data_dir, '初赛训练集')
raw_test_dir = os.path.join(raw_data_dir, '初赛A榜测试集')
data_dir = '/Users/zhulf/data/reid_match'
format_data_dir = os.path.join(data_dir, 'reid_aug')
with open(os.path.join(raw_train_dir, 'train_list.txt'), 'r') as f:
    lines = f.readlines()


d = defaultdict(list)
for line in lines:
    class_dir = line.strip().split()[1]
    img_name = line.strip().split()[0].split('/')[1]
    d[class_dir].append(img_name)


# format data
train_all_path = os.path.join(format_data_dir, 'train_all')
train_path = os.path.join(format_data_dir, 'train')
val_path = os.path.join(format_data_dir, 'val')
if not os.path.isdir(train_all_path):
    os.makedirs(train_all_path)
    os.makedirs(train_path)
    os.makedirs(val_path)

l = []
for key, imgs_name in d.items():
    train_all_class_path = os.path.join(train_all_path, key)
    train_class_path = os.path.join(train_path, key)
    val_class_path = os.path.join(val_path, key)
    os.makedirs(train_all_class_path)
    os.makedirs(train_class_path)
    dst_name = list(imgs_name)
    if len(imgs_name) > 100:
        imgs_name = random.sample(imgs_name, 100)
        dst_name = list(imgs_name)
    elif len(imgs_name) < 10:
        st = len(imgs_name)
        imgs_name = imgs_name * 10
        imgs_name = imgs_name[:10]
        dst_name = list(imgs_name)
        for j in range(st, 10):
            dst_name[j] = dst_name[j].split('.')[0] + '_%d'%j + '.png'
    l.append(len(imgs_name))
    for i, img_name in enumerate(imgs_name):
        # for train_all
        src_path = os.path.join(os.path.join(raw_train_dir, 'train_set', img_name))
        dst_path = os.path.join(train_all_class_path, dst_name[i])
        copyfile(src_path, dst_path)
        dst_path = os.path.join(train_class_path, dst_name[i])
        if not os.path.exists(val_class_path):
            os.makedirs(val_class_path)
            dst_path = os.path.join(val_class_path, dst_name[i])
        copyfile(src_path, dst_path)
        print(src_path, dst_path)


a = np.sort(np.array(l))
b = np.argsort(np.array(l))
print(a[:100], a[-30:])
print(np.sum(a == 1))
print(b[:100], b[-30:])
print(sum(a))