import glob
import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class reid(Dataset):
    def __init__(self, args, transform, dtype):
        self.transform = transform
        self.loader = default_loader
        if dtype == 'train':
            data_path = os.path.join(args.DATA_DIR, dtype, 'train_list.txt')
            imgs, labels = [], []
            with open(data_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    imgs.append(os.path.join(args.data_dir, dtype, 'train_set', line.split()[0].split('/')[1]))
                    labels.append(int(line.strip().split()[1]))
        elif dtype == 'test':
            imgs, labels = [], []
            data_path = os.path.join(args.DATA_DIR, 'test', 'query_a_list.txt')
            with open(data_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    imgs.append(os.path.join(args.data_dir, dtype, 'query_a', line.split()[0].split('/')[1]))
                    labels.append(int(line.strip().split()[1]))
        else:
            data_path = os.path.join(args.DATA_DIR, 'test', 'gallery_a')
            imgs = glob.glob(os.path.join(data_path, '*.png'))
            labels = [0] * len(imgs)

        self.imgs = imgs
        self.labels = labels


    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


    def __len__(self):
        return len(self.imgs)