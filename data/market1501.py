import glob
import os.path as osp
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from data.common import parse_market1501


class market(Dataset):
    def __init__(self, cfg, data_transform, dtype):
        self.is_train = False
        self.loader = default_loader
        self.transform = data_transform(cfg, is_train=False)
        if dtype == 'train':
            data_dir = osp.join(cfg.DATASET.DATA_DIR, 'bounding_box_train')
            self.transform = data_transform(cfg, is_train=True)
        elif dtype =='test':
            data_dir = osp.join(cfg.DATASET.DATA_DIR, 'bounding_box_test')
        elif dtype == 'query':
            data_dir = osp.join(cfg.DATASET.DATA_DIR, 'query')

        imgs_path = glob.glob(osp.join(data_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)
        ids = [parse_market1501(img_path)[1] for img_path in imgs_path if parse_market1501(img_path)[1] != -1]
        print(len(ids))
        unique_ids = sorted(set(ids))
        print(len(unique_ids))
        self.id2label = {}
        for ind, id in enumerate(unique_ids):
            self.id2label[id] = ind
        self.data_source = []
        for img_path in imgs_path:
            path, pid, camid = parse_market1501(img_path)
            if pid == -1:
                continue
            if dtype == 'train':
                label = self.id2label[pid]
            else:
                label = pid

            self.data_source.append([path, label, camid])

    def __getitem__(self, index):
        img_path, label, camid = self.data_source[index]
        img = self.loader(img_path)
        img = self.transform(img)
        return img, label, camid

    def __len__(self):
        return len(self.data_source)


if __name__ == '__main__':
    from config import cfg
    from data.transform import data_transform
    market_test = market(cfg, data_transform, 'train')
    print(market_test.__len__())
    print(market_test.__getitem__(5))