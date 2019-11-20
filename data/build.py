from torch.utils.data.dataloader import DataLoader
from data.market1501 import market
from data.transform import data_transform
from data.sampler import RandomIdentitySampler_alignedreid

dataset_names = {
    'market': market
}

def make_dataloader(cfg):
    train_set = dataset_names[cfg.DATASET.NAME](cfg, data_transform, 'train')
    test_set = dataset_names[cfg.DATASET.NAME](cfg, data_transform, 'test')
    query_set = dataset_names[cfg.DATASET.NAME](cfg, data_transform, 'query')
    if cfg.TRAIN.TRIPLETLOSS:
        train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCHSIZE,
                                  sampler=RandomIdentitySampler_alignedreid(train_set.data_source, cfg.DATALOADER.NUM_INSTANCE),
                                  num_workers=cfg.DATALOADER.NUM_WORKERS)
    else:
        train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS)
    query_loader = DataLoader(query_set, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=False,
                             num_workers=cfg.DATALOADER.NUM_WORKERS)
    return train_loader, test_loader, query_loader


if __name__ == '__main__':
    from config import cfg
    train_loader, test_loader, query_loader = make_dataloader(cfg)
    for data in train_loader:
        print(data)