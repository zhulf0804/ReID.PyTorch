from yacs.config import CfgNode as CN

_C = CN()

# Dasaset
_C.DATASET = CN()
_C.DATASET.NAME = 'market'
_C.DATASET.DATA_DIR = '/root/Market-1501-v15.09.15'
_C.DATASET.NUM_CLASS = 751


# Input
_C.INPUT = CN()
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.SIZE_TRAIN = [256, 128]
_C.INPUT.PADDING = 10
_C.INPUT.RE_PROB = 0.5


# Dataloader
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.NUM_INSTANCE = 16


## MODEL
_C.MODEL = CN()
_C.MODEL.NAME = 'resnet50'

# Train
_C.TRAIN = CN()
_C.TRAIN.BATCHSIZE = 32
_C.TRAIN.EPOCHES = 60
_C.TRAIN.LR = 0.05
_C.TRAIN.WEIGHT_DECAY = 5e-4
_C.TRAIN.TRIPLETLOSS = False


# Logs
_C.LOGS = CN()
_C.LOGS.INTERVAL = 1
_C.LOGS.DIR = 'logs'


# Checkpoints
_C.CHECKPOINTS = CN()
_C.CHECKPOINTS.INTERVAL = 10
_C.CHECKPOINTS.DIR = 'checkpoints'


# Test
_C.TEST = CN()
_C.TEST.BATCHSIZE = 128
