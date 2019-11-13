import os
import torch
import datetime
from config import cfg
from data.build import make_dataloader
from models.resnet import resnet50
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from models.units import build_optimizer, get_scheduler


train_loader, test_loader = make_dataloader(cfg)
model = resnet50(cfg.DATASET.NUM_CLASS)
CELoss = CrossEntropyLoss()
optimizer = build_optimizer(model, cfg.TRAIN.LR, cfg.TRAIN.WEIGHT_DECAY)
scheduler = get_scheduler(optimizer)


def train(model):
    if not os.path.exists(cfg.LOGS.DIR):
        os.makedirs(cfg.LOGS.DIR)
    writer = SummaryWriter(cfg.LOGS.DIR)

    for epoch in range(cfg.TRAIN.EPOCHES + 1):
        starttime = datetime.datetime.now()

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            model = model.cuda()
            outputs = model(inputs)
            loss = CELoss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        train_loss, train_corrects = 0.0, 0.0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            model = model.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            loss = CELoss(outputs, labels)
            pred = torch.argmax(outputs, dim=1)
            train_loss += loss.item() * cfg.TRAIN.BATCHSIZE

            correct = float(torch.sum(pred == labels))
            train_corrects += correct


        val_loss, val_corrects = 0.0, 0.0
        for data in test_loader:
            inputs, labels = data

            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            loss = CELoss(outputs, labels)
            pred = torch.argmax(outputs, dim=1)
            val_loss += loss.item() * cfg.TRAIN.BATCHSIZE
            correct = float(torch.sum(pred == labels))
            val_corrects += correct
        model.train()

        endtime = datetime.datetime.now()
        total_time = (endtime - starttime).seconds

        if not os.path.exists(cfg.CHECKPOINTS.DIR):
            os.makedirs(cfg.CHECKPOINTS.DIR)
        if epoch % cfg.CHECKPOINTS.INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINTS.DIR, "{}_{}.pth".format(cfg.MODEL.NAME, epoch)))