import os
import torch
import datetime
import argparse
from contextlib import redirect_stdout
from config import cfg
from data.build import make_dataloader
from models.models_zoo import models
from losses.losses import CrossEntropyLoss
from tensorboardX import SummaryWriter
from utils.optim import build_optimizer
from utils.lr import get_scheduler
from evaluate import evaluate


def train(name, cfg, model, train_loader, test_loader, query_loader, CELoss, optimizer, scheduler):
    if not os.path.exists(os.path.join(name, cfg.LOGS.DIR)):
        os.makedirs(os.path.join(name, cfg.LOGS.DIR))
    writer = SummaryWriter(os.path.join(name, cfg.LOGS.DIR))

    for epoch in range(cfg.TRAIN.EPOCHES):
        model.eval()
        train_loss, train_corrects, num = 0.0, 0.0, 0
        for data in train_loader:
            inputs, labels, _ = data
            num += inputs.shape[0]
            inputs, labels = inputs.cuda(), labels.cuda()
            model = model.cuda()
            with torch.no_grad():
                _, outputs = model(inputs)
            loss = CELoss(outputs, labels)
            pred = torch.argmax(outputs, dim=1)
            train_loss += loss.item() * cfg.TRAIN.BATCHSIZE
            correct = float(torch.sum(pred == labels))
            train_corrects += correct
        avg_train_loss, avg_train_acc = train_loss / num, train_corrects / num
        writer.add_scalar('loss', avg_train_loss, epoch)
        writer.add_scalar('acc', avg_train_acc, epoch)
        if epoch % cfg.LOGS.INTERVAL == 0:
            print("Epoch {}, lr {}".format(epoch, optimizer.param_groups[-1]['lr']))
            print("Train loss: {}, train acc: {}".format(avg_train_loss, avg_train_acc))
        if cfg.TEST.ENABLE and (epoch == 0 or epoch % cfg.CHECKPOINTS.INTERVAL == 9):
            CMC, mAP = evaluate(cfg, model, query_loader, test_loader)
            print("Rank@1:{}, Rank@5: {}, Rank@10: {}, mAP: {}".format(CMC[0],
                                                                       CMC[4],
                                                                       CMC[9],
                                                                       mAP))
        starttime = datetime.datetime.now()
        model.train()
        for data in train_loader:
            inputs, labels, _ = data
            inputs, labels = inputs.cuda(), labels.cuda()
            model = model.cuda()
            _, outputs = model(inputs)
            loss = CELoss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        endtime = datetime.datetime.now()
        total_time = (endtime - starttime).seconds
        if epoch % cfg.LOGS.INTERVAL == 0:
            print("train time {}s".format(total_time))
        if not os.path.exists(os.path.join(name, cfg.CHECKPOINTS.DIR)):
            os.makedirs(os.path.join(name, cfg.CHECKPOINTS.DIR))
        if epoch == 0 or epoch % cfg.CHECKPOINTS.INTERVAL == 9:
            torch.save(model.state_dict(), os.path.join(name, cfg.CHECKPOINTS.DIR,
                                                        "{}_{}.pth".format(cfg.MODEL.NAME, epoch + 1)))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='default', help='Dataset directory')
    parser.add_argument('--yaml', type=str, default='', help='Dataset directory')
    args = parser.parse_args()
    if args.yaml:
        cfg.merge_from_file(args.yaml)
        cfg.RERANK.ENABLE = False
        cfg.freeze()
    print(cfg)
    if not os.path.exists(args.name):
        os.makedirs(args.name)
    with open(os.path.join(args.name, cfg.CONFIG.SAVED_FILE), 'w') as f:
        with redirect_stdout(f): print(cfg.dump())
    train_loader, test_loader, query_loader = make_dataloader(cfg)
    model = models[cfg.MODEL.NAME](cfg.DATASET.NUM_CLASS, cfg.TRAIN.DROPOUT, cfg.MODEL.RESNET_STRIDE)
    CELoss = CrossEntropyLoss(cfg.DATASET.NUM_CLASS, label_smooth=cfg.LOSS.SMOOTH)
    optimizer = build_optimizer(model, cfg.TRAIN.LR, cfg.TRAIN.WEIGHT_DECAY)
    scheduler = get_scheduler(cfg, optimizer)
    train(args.name, cfg, model, train_loader, test_loader, query_loader, CELoss, optimizer, scheduler)
