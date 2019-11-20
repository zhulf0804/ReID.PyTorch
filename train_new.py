import os
import torch
import datetime
from config import cfg
from data.build import make_dataloader
from models.resnet import resnet50
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from models.units import build_optimizer, get_scheduler
from evaluate_new import evaluate


train_loader, test_loader, query_loader = make_dataloader(cfg)
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
            inputs, labels, _ = data
            inputs, labels = inputs.cuda(), labels.cuda()
            model = model.cuda()
            _, outputs = model(inputs)
            loss = CELoss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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
        #print(train_corrects, num)
        avg_train_loss, avg_train_acc = train_loss / num, train_corrects / num

        '''
        val_loss, val_corrects, num = 0.0, 0.0, 0
        for data in query_loader:
            inputs, labels, _ = data
            num += inputs.shape[0]
            inputs, labels = inputs.cuda(), labels.cuda()
            model = model.cuda()
            with torch.no_grad():
                _, outputs = model(inputs)
            loss = CELoss(outputs, labels)
            pred = torch.argmax(outputs, dim=1)
            val_loss += loss.item() * cfg.TRAIN.BATCHSIZE
            correct = float(torch.sum(pred == labels))
            val_corrects += correct
        avg_query_loss, avg_query_acc = val_loss / num, val_corrects / num
        '''



        endtime = datetime.datetime.now()
        total_time = (endtime - starttime).seconds
        if epoch % cfg.LOGS.INTERVAL == 0:
            print("Epoch {}, train time {}s".format(epoch, total_time))
            print("Train loss: {}, train acc: {}".format(avg_train_loss, avg_train_acc))
            #print("Query loss: {}, Query acc: {}".format(avg_query_loss, avg_query_acc))

            print("\n")
        if not os.path.exists(cfg.CHECKPOINTS.DIR):
            os.makedirs(cfg.CHECKPOINTS.DIR)
        if epoch % cfg.CHECKPOINTS.INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINTS.DIR, "{}_{}.pth".format(cfg.MODEL.NAME, epoch)))
            CMC, mAP = evaluate(model, query_loader, test_loader, train_loader)
            print("Rank@1:{}, Rank@5: {}, Rank@10: {}, mAP: {}".format(CMC[0],
                                                                       CMC[4],
                                                                       CMC[9],
                                                                       mAP))

        model.train()
        scheduler.step()


train(model)
