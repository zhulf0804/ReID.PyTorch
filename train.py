import argparse
import datetime
import os
import torch
from tensorboardX import SummaryWriter
from datasets import get_train_datasets
from models.resnet import resnet50
from models.densenet import densenet121
from models.units import build_optimizer, get_scheduler, get_loss


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/Users/zhulf/data/reid_match/reid', help='Dataset directory')
parser.add_argument('--epoches', type=int, default=60, help='Number of traing epoches')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--init_lr', type=float, default=0.05, help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on parameters')
parser.add_argument('--log_interval', type=int, default=1, help='Print iterval')
parser.add_argument('--log_dir', type=str, default='logs', help='Train/val loss and accuracy logs')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='Checkpoint saved interval')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--model', type=str, default='resnet50', help='Model to use')
parser.add_argument('--train_all', action='store_true', help="Use train and val data to train")
args = parser.parse_args()


train_image_datasets, train_dataloaders, dataset_sizes, class_names = get_train_datasets(args.data_dir, args.batch_size)
num_classes = len(class_names)
if args.model == 'resnet50':
    model = resnet50(num_classes=num_classes)
elif args.model == 'densenet121':
    model = densenet121(num_classes=num_classes)
criterion = get_loss()
optimizer = build_optimizer(model, args.init_lr, args.weight_decay)
scheduler = get_scheduler(optimizer)

use_gpu = torch.cuda.is_available()

def train(model):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)
    phase = 'train'
    if args.train_all:
        phase = 'train_all'
    for epoch in range(args.epoches + 1):
        starttime = datetime.datetime.now()
        train_loss, train_corrects = 0.0, 0.0
        for data in train_dataloaders[phase]:
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
                model = model.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * args.batch_size
            pred = torch.argmax(outputs, dim=1)
            correct = float(torch.sum(pred == labels))
            train_corrects += correct
        scheduler.step()
        avg_train_loss = train_loss / dataset_sizes[phase]
        avg_train_ac = train_corrects / dataset_sizes[phase]

        model.eval()
        val_loss, val_corrects = 0.0, 0.0
        for data in train_dataloaders['val']:
            inputs, labels = data
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * args.batch_size
            pred = torch.argmax(outputs, dim=1)
            correct = float(torch.sum(pred == labels))
            val_corrects += correct

        model.train()
        avg_val_loss = val_loss / dataset_sizes['val']
        avg_val_ac = val_corrects / dataset_sizes['val']
        endtime = datetime.datetime.now()
        total_time = (endtime - starttime).seconds

        writer.add_scalars('loss', {'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, epoch)
        writer.add_scalars('accuracy', {'train_ac': avg_train_ac, 'val_ac': avg_val_ac}, epoch)
        if epoch % args.log_interval == 0:
            print("="*20, "Epoch {} / {}".format(epoch, args.epoches), "="*20)
            print("train loss {:.2f}, train ac {:.2f}".format(avg_train_loss, avg_train_ac))
            print("val loss {:.2f}, val ac {:.2f}".format(avg_val_loss, avg_val_ac))
            print("Training time is {:.2f} s".format(total_time))
            print("\n")
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "resnet50_{}.pth".format(epoch)))


if __name__ == '__main__':
    train(model)

# nohup python -u train.py --data_dir /root/data/Market/pytorch  --model resnet50 &
# nohup python -u train.py --data_dir /root/data/Market/pytorch --model densenet121 &
# nohup python -u train.py --data_dir /root/data/Market/pytorch --train_all --model resnet50 &
# nohup python -u train.py --data_dir /root/data/Market/pytorch --train_all --model densenet121 &