import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.optim import lr_scheduler


######################################################################
# Functions of weights_init_kaiming and weights_init_classifier are from
# https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/model.py
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, dropout, relu=False, bn=True, num_bottleneck=512, linear=True):
        super(Classifier, self).__init__()
        blocks = []
        if linear:
            blocks.append(nn.Linear(in_dims, num_bottleneck))
        else:
            num_bottleneck = in_dims
        if bn:
            blocks.append(nn.BatchNorm1d(num_bottleneck))
        if relu:
            blocks.append(nn.LeakyReLU(0.1))
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
        blocks = nn.Sequential(*blocks)
        blocks.apply(weights_init_kaiming)
        self.blocks = blocks

        classifier = nn.Linear(num_bottleneck, out_dims)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier
    def forward(self, x):
        features = self.blocks(x)
        x = self.classifier(features)
        return features, x


def build_optimizer(model, lr, weight_decay):

    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * lr},
                           {'params': model.classifier.parameters(), 'lr': lr}],
                          weight_decay=weight_decay, momentum=0.9, nesterov=True)

    #optimizer = optim.SGD(model.parameters(), weight_decay=weight_decay, lr=lr, momentum=0.9, nesterov=True)
    return optimizer


def get_scheduler(optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    return scheduler