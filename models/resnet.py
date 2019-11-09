import torch
import torch.nn as nn
import torchvision.models as models
from models.units import Classifier


class resnet18(nn.Module):
    def __init__(self, num_classes, dropout=0.5, stride=2):
        super(resnet18, self).__init__()
        model = models.resnet18(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv1.stride = (1, 1)
        self.model = model
        fc_features = model.fc.in_features
        self.classifier = Classifier(fc_features, num_classes, dropout)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class resnet34(nn.Module):
    def __init__(self, num_classes, dropout=0.5, stride=2):
        super(resnet34, self).__init__()
        model = models.resnet34(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv1.stride = (1, 1)
        self.model = model
        fc_features = model.fc.in_features
        self.classifier = Classifier(fc_features, num_classes, dropout)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class resnet50(nn.Module):
    def __init__(self, num_classes, dropout=0.5, stride=2):
        super(resnet50, self).__init__()
        model = models.resnet50(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        self.model = model
        fc_features = model.fc.in_features
        self.classifier = Classifier(fc_features, num_classes, dropout)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


class resnet101(nn.Module):
    def __init__(self, num_classes, dropout=0.5, stride=2):
        super(resnet101, self).__init__()
        model = models.resnet101(pretrained=True)
        if stride == 1:
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        self.model = model
        fc_features = model.fc.in_features
        self.classifier = Classifier(fc_features, num_classes, dropout)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


def test_resnet50():
    model = resnet50(num_classes=20)
    print(model)


def test_resnet101():
    model = resnet101(num_classes=20)
    print(model)


def test_resnet34():
    a = torch.randn(12, 3, 32, 32)
    model = resnet34(num_classes=20)
    print(model)
    b = model(a)

def test_resnet18():
    a = torch.randn(12, 3, 32, 32)
    model = resnet18(num_classes=20)
    print(model)
    b = model(a)


if __name__ == '__main__':
    test_resnet50()
    #test_resnet101()
    #test_resnet34()
    #test_resnet18()