import torch.nn as nn
import torchvision.models as models
from models.units import Classifier


class densenet121(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(densenet121, self).__init__()
        model = models.densenet121(pretrained=True)
        model.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = model.features
        self.classifier = Classifier(1024, num_classes, dropout)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1))
        features, x = self.classifier(x)
        return features, x


if __name__ == '__main__':
    model = densenet121(751)
    print(model)