import torch.nn as nn
from torch.hub import load_state_dict_from_url
from utils import LossMetrics as mmd
import torch
from utils.config import bottle_neck


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x


class DSAN(nn.Module):

    def __init__(self, num_classes=65, model='alexnet'):
        super(DSAN, self).__init__()
        self.feature_layers = alexnet(pretrained=True)

        if bottle_neck:
            # change here
            self.bottle = nn.Sequential(
                nn.Linear(9216, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 256),
            )
            self.cls_fc = nn.Linear(256, num_classes)
        else:

            self.cls_fc = nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.Linear(4096, num_classes)
            )

    def forward(self, source, target, s_label):

        source = self.feature_layers(source)

        if bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)

        if self.training ==True:
            target = self.feature_layers(target)
            if bottle_neck:
                target = self.bottle(target)
            t_label = self.cls_fc(target)
            loss = mmd.lmmd(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
        else:
            loss = 0

        return s_pred, loss

    def predict(self, x):

        features = self.feature_layers(x)
        if bottle_neck:
            features = self.bottle(features)
        clf = self.cls_fc(features)

        return clf


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
