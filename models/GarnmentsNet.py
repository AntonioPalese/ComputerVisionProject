from torchvision import models
import torch.nn as nn
import torch


class MobileNetConvLayer(nn.Module):
    def __init__(self):
        super(MobileNetConvLayer, self).__init__()

        model_ = models.mobilenet_v2(pretrained=True)
        model_.eval()

        self.features = nn.Sequential(
            # stop at last conv
            *list(model_.features.children())
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ClassNetGarnments(nn.Module):
    def __init__(self):
        super(ClassNetGarnments, self).__init__()
        """

        :int num_classes: number of classes
        """
        self.ConvLayer = MobileNetConvLayer()
        self.ConvLayer.eval()

        self.avg = nn.Sequential(
            nn.AvgPool2d(3, stride=2, ceil_mode=False, count_include_pad=True,
                               divisor_override=None)
        )

        self.Linear = nn.Sequential(
            nn.Linear(1280 * 3 * 3, 4096, bias=False),
            nn.BatchNorm1d(num_features=4096),
            nn.ReLU6(inplace=True),
            nn.Linear(4096, 512, bias=False)
        )

    def forward_img(self, X):
        X = self.ConvLayer(X)
        X = self.avg(X)
        #print('shape dopo average pooling ', X.shape)
        X = torch.flatten(X, 1)
        X = self.Linear(X)
        return X

    def forward(self, X1, X2):
        X1 = self.forward_img(X1)
        X2 = self.forward_img(X2)
        return X1, X2