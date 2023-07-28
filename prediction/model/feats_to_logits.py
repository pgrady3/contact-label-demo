import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from prediction.model.gradient_reversal_layer import GradientReverseLayerF


def get_features_to_class(in_features, out_classes):
    layers = []
    layers.append(MLP(in_features, [512, 128, out_classes], nn.BatchNorm1d, dropout=0.2, inplace=False))
    return nn.Sequential(*layers)


def get_raw_compression():
    layers = []
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(Flatten())
    return nn.Sequential(*layers)


def get_conv_compression():
    layers = []
    in_channels = 2048
    out_channels = 2048
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(Flatten())
    return nn.Sequential(*layers)


def get_dann_model():
    layers = []
    in_channels = 2048
    out_channels = 2048
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(Flatten())
    layers.append(MLP(2048, [512, 2], nn.BatchNorm1d, dropout=0.2, inplace=False))
    return nn.Sequential(*layers)


class AllLayerDannModel(nn.Module):
    def __init__(self, num_out_logits):
        super(AllLayerDannModel, self).__init__()
        in_channels = 2048
        out_channels = 2048

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = MLP(3904, [512, num_out_logits], nn.BatchNorm1d, dropout=0.2, inplace=False)

    def forward(self, feature_list, alpha):

        f1 = GradientReverseLayerF.apply(feature_list[-1], alpha)
        f2 = GradientReverseLayerF.apply(feature_list[-2], alpha)
        f3 = GradientReverseLayerF.apply(feature_list[-3], alpha)
        f4 = GradientReverseLayerF.apply(feature_list[-4], alpha)
        f5 = GradientReverseLayerF.apply(feature_list[-5], alpha)

        f1 = self.conv1(f1)
        f1 = self.batch_norm1(f1)
        f1 = self.relu(f1)

        f1 = self.avgpool(f1)
        f1 = f1.reshape(f1.shape[0], -1)

        f2 = self.avgpool(f2).reshape(f1.shape[0], -1)
        f3 = self.avgpool(f3).reshape(f1.shape[0], -1)
        f4 = self.avgpool(f4).reshape(f1.shape[0], -1)
        f5 = self.avgpool(f5).reshape(f1.shape[0], -1)

        x = torch.cat((f1, f2, f3, f4, f5), dim=1)
        x = self.mlp(x)

        # x = torch.sigmoid(x)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BottleneckClassifierModel(nn.Module):
    def __init__(self, num_out_logits):
        super(BottleneckClassifierModel, self).__init__()
        in_channels = 2048
        out_channels = 2048

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = MLP(2048, [512, num_out_logits], nn.BatchNorm1d, dropout=0.2, inplace=False)

    def forward(self, feature_list):
        x = feature_list[-1]

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)

        x = torch.sigmoid(x)

        return x