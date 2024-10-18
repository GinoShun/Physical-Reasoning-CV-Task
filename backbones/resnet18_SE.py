import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Block for ResNet-18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# ResNet-18 Model
class CNN(nn.Module):
    def __init__(self, n_classes=6, singleTask=False):
        super(CNN, self).__init__()
        self.singleTask = singleTask

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = 512 * BasicBlock.expansion
        self.fc_main = nn.Linear(in_features, n_classes)

        # supplementary tasks
        self.fc_shapeset = nn.Linear(in_features, 2)
        self.fc_type = nn.Linear(in_features, 2)
        self.fc_total_height = nn.Linear(in_features, n_classes-1)  # classification
        self.fc_instability = nn.Linear(in_features, 3)
        self.fc_cam_angle = nn.Linear(in_features, 2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        if self.singleTask:
            return self.fc_main(x)
        else:
            out_main = self.fc_main(x)
            out_shapeset = self.fc_shapeset(x)
            out_type = self.fc_type(x)
            out_total_height = self.fc_total_height(x)
            out_instability = self.fc_instability(x)
            out_cam_angle = self.fc_cam_angle(x)
            return out_main, out_shapeset, out_type, out_total_height, out_instability, out_cam_angle
