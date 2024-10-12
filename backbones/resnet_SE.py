import torch
import torch.nn as nn
import torch.nn.functional as F


# Define Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        
        # Pointwise convolution (1x1 conv)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
# Define Squeeze-and-Excitation (SE) Layer
class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):  # Use a standard reduction of 16
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction), 
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Squeeze operation
        y = self.fc1(y).view(b, c, 1, 1)  # Excitation operation
        return x * y  # Recalibrate the input x with the excitation

# Define a Residual Block with SE layers
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualBlock, self).__init__()
        
        # Use Depthwise Separable Convolution in the residual block
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SElayer(out_channels, reduction)  # SE Layer after second convolution

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply SE Layer
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out

# Define the deeper CNN with residual blocks and SE layers integrated
class CNN(nn.Module):
    def __init__(self, n_classes=6):  # Add n_classes as a parameter
        super(CNN, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # Larger kernel size
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks with SE layers
        self.layer1 = self._make_layer(64, 128, stride=2)  # Reduce spatial dimensions
        self.layer2 = self._make_layer(128, 256, stride=2)  # Further reduce spatial dimensions
        self.layer3 = self._make_layer(256, 512, stride=2)  # Deeper features
        self.layer4 = self._make_layer(512, 1024, stride=2)
        self.layer5 = self._make_layer(1024, 2048, stride=2)
        self.layer6 = self._make_layer(2048, 4096, stride=2)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, in_channels, out_channels, stride):
        return ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x):
        # Initial conv -> batch norm -> relu -> pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Pass through residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        # print(x.shape)

        # Global average pooling to reduce feature map to 1x1
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer for n_classes

        return x
