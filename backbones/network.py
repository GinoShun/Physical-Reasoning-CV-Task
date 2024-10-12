import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Define CNN with SE layers integrated
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Define the convolutional layers with SE layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.se1 = SElayer(channel=32)  # SE layer after first convolution
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.se2 = SElayer(channel=64)  # SE layer after second convolution
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.se3 = SElayer(channel=128)  # SE layer after third convolution

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 37 * 37, 512)  # This size needs to be adjusted based on input size
        self.fc2 = nn.Linear(512, 1)  # Single output for regression (height prediction)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional layers with SE layers and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.se1(x)  # Apply SE Layer 1

        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.se2(x)  # Apply SE Layer 2

        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool
        x = self.se3(x)  # Apply SE Layer 3

        # Print the shape after the final pooling layer to confirm the size
        print(f"Shape after conv layers and pooling: {x.shape}")

        # Dynamically flatten the feature maps for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Output layer (predicting height)

        return x
