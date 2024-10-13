import torch
import torch.nn as nn
import torch.nn.functional as F


# Define CNN
class CNN(nn.Module):
    def __init__(self, n_classes=6):  # Add n_classes as a parameter
        super(CNN, self).__init__()
        
        # Define the convolutional layers with SE layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, n_classes)  # Output n_classes for classification

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional layers with SE layers and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool

        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool

        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool

        # Flatten the feature maps for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Output layer (logits for each class)

        return x 