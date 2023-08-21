import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_Like(nn.Module):
    """
        Model based on the VGG architecture: https://arxiv.org/pdf/1409.1556.pdf
        NOTE: Does not work properly yet!
    """
    def __init__(self, num_classes, dropout_rate):
        super(VGG_Like, self).__init__()
        kernel_size = 3
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(16, 64, kernel_size)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2)

        # self.conv3_1 = nn.Conv2d(64, 128, kernel_size)
        # self.conv3_2 = nn.Conv2d(128, 128, kernel_size)
        # self.pool3 = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64, 32)
        self.dropout1 = nn.Dropout1d(dropout_rate)

        self.fc2 = nn.Linear(32, 8)
        self.dropout2 = nn.Dropout1d(dropout_rate)

        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(F.relu(self.conv1_2(x)))

        x = F.relu(self.conv2_1(x))
        x = self.pool2(F.relu(self.conv2_2(x)))

        # x = F.relu(self.conv3_1(x))
        # x = self.pool3(F.relu(self.conv3_2(x)))


        x = self.gap(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # x = F.relu(self.fc2(x))
        # x = self.dropout2(x)

        x = self.fc3(x)

        return x


class LeNet_Like(nn.Module):
    def __init__(self, num_classes):
        super(LeNet_Like, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*13*13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*13*13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet_Like1D(nn.Module):
    def __init__(self, num_classes, num_input_features):
        super(LeNet_Like1D, self).__init__()
        self.conv1 = nn.Conv1d(num_input_features, 6, 5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(16*9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

