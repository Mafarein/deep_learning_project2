import torch
import torch.nn as nn
import torch.nn.functional as F

# Test cnn from chatgpt xd

class SimpleSpeechCommandModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSpeechCommandModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32 * 1000, 128)  # after 2 pools: 16000 / 4 / 4 = 1000
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch_size, 1, 16000)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
