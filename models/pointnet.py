# pointnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(2, 1)  # -> (B, 3, N)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # (B, 256, N)

        x = torch.max(x, dim=2)[0]  # (B, 256)
        # x = F.normalize(x, p=2, dim=1)

        return x


