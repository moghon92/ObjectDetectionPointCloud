# Test Model: Used for verifying Algorithm 1 below is working
from torch import nn
import torch

class TestModel(nn.Module):
    def __init__(self, num_points):
        super(TestModel, self).__init__()
        # Check input size 66145 (this should be the channels for the point cloud data)
        self.conv = nn.Conv1d(num_points, 576, 4)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(576, 9)

    def forward(self, x):
      out = self.conv(x)
      out = self.relu(out)
      out = out.view(1, -1)
      out = self.Linear2(out)
      return out

