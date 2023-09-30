from torch import nn
import torch.nn.functional as F

class ShapeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ShapeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)  # Adjust the dimensions based on your input image size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x