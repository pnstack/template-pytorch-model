from torch import nn
import torch.nn.functional as F
# Ảnh gốc có kích thước 128x128x3
class ShapeClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=128):
        super(ShapeClassifier, self).__init__()
        # Layer 1: Convolutional layer with 3 input channels (RGB) and 16 output channels, using a 3x3 kernel and padding of 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # ra 128x128x16
        
        # Layer 2: Max pooling layer with a 2x2 kernel and stride of 2 to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # ra 64x64x16
        
        # Layer 3: Fully connected layer with input size 16 * 64 * 64 (depends on the input image size) and output size 128
        self.fc1 = nn.Linear(16 * 64 * 64, hidden_size)
        
        # Layer 4: Fully connected layer with input size 128 and output size num_classes
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Forward pass through the network
        
        # Apply convolution, activation function (ReLU), and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Adjust the dimensions for the fully connected layer
        x = x.view(-1, 16 * 64 * 64)
        
        # Apply activation function (ReLU) to the first fully connected layer
        x = F.relu(self.fc1(x))
        
        # Output layer without activation function (applied later during loss computation)
        x = self.fc2(x)
        
        return x
