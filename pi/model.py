import torch
import torch.nn as nn


class PiModel(nn.Module):

    def __init__(self, num_classes=10):
        super(PiModel, self).__init__()

        self.conv1a = nn.Conv2d(3, 128, kernel_size=3, padding='same')
        self.conv1b = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv1c = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.5)

        self.conv2a = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.conv2b = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.conv2c = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.5)

        self.conv3a = nn.Conv2d(256, 512, kernel_size=3, padding='valid')
        self.conv3b = nn.Conv2d(512, 256, kernel_size=1)
        self.conv3c = nn.Conv2d(256, 128, kernel_size=1)

        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.dense = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.lrelu(self.conv1a(x))
        x = self.lrelu(self.conv1b(x))
        x = self.lrelu(self.conv1c(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.lrelu(self.conv2a(x))
        x = self.lrelu(self.conv2b(x))
        x = self.lrelu(self.conv2c(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.lrelu(self.conv3a(x))
        x = self.lrelu(self.conv3b(x))
        x = self.lrelu(self.conv3c(x))

        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.softmax(x)

        return x