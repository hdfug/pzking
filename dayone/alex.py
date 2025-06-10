import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.Linear(2048, 2048),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        y = self.model(x)
        return y


if __name__ == '__main__':
    alexnet = AlexNet()
    x = torch.randn(1, 3, 224, 224)
    y = alexnet(x)
    print("Output shape:", y.shape)