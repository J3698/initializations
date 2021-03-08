import torch
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self, num_classes, nonlinearity = nn.ReLU):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding = 1), nonlinearity(),
            nn.Conv2d(64, 64, 3, padding = 1), nonlinearity(),

            nn.Conv2d(64, 128, 3, 2, 1), nonlinearity(),
            nn.Conv2d(128, 128, 3, 1, 1), nonlinearity(),

            nn.Conv2d(128, 256, 3, 2, 1), nonlinearity(),
            nn.Conv2d(256, 256, 3, 1, 1), nonlinearity(),
            nn.Conv2d(256, 256, 3, 1, 1), nonlinearity(),
            nn.Conv2d(256, 256, 3, 1, 1), nonlinearity(),

            nn.Conv2d(256, 512, 3, 2, 1), nonlinearity(),
            nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),
            nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),
            nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),

            nn.Conv2d(512, 512, 3, 2, 1), nonlinearity(),
            nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),
            nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),

            nn.Conv2d(512, 512, 3, 2, 1), nonlinearity(),
            nn.Conv2d(512, num_classes, 1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)

