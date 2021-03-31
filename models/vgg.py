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
            nn.Conv2d(256, 256, 3, 2, 1), nonlinearity(),
            nn.Conv2d(256, 256, 3, 2, 1), nonlinearity(),
            nn.Conv2d(256, 256, 3, 1, 1), nonlinearity(),
            #nn.Conv2d(256, 256, 3, 1, 1), nonlinearity(),
            #nn.Conv2d(256, 256, 3, 1, 1), nonlinearity(),

            nn.Conv2d(256, 512, 3, 1, 1), nonlinearity(),
            nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),
            nn.Conv2d(512, 10, 3, 1, 1),
            nn.AvgPool2d(2),
            #nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),
            #nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),

            #nn.Conv2d(512, 512, 3, 2, 1), nn.BatchNorm2d(512), nonlinearity(),
            #nn.Conv2d(512, 512, 3, 1, 1), nonlinearity(),
            #nn.Conv2d(512, 512, 3, 2, 1), nn.BatchNorm2d(512), nonlinearity(),

            #nn.Conv2d(512, 512, 3, 2, 1), nn.BatchNorm2d(512), nonlinearity(),
            #nn.Conv2d(512, num_classes, 1),
            nn.Flatten()
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class VGG19BN(nn.Module):
    def __init__(self, num_classes, nonlinearity = nn.ReLU):
        super().__init__()

        layers = []
        to_copy = VGG19(num_classes, nonlinearity)
        for layer in to_copy.layers:
            layers.append(layer)
            if isinstance(layer, nn.Conv2d):
                layers.append(nn.BatchNorm2d(layer.out_channels))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        out = self.layers(x)
        return out


