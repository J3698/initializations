import torch
import torch.nn as nn
import sys
sys.path.append(".")
from dataloaders import create_librispeech_dataloaders


def main():
    train_loader, _ = create_librispeech_dataloaders(15, batch_size = 2)
    model = MLP(context = 15).cuda()

    x = next(iter(train_loader))[0].cuda()
    y = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")


class MLP(nn.Module):
    def __init__(self, num_classes = 346, context = 15, nonlinearity = nn.ReLU):
        super().__init__()

        self.in_feats = (2 * context + 1) * 13

        self.layers = nn.Sequential(
            nn.Linear(self.in_feats, 2048),
            nonlinearity(),
            nn.Linear(2048, 2048),
            nonlinearity(),
            nn.Linear(2048, 1024),
            nonlinearity(),
            nn.Linear(1024, 1024),
            nonlinearity(),
            nn.Linear(1024, 512),
            nonlinearity(),
            nn.Linear(512, 512),
            nonlinearity(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class SmallMLP(nn.Module):
    def __init__(self, num_classes = 10, inputs = 28 ** 2, nonlinearity = nn.ReLU):
        super().__init__()

        self.in_feats = inputs

        self.layers = nn.Sequential(
            nn.Linear(self.in_feats, 1024),
            nonlinearity(),
            nn.Linear(1024, 512),
            nonlinearity(),
            nn.Linear(512, 256),
            nonlinearity(),
            nn.Linear(256, 128),
            nonlinearity(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class MediumMLP(nn.Module):
    def __init__(self, num_classes = 10, inputs = 28 ** 2, nonlinearity = nn.ReLU):
        super().__init__()

        self.in_feats = inputs

        self.layers = nn.Sequential(
            nn.Linear(self.in_feats, 2048),
            nonlinearity(),
            nn.Linear(2048, 1024),
            nonlinearity(),
            nn.Linear(1024, 512),
            nonlinearity(),
            nn.Linear(512, 256),
            nonlinearity(),
            nn.Linear(256, 128),
            nonlinearity(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)



class MLPBN(nn.Module):
    def __init__(self, num_classes = 346, inputs = 15, nonlinearity = nn.ReLU,
                 base_mlp = MLP):
        super().__init__()

        layers = []
        to_copy = base_mlp(num_classes, inputs, nonlinearity)
        for layer in to_copy.layers:
            layers.append(layer)
            if isinstance(layer, nn.Linear):
                layers.append(nn.BatchNorm1d(layer.out_features))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        out = self.layers(x)
        return out


if __name__ == "__main__":
    main()
