import torch
import torch.nn as nn
import sys
sys.path.append(".")
from librispeech_dataloaders import create_librispeech_dataloaders


def main():
    train_loader, _ = create_librispeech_dataloaders(15, batch_size = 2)
    model = MLP(context = 15).cuda()

    x = next(iter(train_loader))[0].cuda()
    y = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")



class MLP(nn.Module):
    def __init__(self, num_classes = 346, context = 15):
        super().__init__()

        self.in_feats = (2 * context + 1) * 13

        self.layers = nn.Sequential(
            nn.Linear(self.in_feats, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        print(x.shape)
        return self.layers(x)


class MLPBN(nn.Module):
    def __init__(self, num_classes, nonlinearity = nn.ReLU):
        super().__init__()

        layers = []
        to_copy = MLP(num_classes, nonlinearity)
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
