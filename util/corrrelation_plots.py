import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from models.mlp import SmallMLP, MediumMLP
from initializers.common import check_architecture_is_sequential
from dataloaders import create_MNIST_dataloaders


def main():
    model = MediumMLP()
    model.train()
    plotter = CorrelationPlotter(model, "testm/test")
    plotter.record_datapoint()

    train_loader, val_loader = create_MNIST_dataloaders(True, 1024)
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    criterion = nn.CrossEntropyLoss()

    for e in range(5):
        for i, (x, y) in tqdm.tqdm(enumerate(train_loader),\
                                   total = len(train_loader)):
            plotter.record_datapoint()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    plotter.plot(False)


class CorrelationPlotter:
    @torch.no_grad()
    def __init__(self, model, name):
        check_architecture_is_sequential(model)
        self.model = model
        self.name = name
        self.avg_correlations_through_time = []

        affine_types = (nn.Conv2d, nn.Linear)
        affine_layers = [i for i in model.layers if isinstance(i, affine_types)]
        self.affine_weights_2d = [l.weight.flatten(1) for l in affine_layers]


    @torch.no_grad()
    def plot(self, negative_log = True):
        data = np.array(self.avg_correlations_through_time).T
        for i, layer_correlations in enumerate(data):
            if negative_log:
                layer_correlations = -np.log(layer_correlations)
            plt.plot(layer_correlations)
            plt.suptitle(f"{self.name}: Avg. Correlations (Layer {i})")
            plt.savefig(f"{self.name}-layer-{i}.png")
            plt.close()


    @torch.no_grad()
    def record_datapoint(self):
        curr_avg_correlation = [self._avg_correlation(i) \
                                 for i in self.affine_weights_2d]
        self.avg_correlations_through_time.append(curr_avg_correlation)


    def _avg_correlation(self, weight_2d):
        neurons, feats = weight_2d.shape

        dots = weight_2d @ weight_2d.T
        assert dots.shape == (neurons, neurons)

        norms = torch.norm(weight_2d, dim = 1)
        norm_products = norms[:, None] * norms
        assert norm_products.shape == (neurons, neurons)

        dots /= norm_products
        dots[torch.arange(neurons), torch.arange(neurons)] = 0

        return dots.sum() / (neurons * (neurons - 1))


if __name__ == "__main__":
    main()


