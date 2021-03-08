from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

class ArchitectureNotSupportedException(Exception):
    def __init__(self, model):
        self.model_string = repr(model)

    def __str__(self):
        return self.model_string


class LayerNotSupportedException(Exception):
    def __init__(self, layer):
        self.str = str(layer)

    def __str__(self):
        return self.str


def check_architecture_is_sequential(model):
    assert model.layers is not None
    assert isinstance(model.layers, nn.Sequential)


def check_all_layers_supported(model: nn.Sequential,
                             supported_layers: Tuple[type, ...]):
    for layer in model:
        if not isinstance(layer, supported_layers):
            raise LayerNotSupportedException(layer)


def get_first_batch_inputs(train_loader):
    data, labels = next(iter(train_loader))
    return data


def get_batch_of_all_inputs(train_loader: DataLoader, show_progress = False) -> torch.Tensor:
    torch.multiprocessing.set_sharing_strategy('file_system')
    if show_progress:
        train_loader = tqdm.tqdm(train_loader)

    data = [x[None, ...] for i, (x, y) in enumerate(train_loader)
                             if i < len(train_loader) / 20]
    return torch.cat(data, dim =  0)

