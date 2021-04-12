import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm


def main():
    train_loader, val_loader = create_librispeech_dataloaders(15)

    iter(train_loader)
    print(train_loader.dataset.Y[0].shape)
    print(np.max(train_loader.dataset.Y[0]))
    print(np.max([np.max(i) for i in train_loader.dataset.Y]))


def create_librispeech_dataloaders(context, batch_size = 32, num_workers = 4):
    """
    train_dataset = LibriDataset("data/hw1p2/train.npy",
                                 "data/hw1p2/train_labels.npy", context)
    train_loader_args = dict(shuffle = True, batch_size = batch_size, num_workers=num_workers, pin_memory = True)
    train_loader = DataLoader(train_dataset, **train_loader_args)
    """

    val_dataset = LibriDataset("data/hw1p2/dev.npy",
                               "data/hw1p2/dev_labels.npy", context)
    val_loader_args = dict(shuffle = False, batch_size = batch_size, num_workers=num_workers, pin_memory = True)
    val_loader = DataLoader(val_dataset, **val_loader_args)

    return val_loader, val_loader # train_loader, val_loader


class LibriDataset(Dataset):
    def __init__(self, train_file, train_labels_file, context, transform = None, length = None):
        self.X = np.load(train_file, allow_pickle = True)
        self.Y = np.load(train_labels_file, allow_pickle = True) if train_labels_file else None
        self.transform = transform
        self.length = length
        self.context = context

        for i in range(len(self.X)):
            self.X[i] = np.pad(self.X[i], ((self.context, self.context), (0, 0)), mode = 'constant')
        self.lens = np.array([len(i) - 2 * self.context for i in self.X]).cumsum()


    def __len__(self):
        if self.length is None:
            return self.lens[-1]
        return self.length


    def __getitem__(self,index):
        idx_utter = np.searchsorted(self.lens, index + 1)
        offset = 0 if idx_utter == 0 else self.lens[idx_utter - 1]
        index_in_utter = index - offset
        X = self.X[idx_utter][index_in_utter: index_in_utter + 2 * self.context + 1]
        if self.Y is not None:
            Y = self.Y[idx_utter][index_in_utter]
        if self.transform:
            X, Y = self.transform((X, Y))
        X = torch.tensor(X).float().reshape(-1)
        if self.Y is not None:
            Y = torch.tensor(Y).long()
            return X, Y
        return X


if __name__ == "__main__":
    main()
