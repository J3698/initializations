import torch
import torch.nn as nn
import tqdm

import os
import sys
import traceback
sys.path.append(".")

from sklearn.manifold import MDS
import matplotlib.pyplot as plt

from models.vgg import VGG19, VGG19BN
from models.mlp import MLP, MLPBN
from dataloaders import create_CIFAR10_dataloaders, create_librispeech_dataloaders
import init_info
from initializers.common import get_random_conv_inputs, get_random_linear_inputs,\
                                get_batch_of_all_inputs, put_all_batches_through_layer

cuda = torch.cuda.is_available()
NUM_WORKERS = os.cpu_count() if cuda else 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64 if cuda else 1

def main():
    train_loader, val_loader = create_librispeech_dataloaders(15, 256)
    visualize_all_mlp_inits(train_loader, val_loader, [MLP])
    # train_loader, val_loader = create_librispeech_dataloaders(15, 512)
    # visualize_all_mlp_inits(train_loader, val_loader, [MLP])

    # train_loader, val_loader = create_CIFAR10_dataloaders(BATCH_SIZE)
    # visualize_all_cnn_inits(train_loader, val_loader, [VGG19])



def visualize_all_mlp_inits(train_loader, val_loader, models):
    for init, info in init_info.init_types.items():
        for nonlinearity in init_info.nonlinearity_types:
            if "include_nonlinearities" in info and\
               nonlinearity not in info["include_nonlinearities"]:
                break

            for model_type in models:
                test_name = f"{init.__name__}-{model_type.__name__}-" + \
                            f"{nonlinearity.__name__}"

                try:
                    model = model_type(nonlinearity = nonlinearity)

                    init(model, train_loader, show_progress = True)
                    visualize_weights_mlp(model, train_loader, "./MDS", "init", test_name)
                except Exception:
                    print(traceback.format_exc())
                    print("failed:", test_name)


def visualize_weights_mlp(model, train_loader, folder, vis_name, test_name):
    if torch.cuda.is_available():
        model.cuda()

    last_layers_output = get_batch_of_all_inputs(train_loader)
    for i, l in tqdm.tqdm(enumerate(model.layers), total = len(model.layers)):
        with torch.no_grad():
            if isinstance(l, nn.Linear):
                points = l.weight.shape[0] * 2
                inps = get_random_linear_inputs(last_layers_output, l, points)
                if torch.cuda.is_available():
                    inps = inps.cuda()

                lw = l.weight
                m = inps.mean(dim = 0)
                n = torch.norm(inps - m, dim = 1).mean()
                lw -= lw.mean(dim = 0)
                lw /= (torch.norm(lw, dim = 1).mean() + 1e-15)
                lw = lw * n + m
                inps = torch.cat((inps, lw), dim = 0)

                embedding = MDS(n_components = 2, n_jobs = -1)
                inps_transformed = embedding.fit_transform(inps.cpu().numpy())
                plt.scatter(inps_transformed[:points, 0], inps_transformed[:points, 1], color = 'blue')
                plt.scatter(inps_transformed[points:, 0], inps_transformed[points:, 1], color = 'red')
                plt.savefig(os.path.join(folder, f'MDS-{test_name}-{vis_name}-{i}-.png'))
                plt.close()

            if torch.cuda.is_available():
                last_layers_output = last_layers_output.cuda()
            last_layers_output = put_all_batches_through_layer(l, last_layers_output)


def visualize_all_cnn_inits(train_loader, folder, val_loader, models):
    for init, info in init_info.init_types.items():
        for nonlinearity in init_info.nonlinearity_types:
            if "include_nonlinearities" in info and\
               nonlinearity not in info["include_nonlinearities"]:
                break

            for model_type in models:
                test_name = f"{init.__name__}-{model_type.__name__}-" + \
                            f"{nonlinearity.__name__}"

                try:
                    model = model_type(nonlinearity = nonlinearity)
                    init(model, train_loader, show_progress = True)
                    model.cuda()
                    last_layers_output = get_batch_of_all_inputs(train_loader)
                    for i, l in enumerate(model.layers):
                        with torch.no_grad():
                            if isinstance(l, nn.Conv2d):
                                points = 5000
                                inps = get_random_conv_inputs(last_layers_output, l, points)
                                inps = torch.cat((inps.cuda(), l.weight.reshape((l.weight.shape[0], -1))))

                                embedding = MDS(n_components = 2, n_jobs = -1)
                                inps_transformed = embedding.fit_transform(inps.cpu().numpy())
                                plt.scatter(inps_transformed[:points, 0], inps_transformed[:points, 1], color = 'blue')
                                plt.scatter(inps_transformed[points:, 0], inps_transformed[points:, 1], color = 'red')
                                plt.savefig(os.path.join(folder, f'MDS-{i}-{test_name}.png'))
                                plt.close()
                            last_layers_output = put_all_batches_through_layer(l, last_layers_output)

                except Exception:
                    print("failed:", test_name)
                    print(traceback.format_exc())


if __name__ == "__main__":
    main()


