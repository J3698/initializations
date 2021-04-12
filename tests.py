import torch.nn as nn

from models.mlp import MLP
from models.vgg import VGG19, VGG19BN

from initializers.pca import *
from initializers.basic import initialize_he, initialize_orthogonal, \
        initialize_tanh_lecun_uniform, initialize_tanh_xavier_uniform

from util.signal_propagation_plots import signal_propagation_plot

import init_info


def check_all_inits_work(train_loader, val_loader, models):
    for init, info in init_info.init_types.items():
        for nonlinearity in init_info.nonlinearity_types:
            if "include_nonlinearities" in info and\
               nonlinearity not in info["include_nonlinearities"]:
                break

            for model_type in models:
                print(f"Testing {init.__name__} init on model "
                      f"{model_type.__name__} with "
                      f"nonlinearity {nonlinearity.__name__}")

                model = model_type(nonlinearity = nonlinearity)
                init(model, train_loader, show_progress = True)
    print("Finished testing")

