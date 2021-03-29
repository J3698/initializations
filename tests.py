import torch.nn as nn

from models.mlp import MLP
from models.vgg import VGG19

from initializers.pca import initialize_pca
from initializers.basic import initialize_he, initialize_orthogonal, \
        initialize_tanh_lecun_uniform, initialize_tanh_xavier_uniform

from util.signal_propagation_plots import signal_propagation_plot



def check_all_mlp_inits_work(train_loader, val_loader):
    check_mlp_relu_inits_work(train_loader, val_loader)
    check_mlp_tanh_inits_work(train_loader, val_loader)


def check_all_vgg_inits_work(train_loader, val_loader):
    check_vgg_relu_inits_work(train_loader, val_loader)
    check_vgg_tanh_inits_work(train_loader, val_loader)


def check_mlp_relu_inits_work(train_loader, val_loader):
    model_relu = MLP(num_classes = 346)
    in_shape = (5, model_relu.in_feats)

    initialize_he(model_relu, True)
    signal_propagation_plot(model_relu, in_shape, "He ReLU")

    initialize_pca(model_relu, train_loader, verbose = True, show_progress = True)
    signal_propagation_plot(model_relu, in_shape, "PCA ReLU")

    initialize_pca(model_relu, train_loader, zca = True, verbose = True,\
                                                            show_progress= True)
    signal_propagation_plot(model_relu, in_shape, "ZCA ReLU")

    initialize_orthogonal(model_relu, True)
    signal_propagation_plot(model_relu, in_shape, "Orthogonal")


def check_mlp_tanh_inits_work(train_loader, val_loader):
    model_tanh = MLP19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_pca(model_tanh, train_loader)
    initialize_pca(model_tanh, train_loader, zca = True)
    initialize_tanh_lecun_uniform(model_tanh)
    initialize_orthogonal(model_tanh)
    initialize_tanh_xavier_uniform(model_tanh)


def check_vgg_relu_inits_work(train_loader, val_loader):
    model_relu = VGG19(num_classes = 10)
    initialize_he(model_relu)
    initialize_pca(model_relu, train_loader)
    initialize_pca(model_relu, train_loader, zca = True)
    initialize_orthogonal(model_relu)


def check_vgg_tanh_inits_work(train_loader, val_loader):
    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_pca(model_tanh, train_loader)
    initialize_pca(model_tanh, train_loader, zca = True)
    initialize_tanh_lecun_uniform(model_tanh)
    initialize_orthogonal(model_tanh)
    initialize_tanh_xavier_uniform(model_tanh)


