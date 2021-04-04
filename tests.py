import torch.nn as nn

from models.mlp import MLP
from models.vgg import VGG19, VGG19BN

from initializers.pca import *
from initializers.basic import initialize_he, initialize_orthogonal, \
        initialize_tanh_lecun_uniform, initialize_tanh_xavier_uniform

from util.signal_propagation_plots import signal_propagation_plot



def check_all_mlp_inits_work(train_loader, val_loader):
    check_mlp_relu_inits_work(train_loader, val_loader)
    check_mlp_tanh_inits_work(train_loader, val_loader)




def check_mlp_relu_inits_work(train_loader, val_loader):
    model_relu = MLP(num_classes = 346)
    in_shape = (5, model_relu.in_feats)

    initialize_he(model_relu, True)
    signal_propagation_plot(model_relu, in_shape, "He ReLU")

    initialize_pca(model_relu, train_loader, verbose = True)

    initialize_zca(model_relu, train_loader, verbose = True)
    signal_propagation_plot(model_relu, in_shape, "ZCA ReLU")

    initialize_orthogonal(model_relu, True)
    signal_propagation_plot(model_relu, in_shape, "Orthogonal")


def check_all_vgg_inits_work(train_loader, val_loader):
    check_vgg_relu_inits_work(train_loader, val_loader)
    check_vgg_tanh_inits_work(train_loader, val_loader)
    check_vgg_bn_inits_work(train_loader, val_loader)




def check_mlp_tanh_inits_work(train_loader, val_loader):
    model_tanh = MLP19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_pca(model_tanh, train_loader)
    initialize_zca(model_tanh, train_loader)
    initialize_tanh_lecun_uniform(model_tanh)
    initialize_orthogonal(model_tanh)
    initialize_tanh_xavier_uniform(model_tanh)


def check_vgg_relu_inits_work(train_loader, val_loader):
    model_relu = VGG19(num_classes = 10)
    initialize_he(model_relu)
    print("Inited he")
    initialize_pca(model_relu, train_loader)
    print("Inited pca relu")
    initialize_zca(model_relu, train_loader)
    print("Inited zca relu")
    initialize_lsuv_kmeans(model_relu, train_loader, show_progress = True)
    print("Inited kmeans")
    initialize_lsuv_pca(model_relu, train_loader)
    initialize_lsuv_zca(model_relu, train_loader)
    initialize_orthogonal(model_relu)
    print("Inited orth relu")
    initialize_lsuv_random_samples(model_relu, train_loader)
    print("Inited lsuv relu")


def check_vgg_tanh_inits_work(train_loader, val_loader):
    model_tanh = VGG19(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_lsuv_kmeans(model_relu, train_loader, show_progress = True)
    initialize_pca(model_tanh, train_loader)
    print("Inited pca tanh")
    initialize_zca(model_tanh, train_loader)
    print("Inited zca tanh")
    initialize_tanh_lecun_uniform(model_tanh)
    print("Inited tanh lecun")
    initialize_orthogonal(model_tanh)
    print("Inited orth tanh")
    initialize_tanh_xavier_uniform(model_tanh)
    print("Inited tanh xavier")
    initialize_lsuv_random_samples(model_tanh, train_loader)
    print("Inited lsuv tanh")


def check_vgg_bn_inits_work(train_loader, val_loader):
    model_relu = VGG19BN(num_classes = 10)
    initialize_kmeans(model_relu, train_loader)
    initialize_he(model_relu)
    print("Inited he relu")
    initialize_pca(model_relu, train_loader)
    print("Inited pca relu")
    initialize_zca(model_relu, train_loader)
    print("Inited zca relu")
    initialize_orthogonal(model_relu, train_loader)
    print("Inited orth relu")
    initialize_random_samples(model_relu, train_loader)
    print("Inited random relu")

    model_tanh = VGG19BN(num_classes = 10, nonlinearity = nn.Tanh)
    initialize_kmeans(model_tanh, train_loader)
    initialize_pca(model_tanh, train_loader)
    print("Inited pca tanh (bn)")
    initialize_zca(model_tanh, train_loader)
    print("Inited zca tanh (bn)")
    initialize_tanh_lecun_uniform(model_tanh, train_loader)
    print("Inited lecun tanh (bn)")
    initialize_orthogonal(model_tanh, train_loader)
    print("Inited orth tanh (bn)")
    initialize_tanh_xavier_uniform(model_tanh, train_loader)
    print("Inited xavier tanh (bn)")
    initialize_random_samples(model_tanh, train_loader)
    print("Inited random tanh (bn)")


