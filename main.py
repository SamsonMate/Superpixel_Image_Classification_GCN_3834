"""
Please view README.md for details
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib as plt
import networkx as nx

#
# Import our models dataset and segment it accordingly
#
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

classes = trainset.classes

def encode_with_superpixels(img) -> nx.Graph:
    """
    Encodes a given xxx into a graph where:
    Nodes = Superpixels (Almost like upscaled pixels).
    Edges = Relations between neighbouring superpixels.
    """
    pass

def compute_hyperparameters() -> list:
    """
    Computes the appropriate hyperparameters of our model
    """
    pass

def training_iteration():
    """
    Computes one iteration of training on the model
    """
    pass

def compute_performance_metrics():
    pass

def main():
    encode_with_superpixels()
    hps = compute_hyperparameters()
    training_delta = hps[0]
    num_epochs = hps[1]
    for i in range(num_epochs):
        training_iteration()
    compute_performance_metrics()

if __name__ == "main":
    main()