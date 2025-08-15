import torch

# MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def normalize(x):
    # x in [0,1] -> normalized
    return (x - MNIST_MEAN) / MNIST_STD

def denormalize(x):
    # normalized -> [0,1] for display
    return torch.clamp(x * MNIST_STD + MNIST_MEAN, 0.0, 1.0)

def clamp_normalized(x):
    lower = (0.0 - MNIST_MEAN) / MNIST_STD
    upper = (1.0 - MNIST_MEAN) / MNIST_STD
    return torch.clamp(x, lower, upper)