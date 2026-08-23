#

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn #specifically the neural network construction package
DEVICE = torch.device("cpu")

#generate training data in the same way as in manual NN case, into numpy arrays

#f = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(8 * np.pi * x)
f = lambda x: np.sin(1.5 * np.pi * x) + 0.4 * np.sin(8 * np.pi * x)
#f = lambda x: 2*x + 3
#f = lambda x: np.sin(np.pi*(x+1)*(1 + (x+1)))

dimension_x = 1
def make_data(n=256, noise=0.05, gap_start= -0.3, gap_end = 0.0, seed=0):
    x = np.linspace(-1, 1, num=n)
    rng = np.random.default_rng(seed)
    y = f(x) + rng.normal(0, noise, x.shape);
    #make a gap 
    mask = (x < gap_start) | (x > gap_end)
    x_gapremoved = x[mask]
    y_gapremoved = y[mask]
    dataset_size = len(x_gapremoved)
    return x_gapremoved, y_gapremoved, dataset_size

[x_TR_numpy, y_TR_numpy, dataset_size] = make_data()

#put data into tensor form for pytorch
x_TR = torch.tensor(x_TR_numpy) 
y_TR = torch.tensor(y_TR_numpy) 
#print(x_TR)

