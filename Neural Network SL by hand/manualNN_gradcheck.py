#manually coded neural network with forward pass(h(x) eval) and cost gradient compared between chain rule and finite differences
#check theory understanding and refresh python
#NN structure hard coded in rather than adaptable in the interest of time

import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

#defining key parameters, this is hard coded in later at this stage, next step is to make code versatile
n_layers = 3
n_width = 2

#generate random training data set D_TR, assuming x 1D and y 1D
dataset_size = 10
dimension_x = 1
rng= np.random.default_rng()
x_array = rng.random(dataset_size) #is this an array or another data type? 
y_array = rng.random(dataset_size)
print("x:", x_array)
print("y:", y_array)


#augment x states
# Create an array of 1.0s with the same length as your original array
ones = np.ones_like(x_array, dtype=float)
x_aug_vectors = np.vstack((x_array, ones))
print("x_aug_vectors:", x_aug_vectors)

#generate random NN weights
w = rng.random(2*dimension_x)
U_1 = rng.random((n_width, (dimension_x+1)))
U_2 = rng.random((n_width, (dimension_x+1)))
print("w:", w)
print("U_1:", U_1)
print("U_2:", U_2)

class Params(NamedTuple): #define a named tuple structure to store weight in 
    w: np.ndarray   # (2x,)
    U_1: np.ndarray   # (n_width,x+1)
    U_2: np.ndarray   # (n_width,x+1)

def set_params(w_in, U_1_in, U_2_in): #construct the weight tuple
    return Params(
        w = w_in,
        U_1= U_1_in,
        U_2= U_2_in,
    )

params = set_params(w, U_1, U_2)


#define non_linear activation function
def sigma(A): #using tanh for now, need a smooth function for gradient check accuracy
    sigmaval = np.tanh(A) 
    return sigmaval #vector x long

#define h(x) eval
def h(x, params):
    phi_d = sigma(params.U_2@x)
    phi = sigma(params.U_1@phi_d)
    hval = ((params.w).T)@phi
    return hval

#evaluate h(x) manually
hval = h(x_aug_vectors[:,0], params)

#plot h(x) estimate
def ploth(params):
    x_grid = np.linspace(0.0, 1.0, 50)
    h_array = []
    for i in range(50):
        xval_aug = np.array([[x_grid[i]], [1.0]])
        h_array.append(h(xval_aug, params))
    plt.scatter(x_array, y_array)
    plt.plot(x_grid, h_array)
    plt.show()  

ploth(params)
    

#define loss function






#loss function definition



#evaluate manual NN gradients via chain rule



#evaluate manual NN gradients via finite differencing





