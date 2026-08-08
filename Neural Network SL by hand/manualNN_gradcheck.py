#manually coded neural network with forward pass(h(x) eval) and cost gradients found in 3 ways: finite difference, manual chain rule backprop algebra, backprop algorithm (reverse mode AD) (last skipped for now but will be returned to)
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

#ploth(params)
    
#define loss function of h(x) across D_TR - squared loss, no regularizer
def Loss(params, x_array, y_array): 
    loss = 0.0
    for i in range(dataset_size):
        xval_aug = np.array([[x_array[i]], [1.0]])
        loss += (h(xval_aug, params) - y_array[i])**2
    loss = loss/dataset_size
    return loss

print("loss = ", Loss(params, x_array, y_array))


#Find loss gradients w.r.t. parameters:

#1) evaluate manual NN gradients via finite differencing
eps = 1e-12

dLdw_fdiff = np.zeros((2*dimension_x, 1), dtype=np.float64)
for i in range(2*dimension_x):
    
    w_lower = params.w.copy() #make a copy otherwise changes to w_lower would also change w
    w_lower[i] = w_lower[i] - eps
    params_lower = set_params(w_lower, U_1, U_2)

    w_upper = params.w.copy() #make a copy otherwise changes to w_lower would also change w
    w_upper[i] = w_upper[i] + eps
    params_upper = set_params(w_upper, U_1, U_2)

    dLdw_fdiff[i] = (Loss(params_upper, x_array, y_array) - Loss(params_lower, x_array, y_array))/(2.0*eps)

print("dLdw_fdiff", dLdw_fdiff)


dLdU1_fdiff = np.zeros((n_width, (dimension_x+1)), dtype=np.float64)

for column in range((dimension_x+1)):
    for row in range(n_width):

        U_1_lower = params.U_1.copy()
        U_1_lower[row, column] = U_1_lower[row, column] - eps
        params_lower = set_params(w, U_1_lower, U_2)

        U_1_upper = params.U_1.copy()
        U_1_upper[row, column] = U_1_upper[row, column] + eps
        params_upper = set_params(w, U_1_upper, U_2)

        dLdU1_fdiff[row, column] = ((Loss(params_upper, x_array, y_array) - Loss(params_lower, x_array, y_array))/(2.0*eps)).item()

print("dLdU1_fdiff", dLdU1_fdiff)

dLdU2_fdiff = np.zeros((n_width, (dimension_x+1)), dtype=np.float64)

for column in range((dimension_x+1)):
    for row in range(n_width):

        U_2_lower = params.U_2.copy()
        U_2_lower[row, column] = U_2_lower[row, column] - eps
        params_lower = set_params(w, U_1, U_2_lower)

        U_2_upper = params.U_2.copy()
        U_2_upper[row, column] = U_2_upper[row, column] + eps
        params_upper = set_params(w, U_1, U_2_upper)

        dLdU2_fdiff[row, column] = ((Loss(params_upper, x_array, y_array) - Loss(params_lower, x_array, y_array))/(2.0*eps)).item()

print("dLdU2_fdiff", dLdU2_fdiff)


#2) evaluate manual NN gradients via chain rule algebraically


#3) evaluate NN gradients backprop algorithmically (reverse mode auto diff)- mirroring actual modern process
#skipped for this project in interest of time but will return once theory is necessary








