# 

import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
from dataclasses import dataclass

#generate training data set, D_TR
f = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(8 * np.pi * x)

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

[x_TR, y_TR, dataset_size] = make_data()
plt.scatter(x_TR, y_TR)
plt.show()
print(x_TR)
print(dataset_size)



#manual NN setup, define and initialise weights, and define all key functions (h(x), L(h), loss gradients etc)
#defining key parameters, this is hard coded in later at this stage, next step is to make code versatile
n_layers = 3
n_width = 2

#augment x states
# Create an array of 1.0s with the same length as your original array
ones = np.ones_like(x_TR, dtype=float)
x_aug_vectors = np.vstack((x_TR, ones))
print("x_aug_vectors:", x_aug_vectors)

#generate random NN weights and store in dataclass
@dataclass
class ModelParams:
    w: np.ndarray
    U_1: np.ndarray
    U_2: np.ndarray
rng= np.random.default_rng()
params = ModelParams(
    w= rng.random((2*dimension_x, 1)),       # initializing a 10x1 column vector of zeros
    U_1= rng.random((n_width, (dimension_x+1))), # initializing a 5x5 matrix of random numbers
    U_2= rng.random((n_width, (dimension_x+1)))        # initializing a 5x1 column vector of ones
)

print("w:", params.w)
print("U_1:", params.U_1)
print("U_2:", params.U_2)


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

#plot h(x) estimate
def ploth(params):
    x_grid = np.linspace(-1.5, 1.5, 50)
    h_array = []
    for i in range(50):
        xval_aug = np.array([[x_grid[i]], [1.0]])
        h_array.append(h(xval_aug, params))
    plt.scatter(x_TR, y_TR)
    plt.plot(x_grid, h_array)
    plt.show()  
    
#define loss function of h(x) across D_TR - squared loss, no regularizer. Batch size selects the proportion of D_TR used to evaluate loss, for SGD
def Loss(params, x_TR, y_TR, dataset_size, batch_proportion): 
    loss = 0.0

    batch_length = int(np.floor(dataset_size*batch_proportion))
    if batch_length < 1:
        batch_length = 1

    random_indices = rng.choice(len(x_TR), size=batch_length, replace=False)
    x_TR_batch = x_TR[random_indices]
    y_TR_batch = y_TR[random_indices]

    for i in range(batch_length):
        xval_aug = np.array([[x_TR_batch[i]], [1.0]])
        loss += (h(xval_aug, params) - y_TR[i])**2
    loss = loss/batch_length
    return loss
#print("loss = ", Loss(params, x_TR, y_TR))


#evaluate manual NN loss gradients via finite differencing, batch size there for stochastic gradient descent
def lossgrads_finitediff(x_TR, y_TR, dimension_x, params, dataset_size, batch_proportion):
    eps = 1e-5

    dLdw_fdiff = np.zeros((2*dimension_x, 1), dtype=np.float64)
    for i in range(2*dimension_x):
        
        w_lower = params.w.copy() #make a copy otherwise changes to w_lower would also change w
        w_lower[i] = w_lower[i] - eps
        params_lower = ModelParams(w=w_lower, U_1=params.U_1, U_2=params.U_2)

        w_upper = params.w.copy() #make a copy otherwise changes to w_lower would also change w
        w_upper[i] = w_upper[i] + eps
        params_upper = ModelParams(w=w_upper, U_1=params.U_1, U_2=params.U_2)

        dLdw_fdiff[i] = (Loss(params_upper, x_TR, y_TR, dataset_size, batch_proportion) - Loss(params_lower, x_TR, y_TR, dataset_size, batch_proportion))/(2.0*eps)

    #print("dLdw_fdiff", dLdw_fdiff)
    dLdU1_fdiff = np.zeros((n_width, (dimension_x+1)), dtype=np.float64)

    for column in range((dimension_x+1)):
        for row in range(n_width):

            U_1_lower = params.U_1.copy()
            U_1_lower[row, column] = U_1_lower[row, column] - eps
            params_lower = ModelParams(w=params.w, U_1=U_1_lower, U_2=params.U_2)

            U_1_upper = params.U_1.copy()
            U_1_upper[row, column] = U_1_upper[row, column] + eps
            params_upper = ModelParams(w=params.w, U_1=U_1_upper, U_2=params.U_2)

            dLdU1_fdiff[row, column] = ((Loss(params_upper, x_TR, y_TR, dataset_size, batch_proportion) - Loss(params_lower, x_TR, y_TR, dataset_size, batch_proportion))/(2.0*eps)).item()

    #print("dLdU1_fdiff", dLdU1_fdiff)
    dLdU2_fdiff = np.zeros((n_width, (dimension_x+1)), dtype=np.float64)

    for column in range((dimension_x+1)):
        for row in range(n_width):

            U_2_lower = params.U_2.copy()
            U_2_lower[row, column] = U_2_lower[row, column] - eps
            params_lower = ModelParams(w=params.w, U_1=params.U_1, U_2=U_2_lower)

            U_2_upper = params.U_2.copy()
            U_2_upper[row, column] = U_2_upper[row, column] + eps
            params_upper = ModelParams(w=params.w, U_1=params.U_1, U_2=U_2_upper)

            dLdU2_fdiff[row, column] = ((Loss(params_upper, x_TR, y_TR, dataset_size, batch_proportion) - Loss(params_lower, x_TR, y_TR, dataset_size, batch_proportion))/(2.0*eps)).item()

    #print("dLdU2_fdiff", dLdU2_fdiff)
    return dLdw_fdiff, dLdU1_fdiff, dLdU2_fdiff

#function to update model parameters based off of loss gradient 
def updateparams(params, dLdw_fdiff, dLdU1_fdiff, dLdU2_fdiff, learning_rate):
    params.w -= learning_rate * dLdw_fdiff
    params.U_1 -= learning_rate * dLdU1_fdiff
    params.U_2 -= learning_rate * dLdU2_fdiff


# learning loop and logic
loop_n = 1000
loss_array = np.empty((loop_n, 1), dtype=np.float64)
batch_proportion = 0.5
learning_rate = 1e-4
for loop_i in range(loop_n):

    #evaluate L(h)
    lossval = Loss(params, x_TR, y_TR, dataset_size, batch_proportion)
    #print(lossval)
    loss_array[loop_i] = lossval

    #evaluate loss gradients 
    [dLdw_fdiff, dLdU1_fdiff, dLdU2_fdiff] = lossgrads_finitediff(x_TR, y_TR, dimension_x, params, dataset_size, batch_proportion)

    #update params according to stochastic gradient descent optimisation
    updateparams(params, dLdw_fdiff, dLdU1_fdiff, dLdU2_fdiff, learning_rate)

plt.plot(np.arange(1, loop_n + 1), loss_array)
plt.show()
print(loss_array)
#ploth(params) needs debugging
#plt.show()
