#manually coded neural network with forward pass(h(x) eval) and cost gradient compared between chain rule and finite differences
#check theory understanding and refresh python
#NN structure hard coded in rather than adaptable in the interest of time

import numpy as np

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

#generate random NN weights
w = rng.random(dimension_x+1)
U_1 = rng.random((n_width, (dimension_x+1)))
U_2 = rng.random((n_width, (dimension_x+1)))
print("w:", w)
print("U_1:", U_1)
print("U_2:", U_2)

#augment x states
x_aug_vectors = np.column_stack((x_array, np.ones_like(x_array)))
print("x_aug_vectors:", x_aug_vectors)

#define non_linear activation function
def sigma(A) #using tanh for now, need a smooth function for gradient check accuracy
    sigmaval = np.tanh(A) #what is the dimension of A?? how do I make act elementwise??
    return sigmaval

#define h(x) eval
def h(x)



    return hval

#define loss function



#evaluate h(x) manually


#loss function definition



#evaluate manual NN gradients via chain rule



#evaluate manual NN gradients via finite differencing





