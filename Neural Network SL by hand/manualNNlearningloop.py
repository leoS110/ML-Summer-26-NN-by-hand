# 

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
#from matplotlib.animation import FFMpegWriter
from matplotlib.animation import PillowWriter
from pathlib import Path

#generate training data set, D_TR
#f = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(8 * np.pi * x)
f = lambda x: np.sin(1 * np.pi * x) + 0.001 * np.sin(8 * np.pi * x)
#f = lambda x: 2*x + 3

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
#print(x_TR)
#sprint(dataset_size)



#manual NN setup, define and initialise weights, and define all key functions (h(x), L(h), loss gradients etc)
#defining key parameters, this is hard coded in later at this stage, next step is to make code versatile
n_layers = 3
n_width = 10

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
#define key dimensions:
d_x_aug = dimension_x + 1
length_w = n_width
U_1_rows = n_width
U_1_columns = n_width
U_2_rows = n_width
U_2_columns = d_x_aug
params = ModelParams(
    w= rng.random((length_w, 1)) - 0.5,      
    U_1= rng.random((U_1_rows, U_1_columns)) - 0.5, 
    U_2= rng.random((U_2_rows, U_2_columns)) - 0.5        
)
#print("w:", params.w)
#print("U_1:", params.U_1)
#print("U_2:", params.U_2)


#define non_linear activation function
def sigma(A): 
    #sigmaval = np.tanh(A) 
    sigmaval = np.maximum(A, 0) #ReLu
    return sigmaval #vector x long

#define h(x) eval
def h(x, params):
    phi_d = sigma(params.U_2@x)
    phi = sigma(params.U_1@phi_d)
    hval = ((params.w).T)@phi
    return hval

#plot h(x) estimate
def ploth(params, h_line):
    x_grid = np.linspace(-1.5, 1.5, 50)
    h_array = np.empty((50,1))
    for i in range(50):
        xval_aug = np.array([[x_grid[i]], [1.0]])
        h_array[i] = (h(xval_aug, params))

    if h_line is not None:
        h_line.remove()

    h_line, = plt.plot(x_grid, h_array)
    return h_line
    #plt.shw()  
    
#define loss function of h(x) across D_TR - squared loss, no regularizer. Batch size selects the proportion of D_TR used to evaluate loss, for SGD
def Loss(params, x_TR_batch, y_TR_batch): 
    loss = 0.0

    batch_length = len(x_TR_batch)

    for i in range(batch_length):
        xval_aug = np.array([[x_TR_batch[i]], [1.0]])
        loss += (h(xval_aug, params) - y_TR_batch[i])**2
    loss = loss/batch_length
    return loss
#print("loss = ", Loss(params, x_TR, y_TR))


#evaluate manual NN loss gradients via finite differencing, batch size there for stochastic gradient descent
def lossgrads_finitediff(x_TR, y_TR, dimension_x, params, dataset_size, batch_proportion):
    eps = 1e-4

    #define batch of training points to use for gradient calc, for stochastic gradient descent
    batch_length = int(np.floor(dataset_size*batch_proportion))
    if batch_length < 1:
        batch_length = 1
    
    random_indices = rng.choice(len(x_TR), size=batch_length, replace=False)
    x_TR_batch = x_TR[random_indices]
    y_TR_batch = y_TR[random_indices]
    
    dLdw_fdiff = np.zeros((length_w, 1), dtype=np.float64)
    for i in range(length_w):
        
        w_lower = params.w.copy() #make a copy otherwise changes to w_lower would also change w
        w_lower[i] = w_lower[i] - eps
        params_lower = ModelParams(w=w_lower, U_1=params.U_1, U_2=params.U_2)

        w_upper = params.w.copy() #make a copy otherwise changes to w_lower would also change w
        w_upper[i] = w_upper[i] + eps
        params_upper = ModelParams(w=w_upper, U_1=params.U_1, U_2=params.U_2)

        dLdw_fdiff[i] = (Loss(params_upper, x_TR_batch, y_TR_batch) - Loss(params_lower, x_TR_batch, y_TR_batch))/(2.0*eps)

    #print("dLdw_fdiff", dLdw_fdiff)
    dLdU1_fdiff = np.zeros((U_1_rows, U_1_columns), dtype=np.float64)

    for column in range(U_1_columns):
        for row in range(U_1_rows):

            U_1_lower = params.U_1.copy()
            U_1_lower[row, column] = U_1_lower[row, column] - eps
            params_lower = ModelParams(w=params.w, U_1=U_1_lower, U_2=params.U_2)

            U_1_upper = params.U_1.copy()
            U_1_upper[row, column] = U_1_upper[row, column] + eps
            params_upper = ModelParams(w=params.w, U_1=U_1_upper, U_2=params.U_2)

            dLdU1_fdiff[row, column] = ((Loss(params_upper, x_TR_batch, y_TR_batch) - Loss(params_lower, x_TR_batch, y_TR_batch))/(2.0*eps)).item()

    #print("dLdU1_fdiff", dLdU1_fdiff)
    dLdU2_fdiff = np.zeros((U_2_rows, U_2_columns), dtype=np.float64)

    for column in range(U_2_columns):
        for row in range(U_2_rows):

            U_2_lower = params.U_2.copy()
            U_2_lower[row, column] = U_2_lower[row, column] - eps
            params_lower = ModelParams(w=params.w, U_1=params.U_1, U_2=U_2_lower)

            U_2_upper = params.U_2.copy()
            U_2_upper[row, column] = U_2_upper[row, column] + eps
            params_upper = ModelParams(w=params.w, U_1=params.U_1, U_2=U_2_upper)

            dLdU2_fdiff[row, column] = ((Loss(params_upper, x_TR_batch, y_TR_batch) - Loss(params_lower, x_TR_batch, y_TR_batch))/(2.0*eps)).item()

    #print("dLdU2_fdiff", dLdU2_fdiff)
    return dLdw_fdiff, dLdU1_fdiff, dLdU2_fdiff

#function to update model parameters based off of loss gradient 
def updateparams(params, dLdw_fdiff, dLdU1_fdiff, dLdU2_fdiff, learning_rate):
    params.w -= learning_rate * dLdw_fdiff
    params.U_1 -= learning_rate * dLdU1_fdiff
    params.U_2 -= learning_rate * dLdU2_fdiff


# learning loop and logic
loop_n = 2000
loss_array = np.empty((loop_n, 1), dtype=np.float64)
batch_proportion = 0.015
learning_rate = 5e-2

#plt.ion() 
#fig = plt.figure() 
#fig, (hfit, losscurve) = plt.subplots(1, 2, figsize=(11, 4.5))
#hfit.plt.scatter(x_TR, y_TR, label="Training Data", color="blue")

x_grid = np.linspace(-1.5, 1.5, 200)
X_grid_aug = np.vstack((x_grid, np.ones_like(x_grid)))
plt.ion()
fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))
out_path = Path(r"C:\Users\leosu\ML26\Neural Network SL by hand\training.gif")
writer = PillowWriter(fps=20)

with writer.saving(fig, str(out_path), dpi=80):
    for loop_i in range(loop_n):
        #evaluate L(h)
        lossval = Loss(params, x_TR, y_TR)
        #print(lossval)
        loss_array[loop_i] = lossval

        #evaluate loss gradients 
        [dLdw_fdiff, dLdU1_fdiff, dLdU2_fdiff] = lossgrads_finitediff(x_TR, y_TR, dimension_x, params, dataset_size, batch_proportion)

        #update params according to stochastic gradient descent optimisation
        updateparams(params, dLdw_fdiff, dLdU1_fdiff, dLdU2_fdiff, learning_rate)
        print((loop_i/loop_n) * 100, "%")

        if loop_i % 10 == 0:
            #h_line = ploth(params, h_line)
            #plt.pause(0.01) # Briefly pause to allow the plot to draw on screen

            ax_fit.cla()
            ax_fit.scatter(x_TR, y_TR, s=8)
            ax_fit.plot(x_grid, h(X_grid_aug, params).ravel(), "r")
            ax_fit.set_ylim(-2, 2)             # must re-set: cla() wipes it

            ax_loss.cla()
            ax_loss.semilogy(loss_array[:loop_i + 1])

            plt.pause(0.01)
            writer.grab_frame()

plt.ioff()
plt.figure()
print(loss_array)
#plt.show()

plt.figure()
plt.plot(np.arange(1, loop_n + 1), loss_array)
plt.show()
