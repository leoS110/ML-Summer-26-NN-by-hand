#manually coded neural network with forward pass(h(x) eval) and cost gradient compared between hand derived backprop chain rule and finite differences
#main goal: refresh python and check neural network theory - backprop doesn't used cached values to evaluate gradients as it should, rather it re-evaluates, to save time in implementation and move onto RL

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.animation import PillowWriter   
from pathlib import Path

#generate training data set, D_TR
#f = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(8 * np.pi * x)
f = lambda x: np.sin(1.5 * np.pi * x) + 0.1 * np.sin(8 * np.pi * x)
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

[x_TR, y_TR, dataset_size] = make_data()
#plt.scatter(x_TR, y_TR)
#plt.show()

#manual NN setup, define and initialise weights, and define all key functions (h(x), L(h), loss gradients etc)
#defining key parameters, this is hard coded in later at this stage, next step is to make code versatile
#2 layers
n_width = 25

#augment x states
# Create an array of 1.0s with the same length as your original array
ones = np.ones_like(x_TR, dtype=float)
x_aug_vectors = np.vstack((x_TR, ones))

#generate random NN weights and store in dataclass
@dataclass
class ModelParams:
    w: np.ndarray
    U: np.ndarray
rng= np.random.default_rng()
#define key dimensions:
d_x_aug = dimension_x + 1
length_w = n_width
U_rows = n_width
U_columns = d_x_aug
params = ModelParams(
    w= rng.random((length_w)) - 0.5,      
    U= rng.random((U_rows, U_columns)) - 0.5,   
)
#print("w:", params.w)
#print("U_1:", params.U_1)
#print("U_2:", params.U_2)


#define non_linear activation function
def sigma(A): 
    #sigmaval = np.tanh(A) 
    sigmaval = np.maximum(A, 0) #ReLu
    return sigmaval #vector x long

def sigma_dot(A):
    #sigmadotval = 1 - np.tanh(A)**2 #for tanh
    #for ReLu:
    #sigmadotval = np.empty((len(A)))
    #for i in range(len(A)):
    #    val = A[i]
    #    if val > 0:
    #        sigmadotval[i] = 1.0
    #    else:
    #        sigmadotval[i] = 0.0
    sigmadotval = (A > 0).astype(int)
    return sigmadotval

#define h(x) eval
def h(x, params):
    phi = sigma(params.U@x)
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
        xval_aug = np.array([x_TR_batch[i], 1.0])
        loss += (float(h(xval_aug, params).item()) - float(y_TR_batch[i]))**2
    loss = loss/batch_length
    return loss
#print("loss = ", Loss(params, x_TR, y_TR))


#evaluate manual NN loss gradients via finite differencing, batch size there for stochastic gradient descent
def lossgrads_finitediff(x_TR, y_TR, dimension_x, params, dataset_size):
    eps = 1e-4

    
    dLdw_fdiff = np.zeros(length_w, dtype=np.float64)
    for i in range(length_w):
        
        w_lower = params.w.copy() #make a copy otherwise changes to w_lower would also change w
        w_lower[i] = w_lower[i] - eps
        params_lower = ModelParams(w=w_lower, U=params.U)

        w_upper = params.w.copy() #make a copy otherwise changes to w_lower would also change w
        w_upper[i] = w_upper[i] + eps
        params_upper = ModelParams(w=w_upper, U=params.U)

        dLdw_fdiff[i] = (Loss(params_upper, x_TR, y_TR) - Loss(params_lower, x_TR, y_TR))/(2.0*eps)

    #print("dLdw_fdiff", dLdw_fdiff)
    dLdU_fdiff = np.zeros((U_rows, U_columns), dtype=np.float64)

    for column in range(U_columns):
        for row in range(U_rows):

            U_lower = params.U.copy()
            U_lower[row, column] = U_lower[row, column] - eps
            params_lower = ModelParams(w=params.w, U=U_lower)

            U_upper = params.U.copy()
            U_upper[row, column] = U_upper[row, column] + eps
            params_upper = ModelParams(w=params.w, U=U_upper)

            dLdU_fdiff[row, column] = ((Loss(params_upper, x_TR, y_TR) - Loss(params_lower, x_TR, y_TR))/(2.0*eps))


    #print("dLdU2_fdiff", dLdU2_fdiff)
    return dLdw_fdiff, dLdU_fdiff


def lossgrads_backprop(x_TR, y_TR, params, dataset_size):

    #gradient 1: dL/dw: #h and phi evals should really be cached from loss evaluation but recalculating for now
    vector_sum = np.zeros(length_w, dtype=np.float64)
    for i in range(dataset_size):
        xval_aug = np.array([x_TR[i], 1.0])
    
        vector_sum += (h(xval_aug, params) - y_TR[i]) * (sigma(params.U@xval_aug))
    dLdw_backprop = ((2/dataset_size)*vector_sum)

    #gradient 2: dL/dU - split into chain rule. Backprop logic for further layers would extend this chain rule and reuse terms
    dLdU_backprop = np.zeros((U_rows, U_columns), dtype=np.float64)
    #dL/dU[q,r] = sum over training data ( sum over p ( dL/da_i[p] * da_i[p]/dU[q,r]) )

    #evaluate dL/da values, one vector for each training datapoint
    dLda_matrix = np.zeros((n_width, dataset_size), dtype=np.float64)
    dh_dphi = (params.w).T

    for i in range(dataset_size): 
        xval_aug = np.array([x_TR[i], 1.0])

        dl_dh_i = (2/dataset_size)*(h(xval_aug, params) - y_TR[i])

        a = (params.U)@xval_aug
        dphi_da = np.diag(sigma_dot(a))

        dLda_i = dl_dh_i*(dh_dphi@dphi_da)

        dLda_matrix[:,i] = dLda_i

    #evaluate chain rule, (dL/da)(da/dU)
    dLdU_backprop = np.zeros((U_rows, U_columns), dtype=np.float64)
    for column in range(U_columns):
        for row in range(U_rows):
            sum = 0.0
            for i in range(dataset_size):
                dLda_vector = dLda_matrix[:, i]
                xval_aug = np.array([x_TR[i], 1.0])

                sum += dLda_vector[row]*xval_aug[column]
 
            dLdU_backprop[row, column] = sum
    return dLdw_backprop, dLdU_backprop

#function to update model parameters based off of loss gradient 
def updateparams(params, dLdw_fdiff, dLdU_fdiff, learning_rate):
    params.w -= learning_rate * dLdw_fdiff
    params.U -= learning_rate * dLdU_fdiff





# learning loop and logic
loop_n = 2500
batch_proportion = 0.015
learning_rate = 6e-2

loss_array = np.empty((loop_n, 1), dtype=np.float64)

#plotting setup
SAVE_GIF = False
out_path = Path(r"C:\Users\leosu\ML26\Neural Network SL by hand")

x_grid = np.linspace(-1.5, 1.5, 200)
X_grid_aug = np.vstack((x_grid, np.ones_like(x_grid)))
plt.ion()
fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))
writer = PillowWriter(fps=20)
if SAVE_GIF:
    writer.setup(fig, str(out_path), dpi=80)


for loop_i in range(loop_n):

    #evaluate L(h)
    lossval = Loss(params, x_TR, y_TR)
    #print(lossval)
    loss_array[loop_i] = lossval

    #generate batch of training data to evaluate loss gradient with for stochastic gradient descent 
    batch_length = int(np.floor(dataset_size*batch_proportion))
    if batch_length < 1:
        batch_length = 1
        
    random_indices = rng.choice(len(x_TR), size=batch_length, replace=False)
    x_TR_batch = x_TR[random_indices]
    y_TR_batch = y_TR[random_indices]

    #evaluate loss gradients 
    [dLdw, dLdU] = lossgrads_backprop(x_TR_batch , y_TR_batch, params, batch_length)
    #[dLdw, dLdU] = lossgrads_finitediff(x_TR_batch, y_TR_batch, dimension_x, params, dataset_size)

    #update params according to stochastic gradient descent optimisation
    updateparams(params, dLdw, dLdU, learning_rate)

    completion_percentage = (loop_i/loop_n) * 100
    if completion_percentage % 1.0 == 0:
        print(completion_percentage, "%")

    if loop_i % 10 == 0: #plotting
        ax_fit.cla()
        ax_fit.scatter(x_TR, y_TR, s=8)
        ax_fit.plot(x_grid, h(X_grid_aug, params).ravel(), "r")
        ax_fit.set_ylim(-2, 2)             # must re-set: cla() wipes it

        ax_loss.cla()
        ax_loss.semilogy(loss_array[:loop_i + 1])

        plt.pause(0.01)
        if SAVE_GIF:
            writer.grab_frame()

if SAVE_GIF:
    writer.finish()

plt.ioff()
print(loss_array)










