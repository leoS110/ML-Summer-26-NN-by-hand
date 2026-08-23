#learning to use pytorch

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn #specifically the neural network construction package
#DEVICE = torch.device("cpu") #currently keeping everything on CPU, will learn to move everything to GPU for more intense training in next project

from dataclasses import dataclass, field  

from matplotlib.animation import PillowWriter   
from pathlib import Path

#generate training data in the same way as in manual NN case, into numpy arrays

#f = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(8 * np.pi * x)
f = lambda x: np.sin(1.5 * np.pi * x) + 0.4 * np.sin(8 * np.pi * x)
#f = lambda x: 2*x + 3
#f = lambda x: np.sin(np.pi*(x+1)*(1 + (x+1)))

dimension_x = 1
dataset_size = 256
def make_data(noise=0.05, gap_start= -0.3, gap_end = 0.0, seed=0):
    dataset_size = 256
    x = np.linspace(-1, 1, num=dataset_size)
    rng = np.random.default_rng(seed)
    y = f(x) + rng.normal(0, noise, x.shape);
    #make a gap 
    mask = (x < gap_start) | (x > gap_end)
    x_gapremoved = x[mask]
    y_gapremoved = y[mask]
    dataset_size = len(x_gapremoved)
    return x_gapremoved, y_gapremoved, dataset_size

[x_TR_numpy, y_TR_numpy, dataset_size] = make_data()
x_TR_numpy = x_TR_numpy.reshape(-1, 1)
y_TR_numpy = y_TR_numpy.reshape(-1, 1)

#put data into tensor form for pytorch
x_TR_numpy = x_TR_numpy.astype(np.float32) #to align with pytorch default parameters being float32 not 64
y_TR_numpy = y_TR_numpy.astype(np.float32) 
x_TR = torch.as_tensor(x_TR_numpy) 
y_TR = torch.as_tensor(y_TR_numpy) 
#print(x_TR)


#define parameter values to be used:
@dataclass                                        # turns the class body into a typed config object
class Config:
    #input and output data:
    dimension_in: int = 1                                
    dimension_out: int = 1 

    #NN parameters:                              
    n_width: int = 50                       
    n_layers: int = 4 #hidden layers, not including output                       

    #optimisation parameters:
    lr: float = 6e-2                             
    #weight_decay: float = 1e-2                  
    #max_grad_norm: float = 1.0                   
    #batch_size: int = 64                          
    batch_proportion: float = 0.6

    #training loop paramaters
    training_steps: int = 20000

    #validation:
    validation_frac: float = 0.2                         # fraction of data held out for validation

    #random seed:
    seed: int = 0                                 # RNG seed for init, shuffling and the split      

    #devide allocation:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  <- later
    device: torch.device = field(default_factory=lambda: torch.device("cpu")) 

params = Config()


#define module: defines forward pass
class pytorchNN(nn.Module):                             
    def __init__(self, dimension_in, dimension_out, n_width, n_layers, act=nn.ReLU):   
        super().__init__()                        # must run first: creates the parameter/module registries

        layers, d = [], dimension_in                      # layers accumulates modules; d tracks current width
        
        for _ in range(n_layers):                    #loop through each hidden layer
            layers += [nn.Linear(d, n_width), act()]   #linear mapping and then non-linear activation function
            d = n_width                            #reset the length of the tensor being passed at the input to the next layer

        layers.append(nn.Linear(d, dimension_out))        # output layer, no activation: targets are unbounded
        self.net = nn.Sequential(*layers)         # * unpacks the list into positional args; assigning registers it
 
    def forward(self, x):                         # runs on every batch; autograd records it as it executes
        return self.net(x)                        # Sequential applies each child module in order

model = pytorchNN(params.dimension_in, params.dimension_out, params.n_width, params.n_layers)


#loss function
loss_fn = nn.MSELoss()   

#optimiser
optimizer = torch.optim.SGD(model.parameters(), lr=params.lr) 


#training loop:

loss_array = np.empty((params.training_steps, 1), dtype=np.float64) 
torch.manual_seed(params.seed)

#animation 
SAVE_GIF = False
out_path = Path(r"C:\Users\leosu\Github local\NN by hand\pytorchtraining1.gif")
x_grid = np.linspace(-1.5, 1.5, 200)
plt.ion()
fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))
writer = PillowWriter(fps=20)
if SAVE_GIF:
    writer.setup(fig, str(out_path), dpi=80)


for loop_i in range(params.training_steps):

    optimizer.zero_grad(set_to_none=True) #zero the gradients from the last step so that they don't accumulate

    #evaluate loss for loss curve, on full training set not on batch used for backprop, so don't use autograd to unnecessarily store gradient info
    with torch.no_grad():                        
        lossval = loss_fn(model(x_TR), y_TR).item()
        loss_array[loop_i] = lossval

    #for SGD, generate a batch from training data:
    idx = torch.randperm(dataset_size)[:(int(params.batch_proportion * dataset_size))]              # random batch, sampled without replacement
    x_TR_batch, y_TR_batch = x_TR[idx], y_TR[idx]

    #h(x), forward pass, autograd caching
    h_val = model(x_TR_batch)  

    #evaluate loss on batch
    loss = loss_fn(h_val, y_TR_batch) 

    #uses backprop + autograd to generate loss gradients (fills .grad graph)
    loss.backward()

    optimizer.step()                            
 

    if loop_i % 100 == 0: #plotting
            with torch.no_grad():
                ax_fit.cla()
                ax_fit.scatter(x_TR, y_TR, s=8)
                x_grid = torch.linspace(-1.5, 1.5, 200).unsqueeze(1)
                ax_fit.plot(x_grid.numpy(), model(x_grid).numpy(), "r")
                ax_fit.set_ylim(-2, 2)            
        
                ax_loss.cla()
                ax_loss.semilogy(loss_array[:loop_i + 1])
        
                plt.pause(0.005)
                if SAVE_GIF:
                    writer.grab_frame()



if SAVE_GIF:
    writer.finish()

plt.ioff()

#with torch.no_grad():
#    x_grid = torch.linspace(-1.5, 1.5, 200).unsqueeze(1)
#    h_grid = model(x_grid)

#fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.5))
#ax_fit.scatter(x_TR_numpy, y_TR_numpy, s=8)         
#ax_fit.plot(x_grid.numpy(), h_grid.numpy(), "r")
#ax_fit.set_ylim(-2, 2)
#ax_fit.set_title("fit")
#ax_loss.semilogy(loss_array)
#ax_loss.set_title("training loss")
#ax_loss.set_xlabel("step")
#plt.tight_layout()
#plt.show()