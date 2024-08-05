import torch 
import numpy as np

## WHAT ARE TENSORS? ##
# tensors are data obects. A 1d tensor is a vector, a 2d tensor is a matrix etc; 
    
## HOW TO CREATE TENSORS ##

# Using list objects

#initialise a data structure the regular way you would in python 
data = [[1,2],[3,4]]
type(data)

#convert that data structure to a torch tensor 
x_data = torch.tensor(data)
type(x_data) #notice that x_data is a torch.Tensor data type now 

# Using numpy arrays

np_array = np.array(data)   # declaring the numpy array
type(np_array)

x_np = torch.tensor(np_array)   # using prevvious method
type(x_np)

x_np2 = torch.from_numpy(np_array)  # using from_numpy method
type(x_np2)

# Notice above that there appears to be little difference between methods, 
#  i would assume that the from_numpy method probably exists for a reason, so it might be safer to use that 

# From another tensor

x_ones = torch.ones_like(x_data) #retains the properties of x_ones (2x2 matrix)
x_ones

data2 = [[3,4,1], [1,4,3], [3,5,6],[7,2,3]]
xdata2 = torch.Tensor(data2)

x_ones2 = torch.ones_like(xdata2, dtype = int)
x_ones2 #  4x3 matrix, datatype = int


x_rand = torch.rand_like(xdata2, dtype=torch.float)
x_rand

#shape is a tuple of tensor dimensions
#you can use shape as as a way of using functions like ones and rand without having a dataframe to compare it to
shape = (2,3,)
rand_tensor = torch.rand(shape)
rand_tensor

# How to view tensor attributes 
rand_tensor.shape   # shape of the tensor
rand_tensor.dtype   # what data type is stored in the tensor 
rand_tensor.device  # is the tensor running from the GPU or the CPU? (or some other device?)


## Tensor Opperations ##

#tensor opperations can be performed on the GPU faster than on the CPU


tensor = torch.rand(3,4)

# This code is to allocate tensors to the GPU if possible, i can't unnfortunately because of my graphics card
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device Tensor is stored on: {tensor.device}")

# indexing and slicing     
tensor = torch.ones(4,4)
tensor[:,1] = 0 # all elements in the column 1 are 0's 
tensor

# joining tensors 
t1 = torch.cat([tensor, tensor, tensor], dim = 1) # the dimension affects the shape of the concatenation
t1

# multiplying tensors 
shape = (3,4,)
tensor = torch.rand(shape)
tensor
tensor2 = tensor*tensor
tensor2
#alternatively
tensor2 = tensor.mul(tensor)
tensor2

# matrix multiplication for tensors
# remember that you can only multiply a N x M matrix with a P x Q matrix if M = P

tensor3 = tensor.matmul(tensor.T)
tensor
tensor.T # tensor.T gives the transpose of a tensor
tensor3

#alternative syntax
tensor3 = tensor@tensor.T
tensor3

# in place opperations 
# in place opperations use a '_' after their opperation 

tensor3.add_(3)
tensor3

# try to avoid using in place opperations as they can be problematic when working with derivatives

# Tensor-Numpy bridging 
# tensors and numpy arrays  may share a memory address, thus changing one will change both 

t = torch.ones(5)
n = t.numpy()
n

t.add_(1)
n

# converting from a numpy array to a tensor 
t = torch.from_numpy(n)
t

#### INTRO TO torch.autograd ###

## torch.autograd is pytorch's differentiation engine 

##training a neural net happens in two steps 
    # forward propagation: 
        # the nn uses it's 'best guess' for the correct output and runs the input through each of it's functions to make this guess
    #backward propagation
        # the nn adjusts it's parameters proportionate to the the error in it's guess. 
        # this is done by traversing backwards through the nn using te derrivatives of the error with respect to the parameters of the functioins
        # and optimising the parameters using gradient decent.

## example: a single training step using a pretrained model (resnet18) from torchvision. we'll use a random 
# tensor to represent a single image with 3 channels and a height and width of 64. 

import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

# we create a model with the default weights from resnet18
model = resnet18(weights = ResNet18_Weights.DEFAULT) # create the initial model 

data = torch.rand(1,3,64, 64) #define random data in the dimensions we need
labels = torch.rand(1,1000) # create labels

#run data through each of it's layers to make a prediction. This the forward pass

prediction = model(data)

# using the models prediction and the corresponding label we can calculate the error (loss)
loss = (prediction - labels).sum()
loss.backward()

# load in an optimiser, SGD (stochastic gradient decent)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# the step() function initiates the gradient decent. the optimiser adjusts adjusts each parameter by its gradient stored in .grad
optim.step()

### Differentiation in autograd ###
## detailed workings in autograd ## 

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([4., 6.], requires_grad=True)

Q = 3*a**3 - b**2
Q
# assume that Q is the error and a and b are the parameters on the NN
# when we call backward() on Q, autograd calculats these gradients and stroes them in the respective tensors' .grad attribute 

# we need to explicitly pass a gradient argument in Q.backward() because it's a vector. gradient is a tensor of the same shape as Q, 
# and it represents the gradient of Q it's self. dQ/dQ = 1
# we can also aggregate Q into a scalar and call it backward implicitly with Q.sum().backward()

external_grad = torch.tensor([1., 1.])
Q.backward(gradient = external_grad)

#gradients are now deposited in a.grad and b.grad
print(9*a**2 == a.grad)
print(a.grad)
print(-2*b == b.grad)
print(b.grad)

# parameters that don't compute gradients are called 'frozen parameters'. It's handy to freeze part of your model if you know you won't 
# need gradients for those parameters. this is helpful for computational efficiency. 


### NEURAL NETWORKS ###

## Typical training procedure for a neural network: 
# - Define a nn that has some learnable parameters (or weights)
# - itterate over the set of inputs 
# - pass each input through the nn
# - calculate loss (how far is that input from correct?)
# - back propogation...ie; calcuate the gradients of the weights and biases based off of the loss calculated
# - update weights of the network model according to loss -> simple update rule: weight = weigtht - learning rate*gradient. 

# in general: 
# build model
# feed forward
# calculate loss
# back propogate 
# adjust model according to gradients calculated

### now to code this 

import torch.nn as nn
import torch.nn.functional as F #library for convolution functions 


#build the model by defining a class
# what is a convolution? 
    # a convoution is when you perform an opperation on two functions to create a third 

class Net(nn.Module): #build a 'Net' class that inherits all the methods and properties from nn.Module
    def __init__(self): 
        super(Net, self).__init__() #this allows Net's init to be the same as nn.module's init

        #we want to make convolutional nn with 1 input image, 6 output channels and a 5x5 convolution kernel
        self.conv1 = nn.Conv2d(1,6,5) #input size= 1, output size = 6, kernel size = 5
        self.conv2 = nn.Conv2d(6, 16, 5) #input size= 6, output size = 16, kernel size = 5

        #fcn is the nth inner layer. fc stands for fully connected 
        self.fc1 = nn.Linear(16*5*5, 120) #16*5*5 = 400 inputs, 120 outputs 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #notice above that the output of fc(n-1) is the same as the input for fcn (for 1<=n<=3)
    
    #Now we need to create a fuction that performs the forward opperation (pass each input throughb the nn)
    def forward(self, input): 
        #c1 is a convolutional layer taking in 1 input image channel, it has 6 output channels and a 5x5 convolution 
        # we'll use the RELU activation function (f(x) = max(0, x)), relu = rectified linear unit
        #c1's output is tensor with size (N, 6, 28, 28) where N is the soze of the batch
        c1 = F.relu(self.conv1(input))
        #s2 is a subsampling layer. it samples using a 2x2 grid. 
        #it outputs a (N, 16, 14, 14) tensor
        s2 = F.max_pool2d(c1, (2,2))
        #C3 is a convolutional layer with 6 input channels, 16 output channels, 5x5 square convolution. 
        # it uses RELU and outputs a (N, 16, 10, 10) tensor  
        c3 = F.relu(self.conv2(s2))
        # s4 is a subsampling layer on a 2 x 2 grid. it outputs a (N, 16, 5, 5) tensor
        s4 = F.max_pool2d(c3, 2)
        #we now flatten s4 to create a (N, 400) tensor output
        s4 = torch.flatten(s4, 1)
        # F5 is a fully connected layer with (N, 400) tensor input. uses RELU and outputs a (N, 120) tensor 
        f5 = F.relu(self.fc1(s4))
        # f6 is a fully connected layer with (N, 120) input and (N, 84) ooutput
        f6 = F.relu(self.fc2(f5))
        #the output layer takes a (N, 84) input and outputs a (N, 10) tensor
        output = self.fc3(f6)
        return output
    # note that we only have to compute a forward fuction. the backwards functino is done automatically using autograd
    #  
net = Net()
print(net)

#The paramaters learned by the model can be accessed using net.parameters()

params = list(net.parameters())
print(len(params))  # no. of parametrs 
print(params[0].size()) #conv1's weight 
print(params[0].size)

#what happens if we try a random 32 x 32 input

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

#zero the gradient buffers if all parameters and backprops with random gradients 
net.zero_grad()
out.backward(torch.randn(1,10))




