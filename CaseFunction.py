# -*- coding: utf-8 -*-
"""
---------------------------------------------------
Neural network for binary classifier.
---------------------------------------------------

The problem: 
------------
A tuple (x_1, x_2) of real numbers is mapped to 1 if and only if x_1*x_2 < 0. 
Else the tuple is mapped to 0. 

This problem is taken from F. Ruehle / Physics Reports 839 (2020) 1–117. 

The following code attempts to solve this problem using a neural network.  

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 0) Function to create training and test data 

def createdata(samplesize, scalefactor):
    X = []
    y = []
    for i in range(samplesize):
        X_vec = scalefactor*np.random.normal(size=2)
        X.append(X_vec)
        # implementing the logic to relate features and output
        if (X_vec[0]*X_vec[1] < 0):
            y.append(1)
        else:
            y.append(0)
    
    # Converting numpy arrays containing numbers of float32 into torch tensors
    X = torch.from_numpy(np.asarray(X).astype(np.float32))
    y = torch.from_numpy(np.asarray(y).astype(np.float32))
    y = y.view(y.shape[0], 1) # creating a column vector 

    return X, y
    
# 0.1) Test data 

X_test, y_test = createdata(1000, 10)

# 0.2) Training data 

X_train, y_train = createdata(100, 10)

 
# 1) The model 

# Number of nodes in each layer
input_dim = 2
hidden1_dim = 4
hidden2_dim = 4
output_dim = 1
# Number of hidden layers and its dimensions suggested by F. Ruehle

# Structure
model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden1_dim),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden1_dim, hidden2_dim),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden2_dim, output_dim),
    torch.nn.Sigmoid()
)

model.train()


# 2) Loss and optimizer

learning_rate = 5 # chosen by some trial and error; could be found by some optimisation function

criterion = nn.BCELoss() # binary cross entropy loss

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # stochastic gradient descent


# 3) Training loop

# 3.1) Implementation of the training 

Loss = [] # collecting losses in list
Epoch = [] # collecting epochs in list
num_epochs = 500 # number of total epochs for the training

for epoch in range(num_epochs):
    
    # Forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # Backward pass and update
    loss.backward()
    optimizer.step()

    # Zero grad before new step
    optimizer.zero_grad()
    
    # Printing some epochs and the current losses
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    
    # Filling lists for epoch and loss
    Epoch.append(epoch)
    Loss.append(Variable(loss))
    
print('Training completed')

# 3.2) Plot of the training errors

plt.plot(Epoch, Loss)
plt.title('Training errors')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.show()
    

# 4) Test and accuracy

testcases = len(X_test)

fail = 0
for i in range(testcases):
    y_predicted = torch.round(model(X_test[i])) # we round the predicted output 
    # count fails
    if y_predicted != y_test[i]:
        fail += 1
    
accuracy = 1 - fail/testcases # computation of accuracy of the neural network
print('Accuracy:', accuracy)




