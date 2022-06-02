from cProfile import run
import os
from turtle import forward
from webbrowser import get
import torch
import torch.optim as optim
from torch import nn
from preprocess import FullRegression, HoldOutRegression, HoldOutTop, Validation
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

class Seq_32_32(nn.Module):
    def __init__(self):
        super(Seq_32_32, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(400, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

model = Seq_32_32()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
training_loader = DataLoader(HoldOutRegression(), batch_size=100, shuffle=False)
validation_loader = DataLoader(Validation(), batch_size=100, shuffle=False)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# print(get_n_params(model))
for epoch in range(20):  # loop over the dataset multiple times
    
    for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # print(i)
        inputs, labels = data

        # # zero the parameter gradients
        optimizer.zero_grad()

        # # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss.item())
        
    # for i, data in enumerate(validation_loader, 0):
    #     inputs, labels = data
    #     outputs = model(inputs)
    #     val_loss = loss_fn(outputs, labels)
        
    # print(f'epoch: {epoch+1}, loss: {loss.item()}')
    
#     print(running_loss)

        # print statistics
        # running_loss += loss.item()
        # print(i)
        # print("hello world")
        # if i % 100 == 99:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0

# for epoch in range(20):
#     inputs, labels = next(iter(training_loader))
#     outputs = model(inputs)
#     loss = loss_fn(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     print(loss)

# inputs, labels = next(iter(training_loader))
# print(inputs)