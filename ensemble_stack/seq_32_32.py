import os
import torch
import torch.optim as optim
from torch import nn
from preprocess import FullRegression, HoldOutRegression, HoldOutTop, Validation
from torch.utils.data import DataLoader


class Seq_32_32(nn.Module):
    def __init__(self):
        super(Seq_32_32, self).__init__()
        self.seq_32_32 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.seq_32_32(x)
    

class Seq_32x1_16(nn.Module):
    def __init__(self):
        super(Seq_32x1_16, self) .__init__()
        self.seq_32x1_16 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(1, 5), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(32 * 5, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.seq_32x1_16(x)


model = Seq_32x1_16()
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()
training_loader = DataLoader(FullRegression(), batch_size=100, shuffle=True)
validation_loader = DataLoader(Validation(), batch_size=100, shuffle=True)


for epoch in range(20):
    val_loss = 0.0
    train_loss = 0.0
    
    for step, (inputs, labels) in enumerate(training_loader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*len(inputs)
        
    for step, (inputs, labels) in enumerate(validation_loader, 0):
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        val_loss += loss.item()*len(inputs)

    print(epoch+1, train_loss / len(validation_loader.sampler))
    # print(epoch+1, val_loss / len(validation_loader.sampler))
