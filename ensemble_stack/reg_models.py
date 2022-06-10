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
            nn.Conv2d(20, 32, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(32 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_32x1_16(x)


class Seq_32x1_16_filt3(nn.Module):
    def __init__(self):
        super(Seq_32x1_16_filt3, self) .__init__()
        self.seq_32x1_16_filt3 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(1, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(32 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_32x1_16_filt3(x)


class Seq_32x2_16(nn.Module):
    def __init__(self):
        super(Seq_32x2_16, self) .__init__()
        self.seq_32x2_16 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(32 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_32x2_16(x)


class Seq_64x1_16(nn.Module):
    def __init__(self):
        super(Seq_64x1_16, self) .__init__()
        self.seq_64x1_16 = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(64 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_64x1_16(x)


class Seq_embed_32x1_16(nn.Module):
    def __init__(self):
        super(Seq_embed_32x1_16, self) .__init__()
        self.seq_embed_32x1_16 = nn.Sequential(
            nn.Conv2d(20, 8, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(8, 64, kernel_size=(1, 5), padding='same'),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_embed_32x1_16(x)


model = Seq_embed_32x1_16()
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

    print(epoch+1, train_loss / len(training_loader.sampler, ))
    # print(epoch+1, val_loss / len(validation_loader.sampler))
