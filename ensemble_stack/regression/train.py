import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from ensemble_stack.regression.models import *
from data import *


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('mps')


def train(model, training_loader, validation_loader):
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    for epoch in range(20):
        val_loss = 0.0
        train_loss = 0.0

        for step, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*len(inputs)

        with torch.no_grad():
            for step, data in enumerate(validation_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()*len(inputs)

        print(epoch+1, train_loss / len(training_loader.sampler),
              val_loss / len(validation_loader.sampler))
    return model


if __name__ == '__main__':
    training_datasets = {
        'Full Regression': DataLoader(FullRegression(), batch_size=100, shuffle=True),
        'Hold out Regression': DataLoader(HoldOutRegression(), batch_size=100, shuffle=True),
        'Hold out Top 4%': DataLoader(HoldOutTop(), batch_size=100, shuffle=True)
    }
    validation_loader = DataLoader(Validation(), batch_size=100)

    for key in training_datasets:
        print('Training Seq_32_32:')
        model = train(Seq_32_32().to(device),
                      training_datasets[key], validation_loader)
        torch.save(model.state_dict(), f'weights/{key}/seq_32_32.pth')

        print('Training Seq_32x1_16:')
        model = train(Seq_32x1_16().to(device),
                      training_datasets[key], validation_loader)
        torch.save(model.state_dict(), f'weights/{key}/seq_32x1_16.pth')

        print('Training Seq_32x1_16_filt3:')
        model = train(Seq_32x1_16_filt3().to(device),
                      training_datasets[key], validation_loader)
        torch.save(model.state_dict(), f'weights/{key}/seq_32x1_16_filt3.pth')

        print('Training Seq_32x2_16:')
        model = train(Seq_32x2_16().to(device),
                      training_datasets[key], validation_loader)
        torch.save(model.state_dict(), f'weights/{key}/seq_32x2_16.pth')

        print('Training Seq_64x1_16:')
        model = train(Seq_64x1_16().to(device),
                      training_datasets[key], validation_loader)
        torch.save(model.state_dict(), f'weights/{key}/seq_64x1_16.pth')

        print('Training Seq_embed_32x1_16:')
        model = train(Seq_embed_32x1_16().to(device),
                      training_datasets[key], validation_loader)
        torch.save(model.state_dict(), f'weights/{key}/seq_embed_32x1_16.pth')
