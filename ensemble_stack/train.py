import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from reg_models import *
from preprocess import *

def train(model, training_loader, validation_loader):
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(20):
        val_loss = 0.0
        train_loss = 0.0

        for step, (data) in enumerate(training_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*len(inputs)

        for step, (data) in enumerate(validation_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()*len(inputs)

        print(epoch+1, train_loss / len(training_loader.sampler),  val_loss / len(validation_loader.sampler))
    return model

if __name__ == '__main__':
    training_datasets = {
        'Full Regression': DataLoader(FullRegression(), batch_size=100, shuffle=True), 
        'Hold out Regression': DataLoader(HoldOutRegression(), batch_size=100, shuffle=True), 
        'Hold out Top 4%': DataLoader(HoldOutTop(), batch_size=100, shuffle=True)
    }
    validation_loader = DataLoader(Validation(), batch_size=100, shuffle=True)

    for key in training_datasets:
        print('Training Seq_32_32:')
        model = train(Seq_32_32().cuda(), training_datasets[key], validation_loader)
        torch.save(model, f'weights/regression/{key}/seq_32_32.pt')

        print('Training Seq_32x1_16:')
        model = train(Seq_32x1_16().cuda(), training_datasets[key], validation_loader)
        torch.save(model, f'weights/regression/{key}/seq_32x1_16.pt')

        print('Training Seq_32x1_16_filt3:')
        model = train(Seq_32x1_16_filt3().cuda(), training_datasets[key], validation_loader)
        torch.save(model, f'weights/regression/{key}/seq_32x1_16_filt3.pt')

        print('Training Seq_32x2_16:')
        model = train(Seq_32x2_16().cuda(), training_datasets[key], validation_loader)
        torch.save(model, f'weights/regression/{key}/seq_32x2_16.pt')

        print('Training Seq_64x1_16:')
        model = train(Seq_64x1_16().cuda(), training_datasets[key], validation_loader)
        torch.save(model, f'weights/regression/{key}/seq_64x1_16.pt')

        print('Training Seq_embed_32x1_16:')
        model = train(Seq_embed_32x1_16().cuda(), training_datasets[key], validation_loader)
        torch.save(model, f'weights/regression/{key}/seq_embed_32x1_16.pt')
        