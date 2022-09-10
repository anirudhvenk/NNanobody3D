import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from models import *
from data import *


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(model, dataset, model_name, training_loader, validation_loader):
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    cur_val_loss = 1e10
    model.train()

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

        print(epoch+1, train_loss / len(training_loader.sampler), val_loss / len(validation_loader.sampler))
        
        if val_loss < cur_val_loss:
            cur_val_loss = val_loss
            print(f'saving model to weights/{dataset}/{model_name}.pth')
            torch.save(model.state_dict(), f'weights/{dataset}/{model_name}.pth')


if __name__ == '__main__':
    training_datasets = {
        'Full Regression': DataLoader(FullRegression(lstm=True), batch_size=100, shuffle=True),
        'Hold out Regression': DataLoader(HoldOutRegression(lstm=True), batch_size=100, shuffle=True),
        'Hold out Top 4%': DataLoader(HoldOutTop(lstm=True), batch_size=100, shuffle=True)
    }
    validation_loader = DataLoader(Validation(lstm=True), batch_size=100)

    for key in training_datasets:
        # print('Training Seq_32_32 on {}'.format(key))
        # train(Seq_32_32().to(device), key, 'seq_32_32', training_datasets[key], validation_loader)

        # print('Training Seq_32x1_16 on {}'.format(key))
        # train(Seq_32x1_16().to(device), key, 'seq_32x1_16', training_datasets[key], validation_loader)

        # print('Training Seq_32x1_16_filt3 on {}'.format(key))
        # train(Seq_32x1_16_filt3().to(device), key, 'seq_32x1_16_filt3', training_datasets[key], validation_loader)

        # print('Training Seq_32x2_16 on {}'.format(key))
        # train(Seq_32x2_16().to(device), key, 'seq_32x2_16', training_datasets[key], validation_loader)

        # print('Training Seq_64x1_16 on {}'.format(key))
        # train(Seq_64x1_16().to(device), key, 'seq_64x1_16', training_datasets[key], validation_loader)

        # print('Training Seq_embed_32x1_16 on {}'.format(key))
        # train(Seq_embed_32x1_16().to(device), key, 'seq_embed_32x1_16', training_datasets[key], validation_loader)

        print('Training Seq_LSTM_32x1_16 on {}'.format(key))
        train(Seq_LSTM_32x1_16().to(device), key, 'seq_LSTM_32x1_16', training_datasets[key], validation_loader)
        
        print('Training Seq_LSTM_32x2_16 on {}'.format(key))
        train(Seq_LSTM_32x2_16().to(device), key, 'seq_LSTM_32x2_16', training_datasets[key], validation_loader)
        
        print('Training Seq_LSTM_64x1_16 on {}'.format(key))
        train(Seq_LSTM_64x1_16().to(device), key, 'seq_LSTM_64x1_16', training_datasets[key], validation_loader)