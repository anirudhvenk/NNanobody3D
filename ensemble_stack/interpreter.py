import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from reg_models import *
from preprocess import *


def load_all_models():
    model_list = {}
    for dataset in ['Full Regression','Hold out Regression','Hold out Top 4%']:
        for loaded_model in ['seq_32x1_16', 'seq_64x1_16','seq_32x2_16','seq_32_32','seq_32x1_16_filt3','seq_embed_32x1_16']:
            model = torch.load(f'weights/regression/{dataset}/{loaded_model}.pt')
            model_list[dataset + ' ' + loaded_model] = model

    return model_list


def get_stacked_predictions(model_list, dataset):
    output = []
    for idx, model in enumerate(model_list.values()):
        predictions = []
        for step, data in enumerate(dataset):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            out = model(inputs).flatten()

            predictions.append(out.tolist())

        output.append([])
        output[idx].append(np.concatenate(predictions))
    output = np.vstack(output)
    return output.reshape(output.shape[0], output.shape[1]).T


class InterpreterData(Dataset):
    def __init__(self, X):
        self.x = torch.from_numpy(X)
        enrichment = np.loadtxt(
            'data/regression/Test set Regression/test_target.txt')
        enrichment = enrichment.reshape(enrichment.shape[0], 1)
        self.y = torch.from_numpy(np.vstack([x for x in enrichment]))
        self.n_samples = len(enrichment)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class Interpreter(nn.Module):
    def __init__(self):
        super(Interpreter, self).__init__()
        self.interpreter = nn.Sequential(
            nn.Linear(18, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.interpreter(x)


if __name__ == '__main__':
    validation_loader = DataLoader(Validation(), batch_size=100)
    models = load_all_models()
    output = get_stacked_predictions(models, validation_loader)
    stacked_dataset = pd.DataFrame(columns=list(models.keys()), data=output)
    Y_train = enrichment = np.loadtxt(
                'data/regression/Test set Regression/test_target.txt')

    interpreter_loader = DataLoader(InterpreterData(stacked_dataset.values), batch_size=100, shuffle=True)
    interpreter = Interpreter().cuda()
    optimizer = optim.Adam(interpreter.parameters())
    loss_fn = torch.nn.MSELoss()

    for epoch in range(20):
        train_loss = 0.0

        for step, data in enumerate(interpreter_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = interpreter(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*len(inputs)

        print(epoch+1, train_loss / len(interpreter_loader.sampler))

    torch.save(interpreter.state_dict(), 'weights/regression/interpret_16x1.pth')