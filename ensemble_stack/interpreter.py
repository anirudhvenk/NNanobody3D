import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
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
        for step, (data) in enumerate(dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            predictions.append(model(inputs))
        output.append([])
        output[idx].append(predictions)

    output = np.array(output)
    return output.reshape(output.shape[0], output.shape[2]).T


class InterpreterData(Dataset):
    def __init__(self, X):
        self.x = X
        enrichment = np.loadtxt(
            'data/regression/Full Regression/data.target')
        enrichment = enrichment.reshape(enrichment.shape[0], 1)
        self.y = torch.from_numpy(np.vstack([x for x in enrichment]))
        self.n_samples = len(raw_seqs)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class Interpreter(nn.Module):
    def __init__(self):
        super(Interpreter, self).__init__()
        self.interpreter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.interpreter(x)


validation_loader = DataLoader(Validation(), batch_size=100, shuffle=True)
models = load_all_models()
print(models)
# output = get_stacked_predictions(models, validation_loader)
# stacked_dataset = pd.DataFrame(columns=list(models.keys()), data=output)

# interpreter_loader = DataLoader(InterpreterData(stacked_dataset.values), batch_size=100, shuffle=True)








