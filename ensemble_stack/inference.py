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

def load_all_models():
    model_list = {}
    for dataset in ['Full Regression','Hold out Regression','Hold out Top 4%']:
        for loaded_model in ['seq_32x1_16', 'seq_64x1_16','seq_32x2_16','seq_32_32','seq_32x1_16_filt3','seq_embed_32x1_16']:
            model = torch.load(f'weights/regression/{dataset}/{loaded_model}.pt')
            model_list[dataset + ' ' + loaded_model] = model

    return model_list


def get_stacked_prediction(sequences):
    model_list = load_all_models()
    output = []
    mapper = load_mapper()
    oh_sequences = torch.stack([one_hot_encode(seq, mapper) for seq in sequences]).cuda()

    for idx, model in enumerate(model_list.values()):
        predictions = model(oh_sequences).cpu().detach().numpy().flatten()
        output.append([])
        output[idx].append(predictions)
    output = np.vstack(output)
    output = output.reshape(output.shape[0], output.shape[1]).T
    # return output
    interpreter = torch.load(f'weights/regression/interpret_16x1.pt')
    return(interpreter(torch.from_numpy(output).cuda()))
