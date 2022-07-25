import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ensemble_stack.regression.models import *
from ensemble_stack.regression.data import *

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('mps')


def load_all_models():
    loaded_models = {
        'Full Regression:seq_32_32': Seq_32_32().to(device),
        'Hold out Regression:seq_32_32': Seq_32_32().to(device),
        'Hold out Top 4%:seq_32_32': Seq_32_32().to(device),
        'Full Regression:seq_32x1_16': Seq_32x1_16().to(device),
        'Hold out Regression:seq_32x1_16': Seq_32x1_16().to(device),
        'Hold out Top 4%:seq_32x1_16': Seq_32x1_16().to(device),
        'Full Regression:seq_32x1_16_filt3': Seq_32x1_16_filt3().to(device),
        'Hold out Regression:seq_32x1_16_filt3': Seq_32x1_16_filt3().to(device),
        'Hold out Top 4%:seq_32x1_16_filt3': Seq_32x1_16_filt3().to(device),
        'Full Regression:seq_32x2_16': Seq_32x2_16().to(device),
        'Hold out Regression:seq_32x2_16': Seq_32x2_16().to(device),
        'Hold out Top 4%:seq_32x2_16': Seq_32x2_16().to(device),
        'Full Regression:seq_64x1_16': Seq_64x1_16().to(device),
        'Hold out Regression:seq_64x1_16': Seq_64x1_16().to(device),
        'Hold out Top 4%:seq_64x1_16': Seq_64x1_16().to(device),
        'Full Regression:seq_embed_32x1_16': Seq_embed_32x1_16().to(device),
        'Hold out Regression:seq_embed_32x1_16': Seq_embed_32x1_16().to(device),
        'Hold out Top 4%:seq_embed_32x1_16': Seq_embed_32x1_16().to(device),
    }
    
    for key in loaded_models:
        dataset = key.split(':')[0]
        model = key.split(':')[1]
        
        # print(os.getcwd())
        loaded_models[key].load_state_dict(torch.load(f'ensemble_stack/regression/weights/{dataset}/{model}.pth'))
        loaded_models[key].eval
    
    return loaded_models


def get_stacked_predictions(model_list, dataset):
    output = []
    for idx, model in enumerate(model_list.values()):
        predictions = []
        for step, data in enumerate(dataset):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs).flatten()

            predictions.append(out.tolist())

        output.append([])
        output[idx].append(np.concatenate(predictions))
    output = np.vstack(output)
    return output.reshape(output.shape[0], output.shape[1]).T


def load_mapper(path):
    mapper = {}
    with open(path, 'r') as f:
        for x in f:
            line = x.strip().split()
            word = line[0]
            vec = [float(item) for item in line[1:]]
            mapper[word] = vec
    return mapper


def one_hot_encode(seq, mapper):
    one_hot_mat = torch.stack([torch.stack([_])
                              for _ in torch.tensor([mapper[aa] for aa in seq]).T])
    return one_hot_mat


class InterpreterData(Dataset):
    def __init__(self, X):
        self.x = torch.from_numpy(X)
        enrichment = np.loadtxt(
            'data/Test set Regression/test_target.txt')
        enrichment = enrichment.reshape(enrichment.shape[0], 1)
        self.y = torch.from_numpy(np.vstack([x for x in enrichment]))
        self.n_samples = len(enrichment)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples

 
class SampleData(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'data/Test set Regression/test.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt(
            'data/Test set Regression/test_target.txt')
        enrichment = enrichment.reshape(enrichment.shape[0], 1)
        mapper = load_mapper('data/mapper')
        self.x = torch.stack([one_hot_encode(seq, mapper) for seq in raw_seqs])
        self.y = torch.from_numpy(np.vstack([x for x in enrichment]))
        self.n_samples = len(raw_seqs)
        

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples