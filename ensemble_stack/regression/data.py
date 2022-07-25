import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os


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


class FullRegression(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'data/Full Regression/data.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt(
            'data/Full Regression/data.target')
        enrichment = enrichment.reshape(enrichment.shape[0], 1)
        mapper = load_mapper('data/mapper')
        self.x = torch.stack([one_hot_encode(seq, mapper) for seq in raw_seqs])

        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class HoldOutRegression(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'data/Hold out Regression/data.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt(
            'data/Hold out Regression/data.target')
        enrichment = enrichment.reshape(enrichment.shape[0], 1)
        mapper = load_mapper('data/mapper')
        self.x = torch.stack([one_hot_encode(seq, mapper) for seq in raw_seqs])
        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class HoldOutTop(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'data/Hold out Top 4%/data.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt('data/Hold out Top 4%/data.target')
        enrichment = enrichment.reshape(enrichment.shape[0], 1)
        mapper = load_mapper('data/mapper')
        self.x = torch.stack([one_hot_encode(seq, mapper) for seq in raw_seqs])
        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class Validation(Dataset):
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
    
    
class ValidationInterpreter(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'regression/data/Test set Regression/test.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt(
            'regression/data/Test set Regression/test_target.txt')
        enrichment = enrichment.reshape(enrichment.shape[0], 1)
        mapper = load_mapper('regression/data/mapper')
        self.x = torch.stack([one_hot_encode(seq, mapper) for seq in raw_seqs])
        self.y = torch.from_numpy(np.vstack([x for x in enrichment]))
        self.n_samples = len(raw_seqs)
        

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples