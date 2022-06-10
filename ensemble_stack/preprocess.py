import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


def load_mapper():
    mapper = {}
    with open('data/regression/mapper', 'r') as f:
        for x in f:
            line = x.strip().split()
            word = line[0]
            vec = [float(item) for item in line[1:]]
            mapper[word] = vec
    return mapper


def one_hot_encode(seq, mapper):
    one_hot_mat = torch.tensor([mapper[aa] for aa in seq]).T
    return one_hot_mat


class FullRegression(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'data/regression/Full Regression/data.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt(
            'data/regression/Full Regression/data.target')
        mapper = load_mapper()
        self.x = torch.stack([one_hot_encode(seq, mapper) for seq in raw_seqs])
        self.y = torch.from_numpy(np.vstack([x for x in enrichment]))
        self.n_samples = len(raw_seqs)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class HoldOutRegression(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'data/regression/Hold out Regression/data.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt(
            'data/regression/Hold out Regression/data.target').flatten()
        self.x = torch.stack([one_hot_encode(seq) for seq in raw_seqs])
        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class HoldOutTop(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'data/regression/Hold out Top 4%/data.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt('data/regression/Hold out Top 4%/data.target')
        self.x = torch.stack([one_hot_encode(seq) for seq in raw_seqs])
        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples


class Validation(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt(
            'data/regression/Test set Regression/test.tsv', dtype='str')[:, 1]
        enrichment = np.loadtxt(
            'data/regression/Test set Regression/test_target.txt')
        mapper = load_mapper()
        self.x = torch.stack([one_hot_encode(seq, mapper) for seq in raw_seqs])
        self.y = torch.from_numpy(np.vstack([x for x in enrichment]))
        self.n_samples = len(raw_seqs)

    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()

    def __len__(self):
        return self.n_samples
