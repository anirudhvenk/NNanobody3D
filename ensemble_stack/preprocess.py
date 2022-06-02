import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

aa_alphabet = 'ILVFMCAGPTSYWQNHEDKRXJ'
fasta_to_tensor = lambda fasta: torch.tensor([aa_alphabet.index(aa) for aa in fasta], dtype=torch.int64)

def one_hot_encode(seq):
    one_hot = F.one_hot(fasta_to_tensor(seq), num_classes=22).to(torch.float)
    for idx, category in enumerate(one_hot):
        if category[-1] == 1:
            one_hot[idx] = torch.full((22,), 0)
        elif category[-2] == 1:
            one_hot[idx] = torch.full((22,), 0.05)
            
    return torch.stack([one_hot[idx][:-2] for idx, _ in enumerate(one_hot)])

class FullRegression(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt('../regression/Full Regression/data.tsv', dtype='str')[:,1]
        enrichment = np.loadtxt('../regression/Full Regression/data.target').flatten()
        self.x = torch.stack([one_hot_encode(seq) for seq in raw_seqs])
        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)
        
    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()
    
    def __len__(self):
        return self.n_samples
    
class HoldOutRegression(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt('regression/Hold out Regression/data.tsv', dtype='str')[:,1]
        enrichment = np.loadtxt('regression/Hold out Regression/data.target').flatten()
        self.x = torch.stack([one_hot_encode(seq) for seq in raw_seqs])
        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)
        
    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()
    
    def __len__(self):
        return self.n_samples
    
class HoldOutTop(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt('regression/Hold out Top 4%/data.tsv', dtype='str')[:,1]
        enrichment = np.loadtxt('regression/Hold out Top 4%/data.target')
        self.x = torch.stack([one_hot_encode(seq) for seq in raw_seqs])
        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)
        
    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()
    
    def __len__(self):
        return self.n_samples
    
class Validation(Dataset):
    def __init__(self):
        raw_seqs = np.loadtxt('regression/Test set Regression/test.tsv', dtype='str')[:,1]
        enrichment = np.loadtxt('regression/Test set Regression/test_target.txt')
        self.x = torch.stack([one_hot_encode(seq) for seq in raw_seqs])
        self.y = torch.from_numpy(enrichment)
        self.n_samples = len(raw_seqs)
        
    def __getitem__(self, index):
        return self.x[index].float(), self.y[index].float()
    
    def __len__(self):
        return self.n_samples

# cdr3_one_hot = torch.stack([one_hot_encode(seq) for seq in data])
# print(cdr3_one_hot.shape)