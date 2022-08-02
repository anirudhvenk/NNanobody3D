import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import json
import csv
import math
import random
import sys
import numpy as np
import argparse
import os
import csv

from graph_generation.data import *
from graph_generation.hierarchical import *

from ensemble_stack.inference import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_path = 'graph_generation/data/sabdab/hcdr3_cluster/train_data.jsonl'
val_path = 'graph_generation/data/sabdab/hcdr3_cluster/val_data.jsonl'
test_path = 'graph_generation/data/sabdab/hcdr3_cluster/test_data.jsonl'

loaders = []
for path in [train_path, val_path, test_path]:
    data = AntibodyDataset(path, cdr_type='3')
    loader = StructureLoader(data.data, batch_tokens=1, interval_sort=3)
    loaders.append(loader)

loader_train, loader_val, loader_test = loaders

model_ckpt, opt_ckpt, model_args = torch.load('graph_generation/weights/model.best', map_location=device)
model = HierarchicalDecoder(model_args).to(device)
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(model_ckpt)
optimizer.load_state_dict(opt_ckpt)

print('Training:{}, Validation:{}, Test:{}'.format(
    len(loader_train.dataset), len(loader_val.dataset), len(loader_test.dataset))
)

best_ppl, best_epoch = 100, -1
for e in range(10):
    model.train()
    meter = 0
    pbar = tqdm(loader_train)
    
    for i, hbatch in enumerate(pbar):
        optimizer.zero_grad()
        hchain = completize(hbatch)
        if hchain[-1].sum().item() == 0:
            continue
        
        print(hchain)
        
        loss = model(*hchain)
        loss.backward()
        optimizer.step()

        meter += loss.exp().item()
        if (i + 1) % 50 == 0:
            meter /= 50
            print(f'[{i + 1}] Train Loss = {meter:.3f}')
            meter = 0