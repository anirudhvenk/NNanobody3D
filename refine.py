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
from generate_seqs import Generator

from graph_generation.data import *
from graph_generation.hierarchical import *

from ensemble_stack.inference import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# TODO: similarity scoring (alphafold?)

for epoch in range(5):
    print("Generating sequences...")
    
    if epoch == 0:
        generator = Generator(f'graph_generation/weights/refine/model.init')
        generator.generate_sequences(1000, 0.5, 0)
        model_ckpt, opt_ckpt, model_args = torch.load(f'graph_generation/weights/refine/model.init', map_location=device)
    else:
        generator = Generator(f'graph_generation/weights/refine/model.ckpt.{epoch}')
        generator.generate_sequences(1000, 0.5, epoch)
        model_ckpt, opt_ckpt, model_args = torch.load(f'graph_generation/weights/refine/model.ckpt.{epoch}', map_location=device)
        
    model = HierarchicalDecoder(model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(model_ckpt)
    optimizer.load_state_dict(opt_ckpt)

    data = pd.read_csv(f'graph_generation/data/generated/3eak_gen_{epoch}.tsv', sep='\t')
    generated_sequences = data['cdr3'].tolist()
    framework = 'QVQLVESGGGLVQPGGSLRLSCAASGGSEYSYSTFSLGWFRQAPGQGLEAVAAIASMGGLTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCWGQGTLVTVS'

    
    print("Training on previously generated sequences...")

    model.train()
    meter = 0

    for i, seq in enumerate(tqdm(generated_sequences)):   
        full_sequence = f'QVQLVESGGGLVQPGGSLRLSCAASGGSEYSYSTFSLGWFRQAPGQGLEAVAAIASMGGLTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC{seq}WGQGTLVTVS'
        cdr3 = ''.join(['3' for _ in range(len(seq))])
        true_cdr = [f'000000000000000000000000011111111111000000000000000002222222200000000000000000000000000000000000000{cdr3}0000000000']
        mask = np.ones((1, len(full_sequence)), dtype=np.int32)
        
        optimizer.zero_grad()
        true_S = np.asarray([[alphabet.index(a) for a in full_sequence]], dtype=np.int32)
        
        loss = model(torch.from_numpy(true_S).long().to(device), true_cdr, torch.from_numpy(mask).float().to(device))
        loss.backward()
        optimizer.step()
        
        meter+=loss.exp().item()
        
    print('Loss: ', meter/len(generated_sequences))
    ckpt = (model.state_dict(), optimizer.state_dict(), model_args)
    torch.save(ckpt, f'graph_generation/weights/refine/model.ckpt.{epoch + 1}')
