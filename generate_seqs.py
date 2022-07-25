import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

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

data = AntibodyDataset('graph_generation/data/seeds/3eak.jsonl', cdr_type='3')
loader = StructureLoader(data.data, batch_tokens=1, interval_sort=3)


model_ckpt, opt_ckpt, model_args = torch.load('graph_generation/weights/model.best', map_location=device)
model = HierarchicalDecoder(model_args).to(device)
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(model_ckpt)
optimizer.load_state_dict(opt_ckpt)
model.eval()
    

with torch.no_grad():
    for hbatch in loader:
        hX, hS, hL, hmask = completize(hbatch)
        generated_cdrs = {}
        
        for i in range(3000):
            cdr3, ppl, _ = model.generate(hS, hL, hmask, return_ppl=True)
            generated_cdrs[cdr3[0]] = ppl.item()
            enrichment = get_stacked_prediction(cdr3).item()
            
            if (enrichment > 0.5):
                print(cdr3, enrichment)