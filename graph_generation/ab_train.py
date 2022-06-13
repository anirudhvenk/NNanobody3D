import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import json
import csv
import math, random, sys
import numpy as np
import argparse
import os

from structgen import *
from tqdm import tqdm


def evaluate(model, loader, args):
    model.eval()
    val_nll = val_tot = 0.
    val_rmsd = []
    with torch.no_grad():
        for hbatch in tqdm(loader):
            hX, hS, hL, hmask = completize(hbatch)
            for i in range(len(hbatch)):
                L = hmask[i:i+1].sum().long().item()
                if L > 0:
                    out = model.log_prob(hS[i:i+1, :L], [hL[i]], hmask[i:i+1, :L])
                    nll, X_pred = out.nll, out.X_cdr
                    val_nll += nll.item() * hL[i].count(args.cdr_type)
                    val_tot += hL[i].count(args.cdr_type)
                    l, r = hL[i].index(args.cdr_type), hL[i].rindex(args.cdr_type)
                    rmsd = compute_rmsd(X_pred[:, :, 1, :], hX[i:i+1, l:r+1, 1, :], hmask[i:i+1, l:r+1])  # alpha carbon
                    val_rmsd.append(rmsd.item())

    return math.exp(val_nll / val_tot), sum(val_rmsd) / len(val_rmsd)


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='data/sabdab/hcdr3_cluster/train_data.jsonl')
parser.add_argument('--val_path', default='data/sabdab/hcdr3_cluster/val_data.jsonl')
parser.add_argument('--test_path', default='data/sabdab/hcdr3_cluster/test_data.jsonl')
parser.add_argument('--save_dir', default='ckpts/tmp')
parser.add_argument('--load_model', default='ckpts/RefineGNN-hcdr3/model.best')

parser.add_argument('--cdr_type', default='3')

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_tokens', type=int, default=100)
parser.add_argument('--k_neighbors', type=int, default=9)
parser.add_argument('--block_size', type=int, default=8)
parser.add_argument('--update_freq', type=int, default=1)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=21)
parser.add_argument('--num_rbf', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=50)

args = parser.parse_args()
print(args)

os.makedirs(args.save_dir, exist_ok=True)

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)

# loaders = []
# for path in [args.train_path, args.val_path, args.test_path]:
#     data = AntibodyDataset(path, cdr_type=args.cdr_type)
#     loader = StructureLoader(data.data, batch_tokens=args.batch_tokens, interval_sort=int(args.cdr_type))
#     loaders.append(loader)

# loader_train, loader_val, loader_test = loaders

model = HierarchicalDecoder(args).cuda()
optimizer = torch.optim.Adam(model.parameters())
if args.load_model:
    model_ckpt, opt_ckpt, model_args = torch.load(args.load_model)
    model = HierarchicalDecoder(model_args).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    model.load_state_dict(model_ckpt)
    optimizer.load_state_dict(opt_ckpt)

data = AntibodyDataset('data/seeds/3eak.jsonl', cdr_type=args.cdr_type)
loader = StructureLoader(data.data, batch_tokens=args.batch_tokens, interval_sort=int(args.cdr_type))

with torch.no_grad():
    for hbatch in loader:
        hX, hS, hL, hmask = completize(hbatch)
        generated_cdrs = {}

        for i in tqdm(range(300)):
            cdr3, ppl, _ = model.generate(hS, hL, hmask, return_ppl=True)
            generated_cdrs[cdr3[0]] = ppl.item()

        f = open('data/generated/3eak.json', 'w')
        f.write(json.dumps(generated_cdrs))
        f.close()