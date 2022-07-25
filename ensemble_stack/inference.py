import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from ensemble_stack.interpreter.data import *
from ensemble_stack.interpreter.model import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('mps')


def get_stacked_prediction(sequences):
    sequences = [seq.center(20, 'J') for seq in sequences]
    model_list = load_all_models()
    output = []
    mapper = load_mapper('ensemble_stack/regression/data/mapper')
    oh_sequences = torch.stack([one_hot_encode(seq, mapper)
                               for seq in sequences]).to(device)

    for idx, model in enumerate(model_list.values()):
        predictions = model(oh_sequences).cpu().detach().numpy().flatten()
        output.append([])
        output[idx].append(predictions)
    output = np.vstack(output)
    output = output.reshape(output.shape[0], output.shape[1]).T
    
    interpreter = Interpreter().to(device)
    interpreter.load_state_dict(torch.load('ensemble_stack/interpreter/weights/interpret_16x1.pth'))
    return(interpreter(torch.from_numpy(output).to(device)))

