import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from data import *
from ensemble_stack.interpreter.model import Interpreter
import sys

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('mps')


if __name__ == '__main__':
    validation_loader = DataLoader(SampleData(), batch_size=100)
    loaded_models = load_all_models()
    output = get_stacked_predictions(loaded_models, validation_loader)
    stacked_dataset = pd.DataFrame(
        columns=list(loaded_models.keys()), data=output)

    interpreter_loader = DataLoader(InterpreterData(
        stacked_dataset.values), batch_size=100, shuffle=True)
    interpreter = Interpreter().to(device)
    optimizer = optim.Adam(interpreter.parameters())
    loss_fn = torch.nn.MSELoss()

    for epoch in range(20):
        train_loss = 0.0

        for step, data in enumerate(interpreter_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = interpreter(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(inputs)

        print(epoch+1, train_loss / len(interpreter_loader.sampler))

    torch.save(interpreter.state_dict(), 'weights/interpret_16x1.pth')
