import os
import torch
import torch.optim as optim
from torch import nn


class Seq_32_32(nn.Module):
    def __init__(self):
        super(Seq_32_32, self).__init__()
        self.seq_32_32 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.seq_32_32(x)


class Seq_32x1_16(nn.Module):
    def __init__(self):
        super(Seq_32x1_16, self).__init__()
        self.seq_32x1_16 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(32 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_32x1_16(x)


class Seq_32x1_16_filt3(nn.Module):
    def __init__(self):
        super(Seq_32x1_16_filt3, self).__init__()
        self.seq_32x1_16_filt3 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(1, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(32 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_32x1_16_filt3(x)


class Seq_32x2_16(nn.Module):
    def __init__(self):
        super(Seq_32x2_16, self).__init__()
        self.seq_32x2_16 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(32 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_32x2_16(x)


class Seq_64x1_16(nn.Module):
    def __init__(self):
        super(Seq_64x1_16, self).__init__()
        self.seq_64x1_16 = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Flatten(),
            nn.Linear(64 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_64x1_16(x)


class Seq_embed_32x1_16(nn.Module):
    def __init__(self):
        super(Seq_embed_32x1_16, self).__init__()
        self.seq_embed_32x1_16 = nn.Sequential(
            nn.Conv2d(20, 8, kernel_size=(1, 5), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(8, 64, kernel_size=(1, 5), padding='same'),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.seq_embed_32x1_16(x)
    

class Seq_LSTM_32x1_16(nn.Module):
    def __init__(self):
        super(Seq_LSTM_32x1_16, self).__init__()
        
        self.lstm = nn.LSTM(input_size=20, hidden_size=32, batch_first=True, bidirectional=True)
        self.body = nn.Sequential(
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32)
        c0 = torch.zeros(2, x.size(0), 32)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.body(out[:, -1, :])
     
        return out
    
    
class Seq_LSTM_32x2_16(nn.Module):
    def __init__(self):
        super(Seq_LSTM_32x2_16, self).__init__()
        
        self.lstm = nn.LSTM(input_size=20, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        self.body = nn.Sequential(
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), 32)
        c0 = torch.zeros(4, x.size(0), 32)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.body(out[:, -1, :])
     
        return out
    
    
class Seq_LSTM_64x1_16(nn.Module):
    def __init__(self):
        super(Seq_LSTM_64x1_16, self).__init__()
        
        self.lstm = nn.LSTM(input_size=20, hidden_size=64, batch_first=True, bidirectional=True)
        self.body = nn.Sequential(
            nn.Linear(128, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64)
        c0 = torch.zeros(2, x.size(0), 64)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.body(out[:, -1, :])
     
        return out
