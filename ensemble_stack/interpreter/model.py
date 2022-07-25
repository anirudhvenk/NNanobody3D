from torch import nn

class Interpreter(nn.Module):
    def __init__(self):
        super(Interpreter, self).__init__()
        self.interpreter = nn.Sequential(
            nn.Linear(18, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.interpreter(x)