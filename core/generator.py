import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(Generator, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs):
        return self.module(inputs)
