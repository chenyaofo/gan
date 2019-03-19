import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(Discriminator, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.module(inputs)
    