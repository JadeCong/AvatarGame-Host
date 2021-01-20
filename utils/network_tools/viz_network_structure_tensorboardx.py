# Reference: https://tensorboardx.readthedocs.io/en/latest/tutorial.html
#            https://github.com/lanpa/tensorboardX

import sys, os
from tensorboardX import SummaryWriter # pip install tensorboard(for tensorflow)/tensorboardX(for pytorch)

import torch
import torch.nn as nn
import numpy as np


# construct the network and set the input
dummy_input = (torch.zeros(1, 3),)

class LinearInLinear(nn.Module):
    def __init__(self):
        super(LinearInLinear, self).__init__()
        self.l = nn.Linear(3, 5)

    def forward(self, x):
        return self.l(x)

# construct and save the network graph
with SummaryWriter(comment='LinearInLinear') as w:
    w.add_graph(LinearInLinear(), dummy_input, True)
