# Reference: https://github.com/Lyken17/pytorch-OpCounter

import torch
from torchvision.models import resnet50
from thop import profile # pip install thop

model = resnet50()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
