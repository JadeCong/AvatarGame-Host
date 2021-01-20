# Reference: https://github.com/microsoft/tensorwatch

import torch
import torchvision.models
import tensorwatch as tw # pip install tensorwatch


# construct the network and set the input
x = torch.rand(8,3,256,512)
alexnet_model = torchvision.models.alexnet()

# construct and save the network graph
# tw.draw_model(alexnet_model, [1, 3, 224, 224])
tw.model_stats(alexnet_model, [1, 3, 224, 224])
