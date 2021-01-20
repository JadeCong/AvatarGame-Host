# Reference: https://github.com/szagoruyko/pytorchviz

import torch
from torchvision.models import AlexNet
from torchviz import make_dot # pip install torchviz(dependency:graphviz)


# construct the network and set the input
x = torch.rand(8,3,256,512)
model = AlexNet()
y = model(x)

# construct the network graph(3 methods)
graph = make_dot(y)
# graph = make_dot(y, params=dict(model.named_parameters()))
# graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))

# save the network graph
graph.view()
# graph.render('espnet_model', view=False)
