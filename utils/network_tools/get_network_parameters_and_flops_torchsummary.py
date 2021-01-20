# Reference: https://github.com/sksq96/pytorch-summary
#            https://github.com/nmhkahn/torchsummaryX

# for torchsummary
from torchsummary import summary # pip install torchsummary
import torch
from torchvision import models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg16().to(device)

summary(vgg, (3, 224, 224))


# for torchsummaryX
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummaryX import summary # pip install torchsummaryX


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# summary(Net(), torch.zeros((1, 1, 28, 28)))
