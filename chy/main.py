import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
    
    def faketool(self, x):
        return F.relu(self.conv1(x))

    def forward(self, x):
        x = self.faketool(x)
        return F.relu(self.conv2(x))

n = Net()
example_forward_input = torch.rand(1, 1, 13, 13)
module = torch.jit.trace(n, example_forward_input)
# module(example_forward_input)
module._c._fun_compile()


# print(module.graph)
# print(module(example_forward_input))
# print(module.graph)