import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x


n = Net()
example_input = torch.rand(1, 64, 127, 127)
module = torch.jit.trace(n, example_input)
module._c._fun_compile()


# print(module.graph)
# print(module(example_forward_input))
# print(module.graph)
