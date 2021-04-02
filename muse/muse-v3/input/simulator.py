import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(46, 21, 7, 6, 2)
    def forward(self, x):
        x = self.conv(x)
        return x

n = Net()
input = torch.rand(1, 46, 423, 423)
module = torch.jit.trace(n, input)
module._c._fun_compile()

