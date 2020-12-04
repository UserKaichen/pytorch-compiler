import re
import os
import sys
import shutil
import functools
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from save_data import save_data as _save_data
from quant_layer import QuantLayer
from vgg_imagenet import vgg
from quantop import activation_quantization


def get_data(model, output_dir='out'):
    def quant_forward(forward, m, bits):
        @functools.wraps(forward)
        def wrapper(input):
            # out = forward(input)
            out = input
            q = bits - m.exp
            return activation_quantization(out, bits=bits, q=q)[0]
        return wrapper

    def scale_fc_forward(forward, fc_scale, bias):
        @functools.wraps(forward)
        def wrapper(data):
            return forward(data) * fc_scale + bias
        return wrapper

    def print_act(forward, name, save_data, m=None, is_act=False):
        @functools.wraps(forward)
        def wrapper(input):
            img_flag = re.sub( "\D" , "", img_path.split(".")[0])
            if 'input' in name:
                for i in range(input.size(0)):
                    save_data(input[i], f'{name}.{img_flag}')
            output = forward(input)
            if 'output' in name:
                q = 7 - m.exp
                for i in range(output.size(0)):
                    save_data(output[i], f'{name}.{img_flag}', is_act=is_act, q=q)
            return output
        return wrapper


    def overloadBN(forward, m):
        @functools.wraps(forward)
        def wrapper(input):
            def expand(x):
                return x.view(1, -1, 1, 1)
            return expand(m.bn_k) * input + expand(m.bn_b)
        return wrapper


    def get_modules(model, instance):
        return [
            (name, m)
            for name, m in model.named_modules()
            if isinstance(m, instance)
        ]


    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for m in model.modules():
        if isinstance(m, QuantLayer):
            print('forward test quant')
            m.forward = quant_forward(m.forward, m, bits=7)

    save_data = functools.partial(_save_data, output_dir=output_dir)

    convs = get_modules(model, nn.Conv2d)
    bns = get_modules(model, nn.BatchNorm2d)
    fcs = get_modules(model, nn.Linear)
    quants = get_modules(model, QuantLayer)
    assert len(convs) == len(bns), "Each conv must be followed by bn"


    # Convert conv weight to int, then modify bn to recover the change
    for (_, conv), (_, bn) in zip(convs, bns):
        conv.weight.data = (conv.weight.data / conv.scale).round()
        bn.running_mean = bn.running_mean / conv.scale
        bn.weight.data = bn.weight.data * conv.scale

    # Convert fc to int, then modify forward to recover the change
    for fc_name, fc in fcs:
        fc.bias_ = fc.bias.clone()
        fc.bias.data[:] = 0.0
        fc.weight.data = (fc.weight.data / fc.scale).round()
        fc.forward = scale_fc_forward(fc.forward, fc.scale, fc.bias_)
        save_data(torch.as_tensor(fc.scale), f'{fc_name}.k', to_hex=True)

    # Convert bn to the form of `bn_k * input + bn_b`
    for name, m in bns:
        # m.bn_k = m.weight.data / (m.running_var.sqrt()+1e-6)
        # m.bn_b = -m.weight.data * m.running_mean / \
        #     (m.running_var.sqrt()+1e-6) + m.bias.data
        # m.bn_k = m.bn_k.half().float()
        # m.bn_b = m.bn_b.half().float()
        m.forward = overloadBN(m.forward, m)

    # Print input of every convs and fcs
    for name, m in convs + fcs:
        m.forward = print_act(m.forward, f'{name}.input', save_data)

    # Print output of every quants
    for name, m in quants:
        m.forward = print_act(m.forward, f'{name}.output', save_data, m=m)

    # For last output, print int version
    for name, m in quants[-1:]:
        m.forward = print_act(
            m.forward, f'{name}.output.int', save_data, m=m, is_act=True)

    return model

assert len(sys.argv) == 3, f'Usage: {sys.argv[0]} path/to/image path/to/output/dir'

img_path = sys.argv[1]
output_dir = sys.argv[2]

model = vgg()
# TODO: merge this into vgg.py
for m in model.modules():
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        m.register_buffer(f'scale', torch.tensor(0.0))
    if isinstance(m, torch.nn.BatchNorm2d):
        m.register_buffer(f'bn_k', torch.zeros_like(m.weight.data))
        m.register_buffer(f'bn_b', torch.zeros_like(m.weight.data))

# Load checkpoint
model.load_state_dict(torch.load('vgg_imagenet.pt'))

# Do quantization to int and print data
model = get_data(model, output_dir)
model = model.eval()

im = Image.open(img_path)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
data = transform(im).unsqueeze(0)  # Add batch dimension
data, _ = activation_quantization(data, bits=7)

model(data)

