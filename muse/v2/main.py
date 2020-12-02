import io
import os
import sys
import math
import time
import torch
import shutil
import threading
import subprocess
import numpy as np
import struct as st


class makenet():
    def __init__(self):
        self.bns = [""]
        self.layer = []
        self.convs = []
        self.pools = []
        self.qulist = []
        self.fcinit = []
        self.fcford = []
        self.avginit = []
        self.avgford = []
        self.classname = ""
        self.fvggnet = open("debug/vggnet.py", "a")
        self.fmakenet = open("debug/makenet.py", "a")

    """
    description: Add import code form vgg_imagenet.py to makenet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def make_config(self, filename):
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.startswith("#") is False:
                    if "import math" in line:
                        self.fmakenet.write("import sys\n")
                    if "from quant_layer import " in line:
                        self.fmakenet.write("sys.path.append(\"input\")\n")
                    self.fmakenet.write(line)
                if line.strip() == "}":
                    break
        self.fmakenet.write('\n')

    """
    description: Add "class BasicBlock" code form vgg_imagenet.py to makenet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def make_block(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.startswith("#") is False:
                    if "class BasicBlock" not in line and read_flag == 0:
                        continue
                    self.classname = "BasicBlock"
                    read_flag = 1
                    if line.startswith("class ") and "BasicBlock" not in line:
                        break
                    self.fmakenet.write(line)

    """
    description: Add "class vgg" code form vgg_imagenet.py to makenet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def make_class(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "class vgg" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "__init__" not in line:
                        break
                    self.fmakenet.write(line)

    """
    description: Add "def make_layers" code form vgg_imagenet.py to makenet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def make_layers(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def make_layers" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "make_layers" not in line:
                        break
                    self.fmakenet.write(line)

    """
    description: Add "def padding" code form vgg_imagenet.py to makenet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def make_padding(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def padding" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "padding" not in line:
                        break
                    self.fmakenet.write(line)

    """
    description: Add "def forward" code form vgg_imagenet.py to makenet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def make_forward(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if line.strip().startswith("class "):
                        self.classname = line.split(' ', 1)[1].split('(', 1)[0]
                    if self.classname != "vgg":
                        continue
                    if "def forward" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "forward" not in line:
                        break
                    self.fmakenet.write(line)

    """
    description: Add "def _initialize_weights" code form vgg_imagenet.py to makenet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def make_weight(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def _initialize_weights" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if "= vgg(" in line:
                        break
                    self.fmakenet.write(line)

    """
    description: Add "main" code for form vgg_imagenet.py to makenet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def make_main(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            for line in file:
                if line.strip().startswith("#") is False and line.strip().startswith("print") is False:
                    if "= vgg(" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    self.fmakenet.write(line)
                if "= vgg(" in line:
                    outvgg = "{}{}{}".format("print(\"vgg_module = \", ", line.strip().split(" ")[0], ")\n")
                    self.fmakenet.write(outvgg)
                    return

    """
    description: Add import code to vggnet.py
    parameter: NULL
    return value: NULL
    """
    def _make_head(self):
        self.fvggnet.write("import sys\n")
        self.fvggnet.write("import torch\n")
        self.fvggnet.write("import torch.nn as nn\n")
        self.fvggnet.write("import torch.nn.functional as F\n")
        self.fvggnet.write("sys.path.append(\"input\")\n")
        self.fvggnet.write("from quant_layer import QuantLayer\n\n")
        self.fvggnet.write("class Net(nn.Module):\n")
        self.fvggnet.write("    def __init__(self, num_classes=10):\n")
        self.fvggnet.write("        super(Net, self).__init__()\n")

    """
    description: Add main code to vggnet.py
    parameter: NULL
    return value: NULL
    """
    def _make_tail(self):
        self.fvggnet.write("\nn = Net()\n")
        self.fvggnet.write("example_input = torch.rand(1, 3, 224, 224)\n")
        self.fvggnet.write("module = torch.jit.trace(n, example_input)\n")
        self.fvggnet.write("module._c._fun_compile()\n")

    """
    description: Add "self.*" code to vggnet.py
    parameter: NULL
    return value: NULL
    """
    def _make_init(self):
        for i in range(len(self.layer)):
            if "Conv2d" in str(self.layer[i]):
                self.convs.append(self.layer[i])
            elif "BatchNorm2d" in str(self.layer[i]):
                if self.bns[0] == "True":
                    self.bns.append(self.layer[i])
            elif "MaxPool2d" in str(self.layer[i]):
                self.pools.append(self.layer[i])

        for i in range(len(self.pools)):
            self.fvggnet.write("{}{}{}{}{}".format("        self.pool", i+1, " = nn.",
                                             self.pools[i].split(":")[1].strip(), "\n"))
        for j in range(len(self.convs)):
            self.fvggnet.write("{}{}{}{}{}".format("        self.conv", j+1, " = nn.",
                                             self.convs[j].split(":")[1].strip(), "\n"))
            if self.bns[0] == "True":
                self.fvggnet.write("{}{}{}{}{}".format("        self.bn", j+1, " = nn.",
                                                 self.bns[j+1].split(":")[1].strip(), "\n"))
        self.fvggnet.write("\n")

        for i in range(len(self.avginit)):
            self.fvggnet.write("{}{}{}".format("        ", self.avginit[i], "\n"))
        for i in range(len(self.qulist)):
            if "= QuantLayer()" in self.qulist[i]:
                self.fvggnet.write("{}{}{}".format("        ", self.qulist[i], "\n"))
        for i in range(len(self.fcinit)):
            self.fvggnet.write("{}{}{}".format("        ", self.fcinit[i], "\n"))

    """
    description: Add "def padding" code from vgg_imagenet.py to vggnet.py
    parameter: filename —— vgg_imagenet.py
    return value: NULL
    """
    def _make_padding(self, filename):
        read_flag = 0
        self.fvggnet.write("\n")

        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def padding" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    if line.strip().startswith("def ") and "padding" not in line:
                        break
                    self.fvggnet.write(line)

    """
    description: Add "x = F.relu(self.bn*(self.conv*(x))) or x = F.relu(self.conv*(x))" code to vggnet.py
    parameter: convcnt —— The number of layers
    return value: NULL
    """
    def _make_convlay(self, convcnt):
        x = "F.relu"
        end = "))\n"

        if self.bns[0] == "True":
            x = "F.relu(self.bn"
            end = ")))\n"
        convcmd = "{}{}{}{}{}{}{}".format("        x = ", x, str(convcnt),
                                          "(self.conv",str(convcnt), "(x", end)
        self.fvggnet.write(convcmd)

    """
    description: Add "x = self.pool*(x)" code to vggnet.py
    parameter: NULL
    return value: NULL
    """
    def _make_forward(self):
        convcnt = poolcnt = 0
        self.fvggnet.write("    def forward(self, x):\n")

        for i in range(len(self.layer)):
            if "Conv2d" in str(self.layer[i]):
                convcnt += 1
                if self.bns[0] == "True":
                    if "BatchNorm2d" in str(self.layer[i+1]):
                        if "ReLU" in str(self.layer[i+2]):
                             self._make_convlay(convcnt)
                else:
                    if "ReLU" in str(self.layer[i+1]):
                        self._make_convlay(convcnt)
            elif "MaxPool2d" in str(self.layer[i]):
                poolcnt += 1
                poolcmd = "{}{}{}".format("        x = self.pool", str(poolcnt), "(x)\n")
                self.fvggnet.write(poolcmd)

    """
    description: Add "x = self.avgpool_*(x)" code to vggnet.py
    parameter: NULL
    return value: NULL
    """
    def _make_avgpool(self):
        self.fvggnet.write("{}{}".format("        x = self.padding(x)", "\n"))
        for i in range(len(self.avgford)):
            self.fvggnet.write("{}{}{}".format("        ", self.avgford[i], "\n"))
            self.fvggnet.write("{}{}{}".format("        ", self.qulist[i+int(len(self.qulist)/2)], "\n"))
        for i in range(len(self.fcford)):
            self.fvggnet.write("{}{}{}".format("        ", self.fcford[i], "\n"))

        self.fvggnet.write("        return x\n")

    """
    description: Get data from layerinfo
    parameter: file —— layerinfo
    return value: NULL
    """
    def splicing_layers(self, file):
        with open(file, "r") as f:
            lines = f.readline()
            while lines:
                lines = f.readline()
                self.layer.append(lines)

    """
    description: Get op code from vgg_imagenet.py
    parameter: code_path —— vgg_imagenet.py    operator —— op
    return value: NULL
    """
    def get_op_code(self, code_path, operator):
        with open(code_path, encoding='utf-8') as f:
            lines = f.readline()
            while lines:
                lines = f.readline()
                if operator == "bn":
                    if "self.layers = self.make_layers" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            return lines.rsplit(" ", 3)[3][:-1]
                elif operator == "avgpool":
                    if "= nn.AvgPool2d(" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.avginit.append(lines)
                    elif "x = self.avgpool" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.avgford.append(lines)
                elif operator == "quant":
                    if "self.quant_avg" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.qulist.append(lines)
                elif operator == "fc":
                    if "self.quant_fc1 =" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.fcinit.append(lines)
                            while lines.strip() != ")":
                                lines = f.readline()
                                lines = lines.strip()
                                if lines.startswith("#") is False:
                                    self.fcinit.append(lines)
                    elif "= x.view(" in lines or "= self.classifier(" in lines:
                        lines = lines.strip()
                        if lines.startswith("#") is False:
                            self.fcford.append(lines)

class load_pt():
    def __init__(self):
        self.in_q = "5"
        self.layermsg = ""
        self.layer_cnts = 0
        self.in_feature_h = 0
        self.in_feature_w = 0
        self.padding_param = ""
        self.layers = [["", ""]]
        self.in_channels_bf = ""
        self.out_channels_bf = ""


    def get_layer_info(self, path, flag):
        """
        description:
                    Get specific layer information
        parameters:
                    path:       The path of vggnet.log
                    flag:       The flag of specific layer
        return code:
                    layer_info: Get specific layer information
        """
        layer_cnt = 0
        layer_info = []
        find_flg = flag

        if int(str(find_flg).split(":", 1)[1]) == 0:
            print("please do not send layer_num:0")
            exit(0)
        next_flg = "{}{}".format("layer_num:", int(str(find_flg).split(":", 1)[1]) + 1)
        with open(path, 'r') as file_read:
            for line in file_read:
                if flag in line:
                    layer_cnt += 1
                if next_flg in line:
                    break
                if "compiler_end" in line:
                    break
                if layer_cnt == 1:
                    layer_info.append(line)

        return layer_info

    def getoneDimList(self, newlist):
        """
        description:
                    Convert high-dimensional list to one-dimensional
        parameters:
                    newlist:    High-dimensional list name
        return code:
                    oneDimList: one-dimensional list name
        """
        oneDimList = []
        for element in newlist:
            if not isinstance(element, list):
                oneDimList.append(element)
            else:
                oneDimList.extend(self.getoneDimList(element))
        return oneDimList

    def write_pt_data(self, filename, filedata, scale):
        """
        description:
                    Write pt data to file
        parameters:
                    filename: Relative path of tensor file
                    filedata: Pt data of Relative files
                    scale:    The scale of weight
        return code:
                    None
        """
        path = "{}{}{}".format("debug/output/", filename, ".txt")
        with open(path, 'w') as fw:
            if "quant_" in filename:
                fw.write(str(filedata))
            else:
                conver = filedata.tolist()
                if filedata.dim() == 0:
                    fw.write(str(conver))
                    return
                elif filedata.dim() > 1:
                    conver = self.getoneDimList(conver)
                for i in range(len(conver)):
                    if "bn.bn" in filename or ".bias" in filename:
                        hexdata = "{:X}".format(st.unpack('H', st.pack('e', conver[i]))[0])
                        fw.write(str(hexdata))
                    elif "weight" in filename:
                        scalelist = scale.tolist()
                        weight = round(conver[i] / scalelist)
                        fw.write(str(weight))
                    else:
                        fw.write(str(conver[i]))
                    fw.write('\n')

        print("%s write data success" % path)

    def write_to_file(self, fw, config):
        """
        description:
                    Write configuration information to file
        parameters:
                    filename: Relative path of tensor file
                    config:   Configuration to be written to file
        return code:
                    None
        """
        for i in range(len(config)):
            if "\n" in config[i]:
                fw.write("\n")
                continue
            fw.write(config[i])
            fw.write("\n")

    def out_to_in(self, out_feature_h, out_feature_w, in_channels, out_channels, type):
        """
        description:
                    The output of the previous layer to the input of the next layer
        parameters:
                    out_feature_h: output feature of in the high direction
                    out_feature_w: output feature of in the width direction
                    in_channels:   Number of channels in the input image
                    out_channels:  Number of channels produced by the convolution
                    type:          type of each layer
        return code:
                    None
        """
        if type == "pool" and int(out_feature_h) == 7 and "padding param: " in self.padding_param:
            self.in_feature_h = str(int(self.padding_param.split(" ", 6)[4].split(":")[1]))
            self.in_feature_w = str(int(self.padding_param.split(" ", 6)[5].split(":")[1]))
        else:
            self.in_feature_h = out_feature_h
            self.in_feature_w = out_feature_w
        if type != "pool":
            self.in_channels_bf = in_channels
            self.out_channels_bf = out_channels

    def get_all_params(self, config, param):
        """
        description:
                    Get all the parameters
        parameters:
                    config:        Configuration information list for input
                    param:         Parameter information list
        return code:
                    config:        Configuration information list for output
                    out_feature_h: output feature of in the high direction
                    out_feature_w: output feature of in the width direction
                    kernel_size_x: Size of the convolving kernel in the x direction
                    kernel_size_y: Size of the convolving kernel in the y direction
                    in_channels:   Number of channels in the input image
                    out_channels:  Number of channels produced by the convolution
                    stride_x:      Stride of the convolution in the x direction  
                    stride_y:      Stride of the convolution in the y direction  
        """
        stride_x = stride_y = ""
        in_channels = out_channels = ""
        out_feature_w = out_feature_h = ""
        kernel_size_x = kernel_size_y = ""

        for i in range(len(param)):
            if "in_channels" in param[i] or "in_features_y" in param[i]:
                in_channels = "{}{}".format("input channel num  = ", param[i].split(":")[1].strip())
                if in_channels.strip() == self.out_channels_bf or len(self.out_channels_bf) == 0:
                    config.append(in_channels)
                else:
                    config.append("{}{}".format("input channel num  = ", self.out_channels_bf.split("=")[1].strip()))
            elif "out_channels" in param[i] or "out_features_y" in param[i]:
                out_channels = "{}{}".format("output channel num = ", param[i].split(":")[1].strip())
                config.append(out_channels)
            elif "feature_map_size_x" in param[i]:
                out_feature_h = param[i].split(":")[1].strip()
            elif "feature_map_size_y" in param[i]:
                out_feature_w = param[i].split(":")[1].strip()
            elif "stride_x" in param[i]:
                stride_x = "{}{}".format("stride_x = ", param[i].split(":")[1].strip())
            elif "stride_y" in param[i]:
                stride_y = "{}{}".format("stride_y = ", param[i].split(":")[1].strip())
            elif "kernel_size_x" in param[i]:
                kernel_size_x = param[i].split(":")[1].strip()
            elif "kernel_size_y" in param[i]:
                kernel_size_y = param[i].split(":")[1].strip()

        return config, out_feature_w, out_feature_h, kernel_size_x, kernel_size_y, \
               in_channels, out_channels, stride_x, stride_y

    def write_common_params(self, config, param, in_q, out_q, relu, layer_type):
        """
        description:
                    Write common parameters to the configuration file
        parameters:
                    config:        Configuration information list
                    param:         Parameter information list
                    in_q:          input q number for quantization
                    out_q:         output q number for quantization
                    relu:          relu information
                    layer_type:    the type of each layer
        return code:
                    out_feature_h: output feature of in the high direction
                    out_feature_w: output feature of in the width direction
                    in_channels:   Number of channels in the input image
                    out_channels:  Number of channels produced by the convolution
        """
        config, out_feature_w, out_feature_h, kernel_size_x, kernel_size_y, in_channels, \
        out_channels, stride_x, stride_y = self.get_all_params(config, param)

        if layer_type == "pool":
            config.append(self.in_channels_bf)
            config.append(self.out_channels_bf)
            ratio = int(kernel_size_x)
            out_feature_h = str(int(int(float(self.in_feature_h)) / ratio))
            out_feature_w = str(int(int(float(self.in_feature_w)) / ratio))
        config.append("{}{}".format("input feature_h = ", self.in_feature_h))
        config.append("{}{}".format("input feature_w = ", self.in_feature_w))
        if layer_type != "pool":
            if (layer_type == "fc"):
                if self.in_feature_h == "1" and self.in_feature_w == "1":
                    kernel_size_x = out_feature_h = self.in_feature_h
                    kernel_size_y = out_feature_w = self.in_feature_w
        config.append("{}{}".format("output feature_h = ", out_feature_h))
        config.append("{}{}".format("output feature_w = ", out_feature_w))
        config.append("\n")
        if layer_type == "conv":
            config.append("padding = 1")
        else:
            config.append("padding = 0")
        config.append(in_q)
        config.append(out_q)
        config.append("{}{}{}{}".format("kernel size = ", kernel_size_x, "×", kernel_size_y))
        if "relu " in relu:
            config.append(relu)
        else:
            config.append("relu 0")
        config.append("w_quan   \"MIX\"")
        config.append("\n")

        if layer_type != "pool":
            config.append("【Pattern Pruning config】")
            config.append(
                "pattern_dic.txt存的是16种pattern所对应的mask，顺序是先横着走再换行；pattern_idx所对应的是每个kernel所采用的pattern编号，顺序与weight顺序一致；weight.txt是不做压缩的，仅作参考；weight_nonzero是最终给到芯片的权重，已经做了4pattern压缩存储。")
            config.append("\n")
            config.append("pattern: 9-pattern")
            config.append("\n")
            config.append("BN中，k=0.01 (0x211F)， b=0")
            config.append("\n")

        return out_feature_w, out_feature_h, in_channels, out_channels

    def write_conv_config(self, fw):
        """
        description:
                    Write conv layer configuration information
        parameters:
                    fw:         File descriptor of the configuration file
        return code:
                    None
        """
        in_q = out_q = ""
        layer_num = relu = ""
        param = config = []

        for i in range(len(self.layermsg)):
            if self.layermsg[i].startswith("layer_num:") is True:
                layer_num = self.layermsg[i].split(" ")[0].strip().split(":")[1]
            elif "in_channels" in self.layermsg[i]:
                param = self.layermsg[i].split(" ")
            elif "relu param" in self.layermsg[i]:
                relu = "relu √"
            elif "in_q =" in self.layermsg[i]:
                in_q = self.layermsg[i]
            elif "out_q =" in self.layermsg[i]:
                out_q = self.layermsg[i]

        out_feature_w, out_feature_h, in_channels, out_channels = \
            self.write_common_params(config, param, in_q, out_q, relu, "conv")

        config.append(
            "{}{}{}".format("act0 file               : layers.", str(int(layer_num) - 1), ".conv.input.6.txt"))
        config.append(
            "{}{}{}".format("output file             : layers.", str(int(layer_num) - 1), ".quant.output.6.txt"))
        config.append("{}{}{}".format("weight file             : layers.", str(int(layer_num) - 1), ".conv.weight.txt"))
        config.append("{}{}{}".format("bn k file               : layers.", str(int(layer_num) - 1), ".bn.bn_k.txt"))
        config.append("{}{}{}".format("bn b file               : layers.", str(int(layer_num) - 1), ".bn.bn_b.txt"))

        self.write_to_file(fw, config)
        self.out_to_in(out_feature_h, out_feature_w, in_channels, out_channels, "conv")

    def write_pool_config(self, fw):
        """
        description:
                    Write pool layer configuration information
        parameters:
                    fw:         File descriptor of the configuration file
        return code:
                    None
        """
        in_q = out_q = ""
        param = config = []
        poolname = "Maxpooling"

        for i in range(len(self.layermsg)):
            if "padding param:" in self.layermsg[i]:
                self.padding_param = self.layermsg[i]
            elif "param:" in self.layermsg[i]:
                param = self.layermsg[i].split(" ")
            if "avgpool" in self.layermsg[i]:
                poolname = "Average pooling"
            elif "in_q =" in self.layermsg[i]:
                in_q = self.layermsg[i]
            elif "out_q =" in self.layermsg[i]:
                out_q = self.layermsg[i]

        out_feature_w, out_feature_h, in_channels, out_channels = \
            self.write_common_params(config, param, in_q, out_q, "", "pool")

        self.write_to_file(fw, config)
        fw.write(poolname)

        self.out_to_in(out_feature_h, out_feature_w, in_channels, out_channels, "pool")

    def write_fc_config(self, fw, quant_list):
        """
        description:
                    Write fc layer configuration information
        parameters:
                    fw:         File descriptor of the configuration file
                    quant_list: Save the list of quantlayer and classifier
        return code:
                    None
        """
        in_q = out_q = ""
        fc_name = relu = ""
        param = config = []

        for i in range(len(self.layermsg)):
            if " param:" in self.layermsg[i]:
                fc_name = self.layermsg[i].split(" ")[0].strip()
                param = self.layermsg[i].split(" ")
            elif "in_q =" in self.layermsg[i]:
                in_q = self.layermsg[i]
            elif "out_q =" in self.layermsg[i]:
                out_q = self.layermsg[i]
            elif "ReLU" in self.layermsg[i]:
                relu = "relu √"

        out_feature_w, out_feature_h, in_channels, out_channels = \
            self.write_common_params(config, param, in_q, out_q, relu, "fc")

        classifier = []
        for i in range(len(quant_list)):
            if (i % 2) == 0:
                if "classifier" in quant_list[i] and "weight" in quant_list[i]:
                    classifier.append("{}{}{}".format(quant_list[i].rsplit(".")[0], ".", quant_list[i].rsplit(".")[1]))

        config.append("{}{}{}".format("act0 file               : ", classifier[int(fc_name[2]) - 1], ".input.6.txt"))
        config.append("{}{}{}{}".format("output file             : ", "quant_", fc_name, ".output.6.txt"))
        config.append("{}{}{}".format("weight file             : ", classifier[int(fc_name[2]) - 1], ".weight.txt"))
        config.append("{}{}{}".format("bn k file               : ", classifier[int(fc_name[2]) - 1], ".k.txt"))
        config.append("{}{}{}".format("bn b file               : ", classifier[int(fc_name[2]) - 1], ".bias.txt"))

        self.write_to_file(fw, config)
        self.out_to_in(out_feature_h, out_feature_w, in_channels, out_channels, "fc")

    def write_layer_config(self, quant_list):
        """
        description:
                    Write configuration information for each layer
        parameters:
                    quant_list: Save the list of quantlayer and classifier
        return code:
                    None
        """
        layername = self.layermsg[0].split(":", 1)[1].split(" ", 1)[0].strip('\n')
        if layername == "1":
            path = "{}{}".format("debug/output/config", ".txt")
        else:
            path = "{}{}{}".format("debug/output/config_", str(int(layername) - 1), ".txt")

        with open(path, 'w') as fw:
            layer_type = " "
            for i in range(len(self.layermsg)):
                if "layer type:" in self.layermsg[i]:
                    if self.layermsg[i].startswith("layer_num:1 ") is True:
                        layer_type = self.layermsg[i].split(":", 3)[2]
                    else:
                        layer_type = self.layermsg[i].split(":", 5)[2].split(" ")[0]
            if "conv" in layer_type:
                self.write_conv_config(fw)
            elif "pool" in layer_type:
                self.write_pool_config(fw)
            elif "fc" in layer_type:
                self.write_fc_config(fw, quant_list)
            else:
                self.print("Unknown layer type...")
                return

        print("%s write config success" % path)

    def deal_out_in_q(self, quant_list, data):
        """
        description:
                    Handle out_q and in_q function
        parameters:
                    quant_list: Save the list of quantlayer and classifier
                    data:       Data of each quantlayer or fclayer
        return code:
                    None
        """
        bits = 7
        if "pool" in self.layermsg:
            out_q = self.in_q
        else:
            out_q = bits - math.ceil(math.log2(0.5 * data))

        self.layermsg.append("{}{}".format("in_q = ", self.in_q))
        self.layermsg.append("{}{}".format("out_q = ", out_q))
        self.in_q = out_q
        write_config = threading.Thread(target=self.write_layer_config,
                                        args=(quant_list, ))
        write_config.start()
        write_config.join()

    def splicing_output(self, num, flag, quant_list):
        """
        description:
                    Format splicing output
        parameters:
                    num:        Number of each layer
                    flag:       Start mark of each layer (subscript)
                    quant_list: Save the list of quantlayer and classifier
        return code:
                    None
        """
        for i in range(len(self.layers)):
            if "quant.alpha" in self.layers[i][0]:
                data = self.layers[i][1]
                if num == 0:
                    self.deal_out_in_q(quant_list, data)
                    return

        conv = scale = weight = ""
        for i in range(num):
            name = self.layers[flag + i][0]
            data = self.layers[flag + i][1]
            if "quant.alpha" in name:
                self.deal_out_in_q(name, data)
                continue
            elif ".scale" in name or ".conv.weight" in name:
                if ".scale" in name:
                    scale = data
                else:
                    conv = name
                    weight = data
                if len(conv) and len(str(scale)) and len(str(weight)):
                    write_data = threading.Thread(target=self.write_pt_data,
                                                  args=(conv, weight, scale))
                    write_data.start()
                    write_data.join()
                    continue
            elif ".bn.bn_b" in name or ".bn.bn_k" in name:
                write_data = threading.Thread(target=self.write_pt_data,
                                              args=(name, data, scale))
                write_data.start()
                write_data.join()

    def get_layercount(self, filename):
        """
        description:
                    Get the number of vggnet's layers
        parameters:
                    filename: Relative path of tensor file
        return code:
                    None
        """
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip() == "":
                    break
                elif ": BasicBlock(" in line or "Pool2d(" in line or "Linear" in line:
                    self.layer_cnts += 1

    def get_tensorinfo(self, filename):
        """
        description:
                    Get tensor input information
        parameters:
                    filename: Relative path of tensor file
        return code:
                    None
        """
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip() == "":
                    continue
                elif "torch.rand" in line:
                    self.in_feature_h = line.split("(")[1].split(")")[0].split(",", 4)[2].strip()
                    self.in_feature_w = line.split("(")[1].split(")")[0].split(",", 4)[3].strip()
                    break

def gen_fpga(filepath):
    os.chdir(filepath)
    cmd_list = ["rm -rf config*txt cfg_*txt *bn* *bias* *alpha* *weight* data_for_fpga/",
                "cp -af ../debug/output/* .",
                "python ../input/config_gen_file_v3_0.py -d ../imagenet_fixed_v4_1/ \
                       -n imagenet_fixed_v4_1_img6 -f",
                "mv data_for_fpga ../output"]

    for i in range(len(cmd_list)):
        os.system(cmd_list[i])

def gen_txt(loadpt):
    """
    description:
                Load pt file and format output
    parameters:
                loadpt: The Class of load_pt
    return code:
                None
    """
    counts = 0
    name_list = []
    data_list = []
    quant_list = []
    onelayer_cnt = []

    loadpt.get_tensorinfo("debug/vggnet.py")
    loadpt.get_layercount("debug/layerinfo")

    with open(pt_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        dict = torch.load(buffer, map_location=torch.device('cpu'))
        for k, v in dict.items():
            if "quant_" in k or "classifier." in k:
                quant_list.append(k)
                quant_list.append(v)
            name_list.append(k)
            data_list.append(v)

    for i in range(loadpt.layer_cnts):
        layer = "{}{}{}".format("layers.", i, ".")
        for j in range(len(name_list)):
            if layer in name_list[j]:
                loadpt.layers.append([name_list[j], data_list[j]])
                counts += 1
        onelayer_cnt.append(str(counts))
        counts = 0

    del (loadpt.layers[0])
    logpath = "{}{}".format(os.getcwd(), "/debug/vggnet.log")
    for i in range(loadpt.layer_cnts):
        layername = "{}{}".format("layer_num:", str(i + 1))
        loadpt.layermsg = loadpt.get_layer_info(logpath, layername)
        loadpt.splicing_output(int(onelayer_cnt[i]), counts, quant_list)
        counts += int(onelayer_cnt[i])

    scale = fcname = weight = ""
    for i in range(len(quant_list)):
        tmpstr = str(quant_list[i])
        if ".scale" in tmpstr or ".weight" in tmpstr:
            if ".scale" in tmpstr:
                scale = quant_list[i + 1]
            else:
                fcname = quant_list[i]
                weight = quant_list[i + 1]
            if len(fcname) and len(str(scale)) and len(str(weight)):
                write_data = threading.Thread(target=loadpt.write_pt_data,
                                              args=(fcname, weight, scale))
                write_data.start()
                write_data.join()
                continue
        elif "quant_" in tmpstr or "classifier" in tmpstr:
            write_data = threading.Thread(target=loadpt.write_pt_data,
                                          args=(quant_list[i], quant_list[i + 1], scale))
            write_data.start()
            write_data.join()


def gen_net(mknet, filename):
    """
    description:
                Generate model files
    parameters:
                mknet:    The Class of makenet
                filename: The network name that the compiler can resolve
    return code:
                None
    """
    mknet.make_config(filename)
    mknet.make_block(filename)
    mknet.make_class(filename)
    mknet.make_layers(filename)
    mknet.make_padding(filename)
    mknet.make_forward(filename)
    mknet.make_weight(filename)
    mknet.make_main(filename)
    mknet.fmakenet.close()

    os.system("python3 debug/makenet.py > debug/layerinfo")
    mknet.splicing_layers("debug/layerinfo")
    mknet.bns[0] = mknet.get_op_code(filename, "bn")
    mknet.get_op_code(filename, "avgpool")
    mknet.get_op_code(filename, "weight")
    mknet.get_op_code(filename, "quant")
    mknet.get_op_code(filename, "fc")

    mknet._make_head()
    mknet._make_init()
    mknet._make_padding(filename)
    mknet._make_forward()
    mknet._make_avgpool()
    mknet._make_tail()
    mknet.fvggnet.close()
    print("make vggnet.py successfully")
        
def output(cnt, total, msg):
    """
    description:
                Output step progress bar
    parameters:
                cnt:   Specify the count of steps
                total: Total counts of steps
                msg:   Output the message of the step progress bar
    return code:
                None
    """
    filling = []
    for i in range(len(os.getcwd()) + 30):
        filling.append("=")
    print(''.join(filling))
    print('[Step %d/%d] %s' %(cnt, total, msg))
    print(''.join(filling))
    
def schedule(i, start, end):
    """
    description:
                Output time progress bar
    parameters:
                i:     Specific step time
                start: Specify the step start time
                end:   Specify the step end time
    return code:
                None
    """
    print('******************************')
    print('Step %d Running time: %.7s s'%(i, end-start))
    print('******************************')

def clean_ups(cleanlist):
    """
    description:
                Clean up working directory
    parameters:
                cleanlist: List of files to be cleaned
    return code:
                None
    """
    for i in range(len(cleanlist)):
        if not os.path.exists("debug"):
            os.mkdir("debug")
        output = "{}{}".format("debug/", cleanlist[i])
        if "data_for_fpga" in cleanlist[i]:
            output = "{}{}".format("output/", cleanlist[i])
        if os.path.exists(output):
            if os.path.isdir(output):
                shutil.rmtree(output)
            else:
                os.remove(output)
            print("%s clean success" % (output))

    print("clean_ups successfully")

def checkfile(filelist):
    """
    description:
                Check files in working directory
    parameters:
                cleanlist: List of files to check
    return code:
                None
    """
    for i in range(len(filelist)):
        file = "input/"
        if "imagenet_fixed_v4_1" in filelist[i]:
            file = ""
        name = "{}{}".format(file, filelist[i])
        if not os.path.exists(name):
            print("%s not found...\nPlease try again!" % name)
            sys.exit(1)

    print("checkfile successfully")


if __name__ == '__main__':
    """
    description:
                main function
    parameters:
                None
    return code: 
                None
    """
    cleanlist = ["makenet.py", "vggnet.py", "vggnet.log", "layerinfo", "output", "data_for_fpga"]
    fileslist = ["imagenet_fixed_v4_1", "vgg_imagenet.py", "vgg_imagenet2.pt", "quant_layer.py"]

    if (len(sys.argv) != 2) or not os.path.exists(sys.argv[1]):
        pt_path = "input/vgg_imagenet2.pt"
    else:
        pt_path = sys.argv[1]

    start = time.time()
    allstart = start
    output(1, 6, "Check the necessary files...")
    checkfile(fileslist)
    end = time.time()
    schedule(1, start, end)

    start = time.time()
    output(2, 6, 'Clean the necessary files...')
    clean_ups(cleanlist)
    end = time.time()
    schedule(2, start, end)

    start = time.time()
    output(3, 6, 'Make network model file...')
    mknet = makenet()
    gen_net(mknet, "./input/vgg_imagenet.py")
    end = time.time()
    schedule(3, start, end)

    start = time.time()
    output(4, 6, 'Run the network model file...')
    os.system("python3 debug/vggnet.py > debug/vggnet.log")
    print("run vggnet.py successfully")
    end = time.time()
    schedule(4, start, end)

    start = time.time()
    os.system("mkdir -p debug/output output")
    output(5, 6, 'Load pt file and format output...')
    loadpt = load_pt()
    gen_txt(loadpt)
    print("pt data and config file successfully")
    end = time.time()
    schedule(5, start, end)

    start = time.time()
    output(6, 6, 'Make the bin file needed by fpga...')
    gen_fpga("imagenet_fixed_v4_1")
    print("generate fpga data successfully")
    end = time.time()
    schedule(6, start, end)
    allend = end

    print('Make data successfully! It costs %.2f Seconds'%(allend - allstart))
