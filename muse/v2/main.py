import io
import os
import sys
import math
import torch
import shutil
import threading
import subprocess
import torch.nn as nn
from quant_layer import QuantLayer

in_q  = 7
layer_cnts = 0
in_feature_h = 0
in_feature_w = 0
padding_param = ""
in_channels_bf = 0
out_channels_bf = 0

class makenet():
    def __init__(self, filename):
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
        self.fvggnet = open("vggnet.py", "a")
        self.fmakenet = open("makenet.py", "a")

        self.make_config(filename)
        self.make_block(filename)
        self.make_class(filename)
        self.make_layers(filename)
        self.make_padding(filename)
        self.make_forward(filename)
        self.make_weight(filename)
        self.make_main(filename)
        self.fmakenet.close()

        os.system("python3 makenet.py > layerinfo")
        self.splicing_layers("layerinfo")
        self.bns[0] = self.get_op_code(filename, "bn")
        self.get_op_code(filename, "avgpool")
        self.get_op_code(filename, "weight")
        self.get_op_code(filename, "quant")
        self.get_op_code(filename, "fc")

        self._make_head()
        self._make_init()
        self._make_padding(filename)
        self._make_forward()
        self._make_avgpool()
        self._make_tail()
        self.fvggnet.close()
        print("makenet success")

    def make_config(self, filename):
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.startswith("#") is False:
                    self.fmakenet.write(line)
                if line.strip() == "}":
                    break
        self.fmakenet.write('\n')

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

    def make_weight(self, filename):
        read_flag = 0
        with open(filename, 'r') as file:
            while True:
                line = file.readline()
                if line.strip().startswith("#") is False:
                    if "def _initialize_weights" not in line and read_flag == 0:
                        continue
                    read_flag = 1
                    # if "= vgg(" and "_initialize_weights" not in line:
                    if "= vgg(" in line:
                        break
                    self.fmakenet.write(line)

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
                    print_vgg = "{}{}{}".format("print(\"vgg_module = \", ", line.strip().split(" ", 1)[0], ")\n")
                    self.fmakenet.write(print_vgg)
                    return

    def _make_head(self):
        self.fvggnet.write("import torch\n")
        self.fvggnet.write("import torch.nn as nn\n")
        self.fvggnet.write("import torch.nn.functional as F\n")
        self.fvggnet.write("from quant_layer import QuantLayer\n\n")
        self.fvggnet.write("class Net(nn.Module):\n")
        self.fvggnet.write("    def __init__(self, num_classes=10):\n")
        self.fvggnet.write("        super(Net, self).__init__()\n")

    def _make_tail(self):
        self.fvggnet.write("\nn = Net()\n")
        self.fvggnet.write("example_input = torch.rand(1, 3, 224, 224)\n")
        self.fvggnet.write("module = torch.jit.trace(n, example_input)\n")
        self.fvggnet.write("module._c._fun_compile()\n")

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

    def _make_convlay(self, convcnt):
        x = "F.relu"
        end = "))\n"

        if self.bns[0] == "True":
            x = "F.relu(self.bn"
            end = ")))\n"
        convcmd = "{}{}{}{}{}{}{}".format("        x = ", x, str(convcnt),
                                          "(self.conv",str(convcnt), "(x", end)
        self.fvggnet.write(convcmd)

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

    def _make_avgpool(self):
        self.fvggnet.write("{}{}".format("        x = self.padding(x)", "\n"))
        for i in range(len(self.avgford)):
            self.fvggnet.write("{}{}{}".format("        ", self.avgford[i], "\n"))
            self.fvggnet.write("{}{}{}".format("        ", self.qulist[i+int(len(self.qulist)/2)], "\n"))
        for i in range(len(self.fcford)):
            self.fvggnet.write("{}{}{}".format("        ", self.fcford[i], "\n"))

        self.fvggnet.write("        return x\n")

    def splicing_layers(self, file):
        with open(file, "r") as f:
            lines = f.readline()
            while lines:
                lines = f.readline()
                self.layer.append(lines)

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

def get_layer_info(path, flag):
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

def getoneDimList(newlist):
    oneDimList = []
    for element in newlist:
        if not isinstance(element, list):
            oneDimList.append(element)
        else:
            oneDimList.extend(getoneDimList(element))
    return oneDimList

def write_pt_data(filename, filedata):
    path = "{}{}{}".format("./output/", filename, ".txt")
    with open(path, 'w') as fw:
        if "quant_" in filename:
            fw.write(str(filedata))
        else:
            conver = filedata.tolist()
            if filedata.dim() == 0:
                fw.write(str(conver))
                return
            elif filedata.dim() > 1:
                conver = getoneDimList(conver)
            for i in range(len(conver)):
                if "bn.bn" in filename:
                    fw.write(str(conver[i]))
                else:
                    fw.write(str(conver[i]))
                fw.write('\n')

    print("%s write data success" % path)

def write_to_file(fw, config):
    for i in range(len(config)):
        if "\n" in config[i]:
            fw.write("\n")
            continue
        fw.write(config[i])
        fw.write("\n")

def out_to_in(out_feature_h, out_feature_w, in_channels, out_channels, type):
    global in_feature_h, in_feature_w
    global in_channels_bf, out_channels_bf

    if type == "pool" and int(out_feature_h) == 7 and "padding param: " in padding_param:
        in_feature_h = str(int(padding_param.split(" ", 6)[4].split(":")[1]))
        in_feature_w = str(int(padding_param.split(" ", 6)[5].split(":")[1]))
    else:
        in_feature_h = out_feature_h
        in_feature_w = out_feature_w
    if type != "pool":
        in_channels_bf = in_channels
        out_channels_bf = out_channels

def get_all_params(config, param):
    stride_x      = stride_y      = ""
    in_channels   = out_channels  = ""
    out_feature_w = out_feature_h = ""
    kernel_size_x = kernel_size_y = ""

    for i in range(len(param)):
        if "in_channels" in param[i] or "in_features_y" in param[i]:
            in_channels = "{}{}".format("input channel num  = ", param[i].split(":")[1].strip())
            config.append(in_channels)
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

def write_common_params(config, param, in_q, out_q, info, layer_type):
    global in_feature_h, in_feature_w

    config, out_feature_w, out_feature_h, kernel_size_x, kernel_size_y, in_channels, \
            out_channels, stride_x, stride_y = get_all_params(config, param)

    if layer_type == "pool":
        config.append(in_channels_bf)
        config.append(out_channels_bf)
        ratio = int(kernel_size_x)
        out_feature_h = str(int(int(float(in_feature_h)) / ratio))
        out_feature_w = str(int(int(float(in_feature_w)) / ratio))
    config.append("{}{}".format("input feature_h = ", in_feature_h))
    config.append("{}{}".format("input feature_w = ", in_feature_w))
    if layer_type != "pool":
        if (layer_type == "fc"):
            if in_feature_h == 1 and in_feature_w == 1:
                kernel_size_x = out_feature_h = in_feature_h
                kernel_size_y = out_feature_w = in_feature_w
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
    if "relu " in info:
        config.append(info)
    else:
        config.append("relu 0")
    config.append("w_quan   \"MIX\"")
    config.append("\n")

    if layer_type != "pool":
        config.append("【Pattern Pruning config】")
        config.append("pattern_dic.txt存的是16种pattern所对应的mask，顺序是先横着走再换行；pattern_idx所对应的是每个kernel所采用的pattern编号，顺序与weight顺序一致；weight.txt是不做压缩的，仅作参考；weight_nonzero是最终给到芯片的权重，已经做了4pattern压缩存储。")
        config.append("\n")
        config.append("pattern: 9-pattern")
        config.append("\n")
        config.append("BN中，k=0.01 (0x211F)， b=0")
        config.append("\n")

    return out_feature_w, out_feature_h, in_channels, out_channels

def write_conv_config(fw, layermsg):
    layer_num  = ""
    param  = config = []
    in_q = out_q = relu = ""

    for i in range(len(layermsg)):
        if layermsg[i].startswith("layer_num:") is True:
            layer_num = layermsg[i].split(" ")[0].strip().split(":")[1]
        elif "in_channels" in layermsg[i]:
            param = layermsg[i].split(" ")
        elif "relu param" in layermsg[i]:
            relu = "relu √"
        elif "in_q =" in layermsg[i]:
            in_q = layermsg[i]
        elif "out_q =" in layermsg[i]:
            out_q = layermsg[i]

    out_feature_w, out_feature_h, in_channels, out_channels = \
    write_common_params(config, param, in_q, out_q, relu, "conv")

    config.append("{}{}{}".format("act0 file               : layers.", str(int(layer_num)-1), ".conv.input.6.txt"))
    config.append("{}{}{}".format("output file             : layers.", str(int(layer_num)-1), ".quant.output.6.txt"))
    config.append("{}{}{}".format("weight file             : layers.", str(int(layer_num)-1), ".conv.weight.txt"))
    config.append("{}{}{}".format("bn k file               : layers.", str(int(layer_num)-1), ".bn.bn_k.txt"))
    config.append("{}{}{}".format("bn b file               : layers.", str(int(layer_num)-1), ".bn.bn_b.txt"))

    write_to_file(fw, config)
    out_to_in(out_feature_h, out_feature_w , in_channels, out_channels, "conv")

def write_pool_config(fw, layermsg):
    in_q = out_q = ""
    param = config = []
    poolname = "Maxpooling"

    for i in range(len(layermsg)):
        if "padding param:" in layermsg[i]:
            global padding_param
            padding_param = layermsg[i]
        elif "param:" in layermsg[i]:
            param = layermsg[i].split(" ")
        if "avgpool" in layermsg[i]:
            poolname = "Average pooling"
        elif "in_q =" in layermsg[i]:
            in_q = layermsg[i]
        elif "out_q =" in layermsg[i]:
            out_q = layermsg[i]

    out_feature_w, out_feature_h, in_channels, out_channels = \
    write_common_params(config, param, in_q, out_q, "", "pool")

    write_to_file(fw, config)
    fw.write(poolname)

    out_to_in(out_feature_h, out_feature_w, in_channels, out_channels, "pool")

def write_fc_config(fw, layermsg, quant_list):
    fc_name = ""
    param  = config = []
    in_q = out_q = relu = ""

    for i in range(len(layermsg)):
        if " param:" in layermsg[i]:
            fc_name = layermsg[i].split(" ")[0].strip()
            param = layermsg[i].split(" ")
        elif "in_q =" in layermsg[i]:
            in_q = layermsg[i]
        elif "out_q =" in layermsg[i]:
            out_q = layermsg[i]
        elif "ReLU" in layermsg[i]:
            relu = "relu √"

    out_feature_w, out_feature_h, in_channels, out_channels = \
    write_common_params(config, param, in_q, out_q, relu, "fc")

    classifier = []
    for i in range(len(quant_list)):
        if (i % 2) == 0:
            if "classifier" in quant_list[i] and "weight" in quant_list[i]:
                classifier.append("{}{}{}".format(quant_list[i].rsplit(".")[0], ".",quant_list[i].rsplit(".")[1]))

    config.append("{}{}{}".format("act0 file               : ", classifier[int(fc_name[2])-1], ".input.6.txt"))
    config.append("{}{}{}{}".format("output file             : ", "quant_", fc_name, ".output.6.txt"))
    config.append("{}{}{}".format("weight file             : ", classifier[int(fc_name[2])-1], ".weight.txt"))
    config.append("{}{}{}".format("bn k file               : ", classifier[int(fc_name[2])-1], ".k.txt"))
    config.append("{}{}{}".format("bn b file               : ", classifier[int(fc_name[2])-1], ".bias.txt"))

    write_to_file(fw, config)
    out_to_in(out_feature_h, out_feature_w,in_channels, out_channels, "fc")

def write_layer_config(layermsg, quant_list):
    layername = layermsg[0].split(":", 1)[1].split(" ", 1)[0].strip('\n')
    if layername == "1":
        path = "{}{}".format("./output/config", ".txt")
    else:
        path = "{}{}{}".format("./output/config_", str(int(layername)-1), ".txt")

    with open(path, 'w') as fw:
        layer_type = " "
        for i in range(len(layermsg)):
            if "layer type:" in layermsg[i]:
                if layermsg[i].startswith("layer_num:1 ") is True:
                    layer_type = layermsg[i].split(":", 3)[2]
                else:
                    layer_type = layermsg[i].split(":", 5)[2].split(" ")[0]
        if "conv" in layer_type:
            write_conv_config(fw, layermsg)
        elif "pool" in layer_type:
            write_pool_config(fw, layermsg) 
        elif "fc" in layer_type:
            write_fc_config(fw, layermsg, quant_list)
        else:
            print("Unknown layer type...")
            return

    print("%s write config success" % path)

def deal_out_in_q(quant_list, data, layer_msg):
    bits = 7
    global in_q
    if "pool" in layer_msg:
        out_q = in_q
    else:
        out_q = bits - math.ceil(math.log2(0.5*data))

    layer_msg.append("{}{}".format("in_q = ", str(in_q)))
    layer_msg.append("{}{}".format("out_q = ", str(out_q)))
    in_q = out_q
    write_config = threading.Thread(target=write_layer_config, args=(layer_msg, quant_list))
    write_config.start()
    write_config.join()

def splicing_output(num, flag, layerlist, layer_msg, quant_list):
    for i in range(len(layerlist)):
        if "quant.alpha" in layerlist[i][0]:
            data = layerlist[i][1]
            if num == 0:
                deal_out_in_q(quant_list, data, layer_msg)
                return

    for i in range(num):
        name = layerlist[flag+i][0]
        data = layerlist[flag+i][1]
        if "bn.running_mean" in name or "bn.running_var" in name \
              or "bn.weight" in name or "bn.bias" in name \
              or "num_batches_tracked" in name:
            continue
        elif "quant.alpha" in name:
            deal_out_in_q(name, data, layer_msg)
            continue
        write_data = threading.Thread(target=write_pt_data, args=(name, data))
        write_data.start()
        write_data.join()

def get_layercount(filename):
    with open(filename, 'r') as file:
        while True:
            line = file.readline()
            if line.strip() == "":
                break 
            elif ": BasicBlock(" in line or "Pool2d(" in line or "Linear" in line:
                global layer_cnts
                layer_cnts += 1

def get_tensorinfo(filename):
    with open(filename, 'r') as file:
        while True:
            line = file.readline()
            if line.strip() == "":
                continue
            elif  "torch.rand" in line:
                global in_feature_h
                global in_feature_w
                in_feature_h = line.split("(")[1].split(")")[0].split(",", 4)[2].strip()
                in_feature_w = line.split("(")[1].split(")")[0].split(",", 4)[3].strip()
                break

def load_pt(pt_path):
    name_list    = []
    data_list    = []
    quant_list   = []
    onelayer_cnt = []

    get_tensorinfo("vggnet.py")
    get_layercount("layerinfo")
    with open(pt_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        dict = torch.load(buffer, map_location=torch.device('cpu'))
        for k, v in dict.items():
           if "quant_" in k or "classifier." in k:
               quant_list.append(k)
               quant_list.append(v)
           name_list.append(k)
           data_list.append(v)
           tensor = v

    counts = 0
    layers = [["", tensor]]

    for i in range(layer_cnts):
        layer = "{}{}{}".format("layers.", i, ".")
        for j in range(len(name_list)):
            if layer in name_list[j]:
                 layers.append([name_list[j], data_list[j]])
                 counts += 1
        onelayer_cnt.append(str(counts))
        counts = 0

    del(layers[0])
    logpath = "{}{}".format(os.getcwd(), "/vggnet.log")
    for i in range(layer_cnts):
        layername = "{}{}".format("layer_num:", str(i+1))
        layer_msg = get_layer_info(logpath, layername)
        splicing_output(int(onelayer_cnt[i]), counts, layers, layer_msg, quant_list)
        counts += int(onelayer_cnt[i])
        
    for i in range(int(len(quant_list))):
        tmpstr = str(quant_list[i])
        if ".alpha" in tmpstr:
            continue
        if "quant_" in tmpstr or "classifier" in tmpstr:
            write_data = threading.Thread(target=write_pt_data, 
                         args=(quant_list[i], quant_list[i+1]))
            write_data.start()
            write_data.join()

def checkfile(filelist):
    for i in range(len(filelist)):
        if not os.path.exists(filelist[i]):
            print("%s not find in %s directory..." %(filelist[i], os.getcwd()))

def cleanup(cleanlist):
    for i in range(len(cleanlist)):
        if os.path.exists(cleanlist[i]):
            if os.path.isdir(cleanlist[i]):
                shutil.rmtree(cleanlist[i])
            else:
                os.remove(cleanlist[i])
            print("%s in %s directory clean success" %(cleanlist[i], os.getcwd()))
        

if __name__ == '__main__':
    fileslist = ["vgg_imagenet.py", "vgg_imagenet2.pt", "quant_layer.py"]
    cleanlist = ["output", "makenet.py", "vggnet.py", "vggnet.log", "layerinfo"]

    if (len(sys.argv) != 2) or not os.path.exists(sys.argv[1]):
        pt_path = "./vgg_imagenet2.pt"
    else:
        pt_path = sys.argv[1]

    cleanup(cleanlist)
    checkfile(fileslist)
    makenet("vgg_imagenet.py")
    os.system("python3 vggnet.py > vggnet.log")
    os.system("mkdir output")
    load_pt(pt_path)
