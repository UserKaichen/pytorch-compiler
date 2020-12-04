import os

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
            self.fvggnet.write(
                f'        self.pool{str(i+1)} = nn.{self.pools[i].split(":")[1].strip()}\n')
        for j in range(len(self.convs)):
            self.fvggnet.write(
                f'        self.conv{str(j+1)} = nn.{self.convs[j].split(":")[1].strip()}\n')
            if self.bns[0] == "True":
                self.fvggnet.write(
                    f'        self.bn{str(j+1)} = nn.{self.bns[j+1].split(":")[1].strip()}\n')
        self.fvggnet.write("\n")

        for i in range(len(self.avginit)):
            self.fvggnet.write(f'        {self.avginit[i]}\n')
        for i in range(len(self.qulist)):
            if "= QuantLayer()" in self.qulist[i]:
                self.fvggnet.write(f'        {self.qulist[i]}\n')
        for i in range(len(self.fcinit)):
            self.fvggnet.write(f'        {self.fcinit[i]}\n')

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
        convcmd = f'        x = {x}{str(convcnt)}(self.conv{str(convcnt)}(x{end}'
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
                poolcmd = f'        x = self.pool{str(poolcnt)}(x)\n'
                self.fvggnet.write(poolcmd)

    """
    description: Add "x = self.avgpool_*(x)" code to vggnet.py
    parameter: NULL
    return value: NULL
    """
    def _make_avgpool(self):
        self.fvggnet.write(f'        x = self.padding(x)\n')
        for i in range(len(self.avgford)):
            self.fvggnet.write(f'        {self.avgford[i]}\n')
            self.fvggnet.write(f'        {self.qulist[i+int(len(self.qulist)/2)]}\n')
        for i in range(len(self.fcford)):
            self.fvggnet.write(f'        {self.fcford[i]}\n')

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

if __name__ == '__main__':
    """
    description:
                main function
    parameters:
                None
    return code: 
                None
    """
    os.chdir("..")
    mknet = makenet()
    gen_net(mknet, "./input/vgg_imagenet.py")
    os.chdir("script")
