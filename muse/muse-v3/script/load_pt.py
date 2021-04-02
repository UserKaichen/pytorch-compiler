import io
import os
import math
import torch
import threading
import struct as st
from script.run_steps import prints

class load_pt():
    def __init__(self, fmain, conf, ptdt, net, pt, log):
        self.in_q = "5"
        self.fw = fmain
        self.padding = ""
        self.layermsg = ""
        self.ptpath = pt
        self.netpath = net
        self.logpath = log
        self.confpath = conf
        self.ptdtpath = ptdt
        self.in_feature_h = 0
        self.in_feature_w = 0
        self.layers = [["", ""]]
        self.in_channels_bf = ""
        self.out_channels_bf = ""
        self.layer_cnts = self.get_cnts(log)

        if "resnet" not in self.netpath.lower():
                os.system(f'mkdir -p {self.confpath}  {self.ptdtpath}')

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
            return 1

        next_flg = f'layer_num:{int(str(find_flg).split(":", 1)[1])+1}'
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
        path = f'{self.ptdtpath}/{filename}.txt'
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
                        hexdata = '%X' % st.unpack('H', st.pack('e', conver[i]))[0]
                        fw.write(str(hexdata))
                    elif "weight" in filename:
                        scalelist = scale.tolist()
                        weight = round(conver[i] / scalelist)
                        fw.write(str(weight))
                    else:
                        fw.write(str(conver[i]))
                    fw.write('\n')

        self.fw.write(f'{path} write data success\n')

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
        if type == "pool" and int(out_feature_h) == 7 and len(self.padding):
            padding_info = self.padding.split(" ", 6)
            self.in_feature_h = str(int(padding_info[4].split(":")[1]))
            self.in_feature_w = str(int(padding_info[5].split(":")[1]))
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
        stride_x      = stride_y      = ""
        in_channels   = out_channels  = ""
        out_feature_w = out_feature_h = ""
        kernel_size_x = kernel_size_y = ""

        for i in range(len(param)):
            if "in_channels" in param[i] or "in_features_y" in param[i]:
                in_channels = f'input channel num  = {param[i].split(":")[1].strip()}'
                if in_channels.strip() == self.out_channels_bf or len(self.out_channels_bf) == 0:
                    config.append(in_channels)
                else:
                    config.append(f'input channel num  = {self.out_channels_bf.split("=")[1].strip()}')
            elif "out_channels" in param[i] or "out_features_y" in param[i]:
                out_channels = f'output channel num = {param[i].split(":")[1].strip()}'
                config.append(out_channels)
            elif "feature_map_size_x" in param[i]:
                out_feature_h = param[i].split(":")[1].strip()
            elif "feature_map_size_y" in param[i]:
                out_feature_w = param[i].split(":")[1].strip()
            elif "stride_x" in param[i]:
                stride_x = f'stride_x = {param[i].split(":")[1].strip()}'
            elif "stride_y" in param[i]:
                stride_x = f'stride_y = {param[i].split(":")[1].strip()}'
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
            if (len(kernel_size_x)):
                ratio = int(kernel_size_x)
                out_feature_h = str(int(int(float(self.in_feature_h)) / ratio))
                out_feature_w = str(int(int(float(self.in_feature_w)) / ratio))
        config.append(f'input feature_h = {self.in_feature_h}')
        config.append(f'input feature_w = {self.in_feature_w}')
        if layer_type != "pool":
            if (layer_type == "fc"):
                if self.in_feature_h == "1" and self.in_feature_w == "1":
                    kernel_size_x = out_feature_h = self.in_feature_h
                    kernel_size_y = out_feature_w = self.in_feature_w
        config.append(f'output feature_h = {out_feature_h}')
        config.append(f'output feature_w = {out_feature_w}')
        config.append("\n")
        if layer_type == "conv":
            config.append("padding = 1")
        else:
            config.append("padding = 0")
        config.append(in_q)
        config.append(out_q)
        config.append(f'kernel size = {kernel_size_x}×{kernel_size_y}')
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
        param     = config = []
        in_q      = out_q  = ""
        layer_num = relu   = ""

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
            f'act0 file               : layers.{str(int(layer_num) - 1)}.conv.input.6.txt')
        config.append(
            f'output file             : layers.{str(int(layer_num) - 1)}.quant.output.6.txt')
        config.append(
            f'weight file             : layers.{str(int(layer_num) - 1)}.conv.weight.txt')
        config.append(
            f'bn k file               : layers.{str(int(layer_num) - 1)}.bn.bn_k.txt')
        config.append(
            f'bn b file               : layers.{str(int(layer_num) - 1)}.bn.bn_b.txt')

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
        param     = config  = []
        in_q      = out_q   = ""
        poolname  = "Maxpooling"

        for i in range(len(self.layermsg)):
            if "padding param" in self.layermsg[i]:
                self.padding = self.layermsg[i]
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
        param   = config = []
        in_q    = out_q  = ""
        fc_name = relu   = ""

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
                    classifier.append(f'{quant_list[i].rsplit(".")[0]}.{quant_list[i].rsplit(".")[1]}')

        config.append(f'act0 file               : {classifier[int(fc_name[2]) - 1]}.input.6.txt')
        config.append(f'output file             : quant_{fc_name}.output.6.txt')
        config.append(f'weight file             : {classifier[int(fc_name[2]) - 1]}.weight.txt')
        config.append(f'bn k file               : {classifier[int(fc_name[2]) - 1]}.k.txt')
        config.append(f'bn b file               : {classifier[int(fc_name[2]) - 1]}.bias.txt')

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
        padding = ""
        if layername != "1":
            padding = f'_{str(int(layername) - 1)}'
        path = f'{self.confpath}/config{padding}.txt'

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
            elif "AdaptAvgpool" in layer_type:
                self.write_pool_config(fw)
            else:
                print("Unknown layer type...", layer_type)
                return 1

        self.fw.write(f'{path} write config success\n')

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

        self.layermsg.append(f'in_q = {self.in_q}')
        self.layermsg.append(f'out_q = {out_q}')
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
                    line_split = line.split("(")[1].split(")")[0].split(",", 4)
                    self.in_feature_h = line_split[2].strip()
                    self.in_feature_w = line_split[3].strip()
                    break

    def get_cnts(self, log):
        """
        description: Get counts of layers
        parameter: NULL
        return value: NULL
        """
        layer_cnts = 0

        with open(log, 'r') as f:
            for lines in f.readlines():
                if "layer_num:" in lines:
                    layer_cnts += 1

        return layer_cnts

    def gen_txt(self):
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

        self.get_tensorinfo(self.netpath)
        with open(self.ptpath, 'rb') as f:
            ptdata = torch.load(self.ptpath, map_location=torch.device('cpu'))
            if "resnet34" in self.netpath.lower():
                act_bit = ptdata['hyper_parameters']['act_quant_bit']
                wet_bit = ptdata['hyper_parameters']['weight_quant_bit']
            else:
                act_bit = 16
                wet_bit = 8
            for k, v in ptdata['state_dict'].items():
                k = k.split(".", 1)[1].strip()
                name_list.append(k)
                data_list.append(v)
                if "resnet" not in self.netpath.lower():
                    if "quant_" in k or "classifier." in k:
                        quant_list.append(k)
                        quant_list.append(v)

        if "resnet" in self.netpath.lower():
            return name_list, data_list, act_bit, wet_bit
        else:
            with open(f'{self.ptdtpath}/img.input.q.txt', 'w') as fq:
                fq.write('{}{}'.format(self.in_q, '\n'))
            self.fw.write(f'{self.ptdtpath}/img.input.q.txt write success\n')

        for i in range(self.layer_cnts):
            layer = f'layers.{i}.'
            for j in range(len(name_list)):
                if layer in name_list[j]:
                    self.layers.append([name_list[j], data_list[j]])
                    counts += 1
            onelayer_cnt.append(str(counts))
            counts = 0

        del (self.layers[0])
        for i in range(self.layer_cnts):
            layername = f'layer_num:{str(i + 1)}'
            self.layermsg = self.get_layer_info(self.logpath, layername)
            self.splicing_output(int(onelayer_cnt[i]), counts, quant_list)
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
                    write_data = threading.Thread(target=self.write_pt_data,
                                                  args=(fcname, weight, scale))
                    write_data.start()
                    write_data.join()
                    continue
            elif "quant_" in tmpstr or "classifier" in tmpstr:
                write_data = threading.Thread(target=self.write_pt_data,
                                              args=(quant_list[i], quant_list[i + 1], scale))
                write_data.start()
                write_data.join()

        prints("run gen_txt successfully")
        return name_list, data_list, act_bit, wet_bit
