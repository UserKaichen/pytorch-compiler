import os
import re
import sys
import math
import threading
from input.config_gen_file import *
from script.run_steps import adwrap
from script.calcu_addr import get_fstin_CHW, get_out_CHW

fmain = 0
layer_cnts = 0

def send_inoutbinvar(fw, cnt):
    global fmain
    global layer_cnts

    fmain = fw
    layer_cnts = cnt

def write_inout_bin(txtfile, binfile, H, W, C, in_q):
    gen_act_file(txtfile, "", binfile, H, W, C, in_q, 1, 0, 0, 1)
    fmain.write(f'{binfile} generate success\n')

def gen_inout_bin_vgg(bindir, bmpdt, net, pool_num, downsample, name_list, data_list):
    fc_output = []
    fc_inputs = []
    alphalist = []
    conv_inputs = []
    conv_output = []

    list = os.listdir(bmpdt)
    for i in range(len(list)):
        if "input" in list[i]:
            if "conv" in list[i]:
                conv_inputs.append(list[i])
            elif "classifier" in list[i]:
                fc_inputs.append(list[i])
        elif "output" in list[i]:
            if "quant." in list[i]:
                conv_output.append(list[i])
            elif "quant_fc" in list[i]:
                fc_output.append(list[i])

    fc_inputs = sorted(fc_inputs)
    fc_output = sorted(fc_output)
    conv_inputs.sort(key=lambda x: int(re.match('\D+(\d+)\.conv.input.6.txt', x).group(1)))
    conv_output.sort(key=lambda x: int(re.match('\D+(\d+)\.quant.output.6.txt', x).group(1)))

    for i in range(len(name_list)):
        name = name_list[i]
        if ".alpha" in name:
            alphalist.append(data_list[i].tolist())

    j = 0
    bit = 7
    in_q = 5

    for i in range(1, layer_cnts - len(fc_inputs) + 1):
        if i == 1:
            C_in, H_in, W_in = get_fstin_CHW(net)
        elif i in downsample:
            C_in, H_in, W_in = get_out_CHW(i-2)
        else:
            C_in, H_in, W_in = get_out_CHW(i-1)
        C_out, H_out, W_out = get_out_CHW(i)
        if i in pool_num:
            out_q = in_q
            continue
        out_q = bit - math.ceil(math.log2(0.5 * alphalist[j]))
        input_txt = f'{bmpdt}/{conv_inputs[j]}'
        output_txt = f'{bmpdt}/{conv_output[j]}'
        input_bin = f'{bindir}/layer{i}_input_act0_file.bin'
        output_bin = f'{bindir}/layer{i}_output_file.bin'
        write_in = threading.Thread(target=write_inout_bin,
                                    args=(input_txt, input_bin, H_in, W_in, C_in, in_q))
        write_in.start()
        write_in.join()
        write_out = threading.Thread(target=write_inout_bin,
                                     args=(output_txt, output_bin, H_out, W_out, C_out, out_q))
        write_out.start()
        write_out.join()
        j += 1
        in_q = out_q

    if layer_cnts <= 1:
    # Adapt one layer toolchains
        return

    for i in range(len(fc_inputs)):
        layernum = i + len(conv_inputs) + len(pool_num) + 1
        C_in, H_in, W_in = get_out_CHW(layernum-1)
        C_out, H_out, W_out = get_out_CHW(layernum)
        out_q = bit - math.ceil(math.log2(0.5 * alphalist[j + len(fc_inputs)]))
        input_txt = f'{bmpdt}/{fc_inputs[i]}'
        output_txt = f'{bmpdt}/{fc_output[i]}'
        input_bin = f'{bindir}/layer{len(conv_inputs) + i + len(pool_num)}_input_act0_file.bin'
        output_bin = f'{bindir}/layer{len(conv_inputs) + i + len(pool_num)}_output_file.bin'
        write_in = threading.Thread(target=write_inout_bin,
                                    args=(input_txt, input_bin, H_in, W_in, C_in, in_q))
        write_in.start()
        write_in.join()
        write_out = threading.Thread(target=write_inout_bin,
                                     args=(output_txt, output_bin, H_out, W_out, C_out, in_q))
        write_out.start()
        write_out.join()
        in_q = out_q

def find_layer_quant(type, netpath, netname):
    i = 1
    get_max = ""
    findobj = ""

    if "quant" in type:
        if "resnet18" == netname or "resnet34" == netname:
            findobj = "class BasicBlock(_BasicBlock):"
        elif "resnet50" == netname:
            findobj = "class Bottleneck(_Bottleneck):"
    elif "layer" in type:
        findobj = "class ResNet(_ResNet):"
    elif "block" in type:
        findobj = f'def {netname}(pretrained=False, progress=True, **kwargs):'

    with open(netpath, 'r') as fd:
        for line in fd:
            if line.startswith(findobj):
                for line in fd:
                    line = line.lstrip()
                    if "quant" in type:
                        findstr = f'self.quant{i} = QuantLayer()'
                    elif "layer" in type:
                        findstr = f'x = self.layer{i}(x)'
                    elif "block" in type:
                        findstr = f"return _resnet('{netname}'"
                    if line.startswith(findstr):
                        if "block" in type:
                            return line.split('[',1)[1].split(']',1)[0]
                        get_max = line
                        i += 1
                    if "class" in line:
                        break

    return re.findall(r"\d+\.?\d*", get_max)

def getblock_max(cnt, name):
    if cnt == 0:
        return 0

    model = ""
    block_max = 0
    blockname = "BasicBlock"
    sys.path.append("input")

    if "resnet18" in name:
        from ResNet import resnet18
        model = resnet18()
    elif "resnet34" in name:
        from ResNet import resnet34
        model = resnet34()
    elif "resnet50" in name:
        from ResNet import resnet50
        model = resnet50()
        blockname = "Bottleneck"
    modeltxt = f'.debug/{name}.txt'
    with open(modeltxt, 'w') as fmd:
        fmd.write(str(model))
    with open(modeltxt, 'r') as frd:
        for line in frd:
            line = line.strip()
            if line.startswith(f'(layer{cnt}): Sequential('):
                for line in frd:
                    line = line.strip()
                    if f"): {blockname}(" in line:
                        block_max = line.split(')')[0].split('(')[1]
                    elif line.startswith(f'(layer{cnt+1}): Sequential(') or "quant_fc" in line:
                        return int(block_max)+1

def getresnet_max(path, name):
    layermax = find_layer_quant("layer", path, name)
    quantmax = find_layer_quant("quant", path, name)
    blockmax = find_layer_quant("block", path, name).split(',')
    return int(layermax[0]), int(quantmax[0]), blockmax

def gen_inout_bin_res(bindir, bmpdt, net, pool_num, downsample, name_list, data_list):
    file_list = []
    alphalist = []
    intxt_list = []
    outtxt_list = []
    from script.run_steps import get_model_name
    name = get_model_name(net)
    layer_max, opear_max, block_max = getresnet_max(net, name)

    def file_exist(name):
        if os.path.exists(f'{bmpdt}/{name}'):
            file_list.append(name)

    for i in range(opear_max):
        inname = f'conv{i}.input.6.txt'
        outname = f'quant{i}.output.6.txt'
        file_exist(inname)
        file_exist(outname)

    for layerindex in range(1, layer_max+1):
        for blockindex in range(int(block_max[layerindex-1].strip())):
            for opearindex in range(1, opear_max+1):
                inname = f'layer{layerindex}.{blockindex}.conv{opearindex}.input.6.txt'
                outname = f'layer{layerindex}.{blockindex}.quant{opearindex}.output.6.txt'
                file_exist(inname)
                file_exist(outname)
            if layerindex and blockindex == 0:
                inname = f'layer{layerindex}.0.downsample.0.input.6.txt'
                outname = f'layer{layerindex}.0.quant_shortcut.output.6.txt'
                file_exist(inname)
                file_exist(outname)

    avgout = "quant_avg.output.6.txt"
    fcin = "fc.input.6.txt"
    fcout = "quant_fc.output.6.txt"
    if os.path.exists(f'{bmpdt}/{avgout}'):
        file_list.append(f'{avgout}')
    if os.path.exists(f'{bmpdt}/{fcin}'):
        file_list.append(f'{fcin}')
    if os.path.exists(f'{bmpdt}/{fcout}'):
        file_list.append(f'{fcout}')

    if not os.path.exists(bindir):
        print(f'run os.mkdir({bindir} error)')
        return -1

    for i in range(len(name_list)):
        name = name_list[i]
        if ".alpha" in name:
            alphalist.append(data_list[i].tolist())

    for i in range(len(file_list)):
        if avgout == file_list[i]:
            avgin = file_list[i-1]
        if "quant_avg" in file_list[i]:
            break
        if "input" in file_list[i]:
            intxt_list.append(file_list[i])
            if "output" in file_list[i+1]:
                if "output" in file_list[i+2]:
                    outname = file_list[i+2]
                else:
                    outname = file_list[i+1]
                outtxt_list.append(outname)

    intxt_list.append(fcin)
    outtxt_list.append(fcout)

    j = 0
    bit = 7
    in_q = 5

    for i in range(1, layer_cnts+1):
        if i == 1:
            C_in, H_in, W_in = get_fstin_CHW(net)
        elif i in downsample:
            C_in, H_in, W_in = get_out_CHW(i-2)
        else:
            C_in, H_in, W_in = get_out_CHW(i-1)
        C_out, H_out, W_out = get_out_CHW(i)
        if i in pool_num:
            out_q = in_q
            continue
        out_q = bit - math.ceil(math.log2(0.5 * alphalist[j]))
        input_txt = f'{bmpdt}/{intxt_list[j]}'
        output_txt = f'{bmpdt}/{outtxt_list[j]}'
        input_bin = f'{bindir}/layer{i}_input_act0_file.bin'
        output_bin = f'{bindir}/layer{i}_output_file.bin'
        write_in = threading.Thread(target=write_inout_bin,
                                    args=(input_txt, input_bin, H_in, W_in, C_in, in_q))
        write_in.start()
        write_in.join()
        write_out = threading.Thread(target=write_inout_bin,
                                     args=(output_txt, output_bin, H_out, W_out, C_out, out_q))
        write_out.start()
        write_out.join()
        j += 1
        in_q = out_q

def get_layernum_to_type(logpath):
    num_type = [[]]
    with open(logpath, 'r') as fd:
        for line in fd:
            line.strip()
            if "layer type:" in line:
                layer_dec = line[line.find("layer type:"):line.find("form layer_num")]
                layer_num = line.split(' ',1)[0].split(':',1)[1].strip()
                num_type.append([layer_num, layer_dec.strip()])
    return num_type

def write_tile_inst(inst_256, insts):
    for i in range(len(insts)):
        for j in range(len(insts[i])):
            inst_256 = '{}{}'.format(inst_256, insts[i][j][1])

    return inst_256

def fillings(c, cnt):
    inst_zero = ""
    for i in range(cnt):
        inst_zero = '{}{}'.format(inst_zero, c)
    return inst_zero

def write_inst(fw, tile_cnt, insts, wetbit):
    cnt = 0
    inst_name = ""

    for i in range(tile_cnt):
        fmain.write(f'write_insts {insts[i][0]}...\n')
        for j in range(1, len(insts[i])):
            if j == 1:
                cnt = 5
                inst_name = "inst_id=2'h00"
            elif j == 2:
                cnt = 182
                inst_name = "inst_id=2'h01"
            elif j == 3:
                cnt = 5
                inst_name = "inst_id=2'h11"
            inst_256 = fillings('0', cnt)
            inst_256 = write_tile_inst(inst_256, insts[i][j])

            for k in range(int(256/wetbit)):
                dec = int(inst_256[k*wetbit:(k+1)*wetbit], 2)
                fw.write(st.pack('B', dec))
            fmain.write(f'{insts[i][0]} {inst_name} write data success\n')

def write_data(data_locate, fw):
    for i in range(len(data_locate)):
        if i == 0:
            fmain.write(f'write_datas show data_locate: {data_locate[i]} detail\n')
            continue
        for j in range(len(data_locate[i])):
            for k in range(len(data_locate[i][j])):
                ints = int(data_locate[i][j][k], 16)
                fw.write(st.pack('B', ints))

def write_conf(num_type, actbit, wetbit, chipid, lenreal, fconf, net, binsize):
    cnts = num_type[0]
    type = num_type[1].split(':')[1]

    if actbit == 16:
        actbit = "01"
    elif actbit == 8:
        actbit = "00"
    elif actbit == 4:
        actbit = "10"
    else:
        actbit = "11"

    if wetbit == 8:
        wetbit = '0'
    elif wetbit == 4:
        wetbit = '1'

    lenreal = '{:012b}'.format(lenreal-1)
    lenread = '{:012b}'.format(binsize-1)
    chipid = '{:02b}'.format(chipid)

    adwrap(fconf, f'{net} layer information:')
    adwrap(fconf, f'layer num        :  {cnts}')
    adwrap(fconf, f'layer type       :  {type}\n')

    adwrap(fconf, f'0x09 register value:')
    adwrap(fconf, f'act_bit          :   {actbit}')
    adwrap(fconf, f'weight_bit       :   {wetbit}')
    adwrap(fconf, f'inst_len_read    :   {lenread}')
    adwrap(fconf, f'inst_len_real    :   {lenreal}')
    adwrap(fconf, f'chiplet_id       :   {chipid}\n')

    adwrap(fconf, f'all files path:')
    if "pool" not in type:
        adwrap(fconf, f'act0   bin       :   layer{cnts}_input_act0_file.bin')
        adwrap(fconf, f'output bin       :   layer{cnts}_output_file.bin')
        adwrap(fconf, f'weight bin       :   layer{cnts}_data.bin')
    adwrap(fconf, f'insts  bin       :   layer{cnts}_inst.bin')
    adwrap(fconf, f'confs  txt       :   config_{cnts}.txt')
