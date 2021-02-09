import io
import os
import sys
import time
import torch
import shutil
import threading
import numpy as np
import struct as st
from optparse import OptionParser
from script.load_pt import load_pt

div_size = 8
ADDRBLOCK = 0
CALCULATE = 0
name_list = []
data_list = []
BUS_WIDTH = 256
ALIGN_WIDTH = 64
pool_fc_outc_bac = 0
pool_fc_outh_bac = 0
pool_fc_outw_bac = 0
dram_base = 0
dram_capacity = 1 << 30

datas_locate = ["", []]

#nnbaton参数
nnbaton_X1 = 2
nnbaton_Y1 = 2
nnbaton_K1 = 2
nnbaton_X2 = 7
nnbaton_Y2 = 1
nnbaton_K2 = 2
nnbaton_Kp = 1
nnbaton_Yp = 4
nnbaton_Kc = 1
nnbaton_Yc = 4
nnbaton_Xc = 2
nnbaton_C1 = 1
nnbaton_C0 = 8
nnbaton_X0 = 8
nnbaton_Y0 = 8

def option_parse():
    usage = "Usage: %prog [options] arg1 arg2 ... "
    parser = OptionParser(usage)

    parser.add_option("-p", "--pt_dir",      dest = "pt_name",    help = "PT file directory",      action = "store",
                      type = "string",       default = f'input/vgg_imagenet.pt')
    parser.add_option("-n", "--net_dir",     dest ="net_name",    help = "Vggnet file directory",  action = "store",
                      type = "string",       default = f'input/vgg_imagenet.py')
    parser.add_option("-o", "--output_dir",  dest ="output_dir",  help = "Output file directory",  action = "store",
                      type = "string",       default = f'{os.getcwd()}/output')
    parser.add_option("-c", "--conf_dir",    dest ="conf_dir",    help = "Config file directory",  action = "store",
                      type = "string",       default = f'debug/config')
    parser.add_option("-t", "--ptdata_dir",  dest ="ptdata_dir",  help = "Ptdata file directory",  action = "store",
                      type = "string",       default = f'debug/ptdata')
    parser.add_option("-m", "--bmpdata_dir", dest ="bmpdata_dir", help ="Bmpdata file directory",  action = "store",
                      type = "string",       default = f'debug/bmpdata')

    (options,args)=parser.parse_args()
    return options

def gen_fpga(filepath):
    """
    description:
                Generate bin files for fpga
    parameters:
                filepath: Run config_gen_file.py directory
    return code:
                None
    """
    os.chdir(filepath)
    cmd_list = [ "rm -rf con*txt cfg*txt *bn* *bias* *alpha* *weight* *input* *output* \
                  *k* data_for_fpga",
                 f'cp -af {confpath}/* {ptdtpath}/* {bmpdtpath}/* imagenet/',
                 "python ../input/config_gen_file.py -d ../imagenet/ -n imagenet_img6 -f",
                 f'mv {filepath}/data_for_fpga {outputpath}' ]

    for i in range(len(cmd_list)):
        if "cp -af " in cmd_list[i] or "mv " in cmd_list[i]:
            os.chdir("..")
        elif "python" in cmd_list[i]:
            os.chdir(filepath)
        os.system(cmd_list[i])
 
    os.chdir("..")

    print("run gen_fpga successfully")

def write_fck_data(fw, data, cnt):
    """
    description:
                write data to classifier.k file
    parameters:
                fw:   File descriptor to be written
                data: Data to be written to file
                cnt:  The counts of times data is written to the file
    return code:
                None
    """
    for i in range(cnt):
        fw.write(data)

def deal_fc_k(vgglog):
    """
    description:
                Generate model files
    parameters:
                vgglog: Path to vgglog file
    return code:
                None
    """
    fc_k_list  = []
    data_list  = []
    fc_channel = []

    with open(vgglog, 'r') as file:
        line = file.readline()
        while line:
            line = file.readline()
            if line.startswith("fc") and "out_features_y" in line:
                fc_channel.append(line.split(' ', 5)[4].split(':')[1].strip())

    list = os.listdir(bmpdtpath)
    for i in range(len(list)):
        if ".k.txt" in list[i]:
            fc_k_list.append(list[i])

    for i in range(len(fc_k_list)):
        with open(f'{bmpdtpath}/{fc_k_list[i]}', 'r') as fd:
            line = fd.readline()
            data_list.append(line)
            
    for i in range(len(fc_k_list)):
        with open(f'{bmpdtpath}/{fc_k_list[i]}', 'w') as fw:
            if len(fc_channel) == 1:
                write_fc_k = threading.Thread(target=write_fck_data,
                                      args=(fw, data_list[i], int(fc_channel[0])))
            else:
                write_fc_k = threading.Thread(target=write_fck_data,
                                      args=(fw, data_list[i], int(fc_channel[i])))
            write_fc_k.start()
            write_fc_k.join()
        if len(fc_channel) == 1:
            print("write %s to %s file %s times success..."
                  %(data_list[i].strip(), fc_k_list[i], fc_channel[0]))
        else:
            print("write %s to %s file %s times success..."
                  %(data_list[i].strip(), fc_k_list[i], fc_channel[i]))
    
def gen_bmp(vgglog):
    """
    description:
                Generate model files
    parameters:
                vgglog: Path to vgglog file
    return code:
                None
    """
    cmd_list = [ f'python3 input/inout_print.py input/im6.bmp {ptpath} {bmpdtpath}',
                 f'ls {bmpdtpath}/quant_fc3.output.int.6.txt' ]

    for i in range(len(cmd_list)):
        os.system(cmd_list[i])

    deal_fc_k(vgglog)
    print("run gen_bmp successfully")

def get_cnts():
    """
    description: Get counts of layers
    parameter: NULL
    return value: NULL
    """
    layer_cnts = 0

    with open(logpath, 'r') as f:
        for lines in f.readlines():
            if "layer_num:" in lines:
                layer_cnts += 1

    return layer_cnts

def bus_address():
    return int(ADDRBLOCK // (BUS_WIDTH / div_size))

def index_to_address(N, C, H, W, n, c, h, w, bit_mode):

    assert((bit_mode == 4) | (bit_mode == 8) | (bit_mode == 16)), "bit_mode should be 4, 8 or 16."

    sub_channel = ALIGN_WIDTH / bit_mode
    bus_width_byte = BUS_WIDTH / bit_mode

    C = C//sub_channel * sub_channel
    C0 = sub_channel

    c0 = np.mod(c, sub_channel)
    c_ = c//C0

    word_address = int(c0 + w*C0 + h*W*C0 + c_*H*W*C0)
    bus_address = int(word_address // bus_width_byte)

    return word_address, bus_address

def dram_allocate(n):
    new_dram_base = dram_base + n
    #assert(new_dram_base <= dram_capacity, "dram_capacity not enough")
    dram_base_bak = dram_base
    dram_base = new_dram_base
    return dram_base_bak

def align(address, factor):
    if address % factor == 0:
        return address
    else:
        return (address // factor + 1) * factor 

def get_layer_num(layer_name):
    #该函数可以不用遍历，在遍历维度的时候保存fclist_num=[21,22,23]即可。减少循环遍历
    pool_num = []
    with open(logpath, 'r') as file:
        for line in file:
            if "fc" in layer_name:
                if line.startswith("layer_num:") and layer_name in line:
                    return(int(line.split(" ")[0].split(":")[1]))
            elif layer_name in line:
                if line.startswith("layer_num:"):
                    poolname = line.split(":", 5)[2].split(" ")[0]
                    if "pool" not in poolname:
                        continue
                    pool_num.append(int(line.split(" ")[0].split(":")[1]))

    return pool_num

def HexProcess(decimal):
    if decimal < 15 and decimal >= 0:
        hex_num = '0' + hex(decimal & 0xFF)[-1:]
    else:
        hex_num = hex(decimal & 0xFF)[-2:]
    return hex_num

def weight_addr():
    """
    description:
                Load pt file and format output
    parameters:
                loadpt: The Class of load_pt
    return code:
                None
    """
    k = padding = 0
    fc_flag = False
    word_address = 1
    weight_bn_k = []
    weight_bn_b = []
    weight_data = []
    global ADDRBLOCK
    layer_locate = []
    N = C = H = W = 0
    scale = fc_cnt = 0
    global datas_locate

    def pt_skip(name):
        if "alpha" in name or "running_" in name or "num_batches" in name\
              or "weight" in name and "bn" in name or "bias" in name:
            return 1

    print("weight_addr start:", ADDRBLOCK, " bus_addr:", bus_address())
    for i in range(len(name_list)):
        name = name_list[i]
        if pt_skip(name):
            continue
        data = data_list[i]
        if "scale" in name:
            scale = data.tolist()
        elif "bn_k" in name:
            weight_bn_k = data
        elif "bn_b" in name:
            weight_bn_b = data
        elif "weight" in name:
            if "classifier" in name:
                fc_flag = True
            weight_data = data
        if scale and(len(weight_bn_k) and len(weight_bn_b) and len(weight_data) or fc_flag):
            dim_list = list(weight_data.size())
            dim_lens = len(dim_list)
            if dim_lens == 4:
                N = dim_list[0]
                C = dim_list[1]
                H = dim_list[2]
                W = dim_list[3]
            elif dim_lens == 2:
                N = C = 1
                H = dim_list[0]
                W = dim_list[1]
            for n in range(N):
                if len(weight_bn_k) and len(weight_bn_b):
                    hexdata_k = '%X' % st.unpack('I', st.pack('f', weight_bn_k[n]))[0]
                    hexdata_b = '%X' % st.unpack('I', st.pack('f', weight_bn_b[n]))[0]
                    for x in range(len(str(hexdata_k))):
                        if (x % 2) == 0:
                            layer_locate.append(hexdata_k[x] + hexdata_k[x+1])
                    for y in range(len(str(hexdata_b))):
                        if (y % 2) == 0:
                            layer_locate.append(hexdata_b[y] + hexdata_b[y+1])
                for c in range(C):
                    for h in range(H):
                        if dim_lens == 2:
                            for i in range(4 * 2):
                                layer_locate.append("00")
                        for w in range(W):
                            word_address = w + h*W + c*H*W + n*C*H*W
                            word_address += padding
                            if dim_lens == 4:
                                rounds = round(weight_data[n][c][h][w].tolist() / scale)
                            elif dim_lens == 2:
                                for i in range(4 * 2):
                                    layer_locate.append("00")
                                rounds = round(weight_data[h][w].tolist() / scale)
                            else:
                                print(f'Unknown weight:{name} length:{dim_lens} dim:{dim_list}')
                                continue
                            hexdata_w = HexProcess(rounds)
                            layer_locate.append(hexdata_w)
                word_address += BUS_WIDTH
                padding = align(word_address, BUS_WIDTH) - word_address
                for i in range(padding-word_address):
                    layer_locate.append("00")
            layer_cnt = int(name.split(".", 4)[1])+1
            if "classifier" in name:
                fc_cnt += 1
                layer_cnt = get_layer_num(f'fc{fc_cnt}')
            if fc_flag and len(layer_locate):
                print(".................Start Rearrange weight..............")
                Rearrange_locate = []
                for i in range(int(len(layer_locate) / (4 * 4))):
                    for j in range(4 * 4):
                        if j % 2 == 0:
                            Rearrange_locate.append(layer_locate[j + 4 * 4 * i])
                    for j in range(4 * 4):
                        if j % 2:
                            Rearrange_locate.append(layer_locate[j + 4 * 4 * i])
                layer_locate = Rearrange_locate
            print(f'layer {str(layer_cnt)} bn_k+bn_b+conv_weight data', " save data success")
            datas_locate[k] = [f'layer {str(layer_cnt)} bn_k+bn_b+conv_weight data', [layer_locate]]
            weight_bn_k = weight_bn_b = weight_data = layer_locate = []
            scale = fc_flag = 0
            datas_locate.append(datas_locate[k])
            k += 1
    ADDRBLOCK += word_address + padding
    print("weight_addr   end:", ADDRBLOCK, "bus_addr:", bus_address())
    ADDRBLOCK = align(ADDRBLOCK, BUS_WIDTH)
    print("###########weight_addr   after align 256:", ADDRBLOCK)

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
    quant_list = []
    global name_list
    global data_list  
    onelayer_cnt = []

    loadpt.get_tensorinfo(netpath)

    with open(f'{ptdtpath}/img.input.q.txt', 'w') as fw:
        fw.write(loadpt.in_q)
        fw.write("\n")
        print(f'{ptdtpath}/img.input.q.txt write success')

    with open(ptpath, 'rb') as f:
        buffer = io.BytesIO(f.read())
        dict = torch.load(buffer, map_location=torch.device('cpu'))
        for k, v in dict.items():
            if "quant_" in k or "classifier." in k:
                quant_list.append(k)
                quant_list.append(v)
            name_list.append(k)
            data_list.append(v)

    for i in range(loadpt.layer_cnts):
        layer = f'layers.{i}.'
        for j in range(len(name_list)):
            if layer in name_list[j]:
                loadpt.layers.append([name_list[j], data_list[j]])
                counts += 1
        onelayer_cnt.append(str(counts))
        counts = 0

    del (loadpt.layers[0])
    for i in range(loadpt.layer_cnts):
        layername = f'layer_num:{str(i + 1)}'
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

    print("run gen_txt successfully")

'''
获取相应的数据
参数： layernum：哪一层的数据；  find_name：获取标志(日志中数据的名称加：)
'''
def find_count(layernum, find_name):
    with open(logpath, 'r') as file:
        for line in file:
            if line.startswith(f"layer_num:{layernum+1} "):
                break
            if line.startswith(f"layer_num:{layernum} "):
                for line in file:
                    if line.startswith(f"layer_num:{layernum + 1} "):
                        break
                    if find_name in line:
                        return(int(line.split(find_name)[1].split(" ")[0]))

'''
获取相应的层的类型
参数： layernum：哪一层的类型；  find_name：获取标志(日志中数据的名称加：)
'''
def find_type(layernum, find_name):
    with open(logpath, 'r') as file:
        for line in file:
            if line.startswith(f"layer_num:{layernum} "):
                return(line.split(find_name)[1].split(" ")[0])

'''
获取无效的参数
参数： countnum:返回多少位无效的数字
'''
def return_value(countnum):
    return_value = []
    for i in range(countnum):
        return_value.append('0')
    return ''.join(return_value)

'''
获取当前指令的ID
'''
def get_Inst_id_00():
    return '{:02b}'.format(00)

'''
获取act  tile的水平(hor)和垂直(var)方向的大小，通过output推算input
'''
def get_act_tile_hor_var(layernum):
    out_H = find_count(layernum, "feature_map_size_y:")
    out_W = find_count(layernum, "feature_map_size_x:")
    stride = find_count(layernum, "stride_x:")
    kernel_size = find_count(layernum, "kernel_size_x:")
    out_tail_hor = out_W / nnbaton_X2
    out_tail_ver = out_H / nnbaton_Yp / nnbaton_Y2
    act_tail_hor = out_tail_hor * stride - 1 + kernel_size
    act_tail_ver = out_tail_ver * stride - 1 + kernel_size
    return act_tail_hor, act_tail_ver
'''
获取act  tile的水平方向的大小
'''
def get_act_tile_hor(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        act_tile_hor = int(get_act_tile_hor_var(layernum)[0])
        return '{:07b}'.format(act_tile_hor - 1)
    else:
        return return_value(7)

'''
获取act  tile的垂直方向的大小
'''
def get_act_tile_ver(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        act_tile_ver = int(get_act_tile_hor_var(layernum)[0])
        return '{:07b}'.format(act_tile_ver - 1)
    else:
        return return_value(7)

'''
获取act tile在input channel方向上有多少个stride大小的数据块
'''
def get_act_tile_chl(layernum):
    # 8bit模式
    input_channel = find_count(layernum, "in_channels:")
    if input_channel != None:
        return '{:09b}'.format(int(align(input_channel, 16) / 16 - 1))
    else:
        return return_value(9)

    # 16bit模式
    input_channel = find_count(layernum, "in_channels:")
    if input_channel != None:
        return '{:09b}'.format(int(align(input_channel, 8) / 8 - 1))
    else:
        return return_value(9)
    # 4bit模式
    input_channel = find_count(layernum, "in_channels:")
    if input_channel != None:
        return '{:09b}'.format(int(align(input_channel, 16) / 16 - 1))
    else:
        return return_value(9)
'''
获取act tile的多少bit为一组，第一层为64bit，其它层为128bit
'''
def get_act_tile_str(layernum):
    if layernum == 1:
        return '{:02b}'.format(0b1)
    else:
        return '{:02b}'.format(0b0)

"""
获取act sub_tile水平和垂直方向大小
"""
def get_act_subtile_hor_ver(layer_num):
    out_H = find_count(layer_num, "feature_map_size_y:")
    out_W = find_count(layer_num, "feature_map_size_x:")
    stride = find_count(layer_num, "stride_x:")
    kernel_size = find_count(layer_num, "kernel_size_x:")
    out_tail_hor = int(out_W / nnbaton_X2 / nnbaton_Xc)
    out_tail_ver = int(out_H / nnbaton_Yp / nnbaton_Y2 / nnbaton_Yc)
    act_tail_hor = out_tail_hor * stride - 1 + kernel_size
    act_tail_ver = out_tail_ver * stride - 1 + kernel_size
    return act_tail_hor, act_tail_ver
'''
获取act tile向下游切换的ping pang新号的数据量大小，相当于input的sub_tile大小
'''
def get_act_tile_sub_chl(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        act_subtail = get_act_subtile_hor_ver(layernum)
        return '{:09b}'.format(act_subtail[0] * act_subtail[1])
    else:
        return return_value(9)
'''
获取sub_core处理tile的水平方向上大小
'''
def get_sub_tile_hor(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        sub_tile_hor = get_act_subtile_hor_ver(layernum)[0]
        return '{:07b}'.format(sub_tile_hor - 1)
    else:
        return return_value(7)

'''
获取sub_core处理tile的垂直方向上大小
'''
def get_sub_tile_ver(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        sub_tile_ver = get_act_subtile_hor_ver(layernum)[1]
        return '{:07b}'.format(sub_tile_ver - 1)
    else:
        return return_value(7)

'''
任务块在output channel方向上需要计算的次数
'''
def get_out_chl_num(layernum):
    pass

'''
获取act mini_tile水平和垂直方向大小
'''
def get_act_minitile_hor_ver(layer_num):
    out_H = find_count(layer_num, "feature_map_size_y:")
    out_W = find_count(layer_num, "feature_map_size_x:")
    stride = find_count(layer_num, "stride_x:")
    kernel_size = find_count(layer_num, "kernel_size_x:")
    out_tail_hor = int(out_W / nnbaton_X2 / nnbaton_Xc / nnbaton_X1)
    out_tail_ver = int(out_H / nnbaton_Yp / nnbaton_Y2 / nnbaton_Yc / nnbaton_Y1)
    act_tail_hor = out_tail_hor * stride - 1 + kernel_size
    act_tail_ver = out_tail_ver * stride - 1 + kernel_size
    return act_tail_hor, act_tail_ver
'''
获取Mini tile的宽度
'''
def get_Mini_tile_hor(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        mini_tile_hor = get_act_minitile_hor_ver(layernum)[0]
        return '{:05b}'.format(mini_tile_hor - 1)
    else:
        return return_value(5)

'''
获取Mini tile的高度
'''
def get_Mini_tile_ver(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        mini_tile_ver = get_act_minitile_hor_ver(layernum)[1]
        return '{:05b}'.format(mini_tile_ver - 1)
    else:
        return return_value(5)

'''
获取任务块中需要计算的chl数据
'''
def get_in_chl_num(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        act_input_channel = find_count(layernum, "out_channels:")
        act_input_channel = int(act_input_channel / nnbaton_Kp / nnbaton_K2)
        return '{:09b}'.format(act_input_channel - 1)
    else:
        return return_value(9)

'''
sub_core计算出来的mini tile的水平方向大小
'''
def get_Out_mini_tile_hor(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        out_hor = find_count(layernum, "feature_map_size_x:")
        mini_tile_hor = int(out_hor / nnbaton_X2 / nnbaton_Xc / nnbaton_X1)
        return '{:07b}'.format(mini_tile_hor - 1)
    else:
        return return_value(7)

'''
sub_core计算出来的mini tile的垂直方向大小
'''
def get_Out_mini_tile_ver(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        out_ver = find_count(layernum, "feature_map_size_y:")
        mini_tile_hor = int(out_ver / nnbaton_Yp / nnbaton_Y2 / nnbaton_Y1)
        return '{:07b}'.format(mini_tile_hor - 1)
    else:
        return return_value(7)

'''
获取output tile的水平方向大小
'''
def get_Out_tile_hor(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        out_W = find_count(layernum, "feature_map_size_x:")
        out_tile_hor = int(out_W / nnbaton_X2)
        return '{:07b}'.format(out_tile_hor - 1)
    else:
        return return_value(7)

'''
获取output tile的垂直方向大小
'''
def get_Out_tile_ver(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        out_H = find_count(layernum, "feature_map_size_y:")
        out_tail_ver = int(out_H / nnbaton_Yp / nnbaton_Y2)
        return '{:07b}'.format(out_tail_ver - 1)
    else:
        return return_value(7)

'''
获取output tile的通道方向大小
'''
def get_Out_tile_chl(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        out_C = find_count(layernum, "out_channels:")
        out_tile_chl = int(out_C / nnbaton_K2)
        return '{:07b}'.format(out_tile_chl - 1)
    else:
        return return_value(7)

'''
获取一个output channel的weight大小
in_channel和out_channel相等，不需要反推
'''
def get_weight_total_size(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        output_channel = int(find_count(layernum, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        kernel_size = find_count(layernum, "kernel_size_x:") * find_count(layernum, "kernel_size_y:")
        weight_size = kernel_size * int(align(output_channel, 8)/8)
        return '{:017b}'.format(weight_size)
    else:
        return return_value(17)

def get_Weight_sub_size(layernum):
    pass

'''
获取weight ram可以存下output_channel/tile_mode的个数
'''
def get_weight_ram_output_chl_num(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        output_channel = int(find_count(layernum, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        tile_mode = int(8 / (nnbaton_Xc * nnbaton_Yc))
        return '{:07b}'.format(int(output_channel / tile_mode) - 1)
    else:
        return return_value(7)

'''
获取卷积类型
00：普通卷积
01：空洞卷积
10：反卷积；
11：保留
'''
def get_conv_type(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        return '{:02b}'.format(0)
    else:
        return return_value(2)

'''
获取chiplet的工作模式
kp = 4时为10（四块chiplet），kp = 1时为00（chiplet各自工作）
'''
def get_Chiplet_mode(layernum):
    if nnbaton_Kp == 4:
        return '{:02b}'.format(0b10)
    elif nnbaton_Kp == 1:
        return '{:02b}'.format(0)

'''
获取多个chiplet协同工作时，每块input channel对应的weight存储
等于weight_total_size/chiplet_mode
'''
def get_Chiplet_tile_size(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        output_channel = int(find_count(layernum, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        kernel_size = find_count(layernum, "kernel_size_x:") * find_count(layernum, "kernel_size_y:")
        weight_size = kernel_size * int(align(output_channel, 8) / 8 / nnbaton_Kp)
        return '{:017b}'.format(weight_size)
    else:
        return return_value(17)

'''
获取kernel宽度乘以高度的大小
FC的kernel_size为1*1
'''
def get_Kernel_num(layernum):
    if "fc" in find_type(layernum, "layer type:"):
        kernel_num = 1
        return '{:08b}'.format(kernel_num - 1)
    else:
        kernel_width = find_count(layernum, "kernel_size_x:")
        kernel_height = find_count(layernum, "kernel_size_y:")
        kernel_num = kernel_height * kernel_width
        return '{:08b}'.format(kernel_num - 1)

'''
获取kernel宽度的大小
'''
def get_Kernel_width(layernum):
    if "fc" in find_type(layernum, "layer type:"):
        Kernel_width = 1
        return '{:04b}'.format(Kernel_width - 1)
    else:
        Kernel_width = find_count(layernum, "kernel_size_y:")
        return '{:04b}'.format(Kernel_width - 1)

'''
获取kernel高度的大小
'''
def get_Kernel_height(layernum):
    if "fc" in find_type(layernum, "layer type:"):
        kernel_height = 1
        return '{:04b}'.format(kernel_height - 1)
    else:
        kernel_height = find_count(layernum, "kernel_size_y:")
        return '{:04b}'.format(kernel_height - 1)

'''
获取步长大小
000：跨度为1；   001：跨度为2；   010：跨度为3；   011：跨度为4；
100：跨度为5；   101：跨度为6；   110：跨度为7；   111：跨度为8；
'''
def get_Kernel_str(layernum):
    Kernel_str = find_count(layernum, "stride_x:")
    if Kernel_str == None:
        return return_value(3)
    else:
        return '{:03b}'.format(Kernel_str - 1)

'''
获取反卷积在水平方向上的跨度
'''
def get_deconv_hor_str(layernum):
    return '{:02b}'.format(0)

'''
获取反卷积在垂直方向上的跨度
'''
def get_deconv_ver_str(layernum):
    return '{:02b}'.format(0)

'''
获取反卷空洞卷积，插入零的个数
'''
def get_Dilation_rate(layernum):
    return '{:03b}'.format(0)

'''
取整方式选择
取整方式先按照1获取，后期在pt文件中获取
'''
def get_pooling_carry_sel(layernum):
    return '{:01b}'.format(1)

'''
返回0：平均池化；(avgpooling)
返回1：最大池化；(maxpooling)
非pool返回0
'''
def get_pool_mode(layernum):
    pool_mode = find_type(layernum, "layer type:")
    if "avgpool" in pool_mode:
        return '{:01b}'.format(0)
    elif "maxpool" in pool_mode:
        return '{:01b}'.format(1)
    else:
        return return_value(1)

'''
表示定点转32位浮点的量化系数（暂时先不管）
'''
def get_Pooling_quan_code_in(layernum):
    pass

'''
池化的尺寸乘积结果
'''
def get_pool_total_size(layernum):
    pool_total_size = find_count(layernum, "pool_size:")
    if pool_total_size == None:
        return return_value(10)
    else:
        return '{:010b}'.format(pool_total_size * pool_total_size - 1)

'''
获取池化的size
'''
def get_pool_size(layernum):
    pool_size = find_count(layernum, "pool_size:")
    if pool_size == None:
        return return_value(5)
    else:
        return '{:05b}'.format(pool_size)

'''
给平均池化使用的任意尺寸的倒数
'''
def get_Pooling_oprands(layernum):
    type = find_type(layernum, "layer type:")
    if "avgpool" in type:
        pool_size = find_count(layernum, "pool_size:")
        return '{:032b}'.format(int(st.unpack('I', st.pack('f', 1 / (pool_size * pool_size)))[0]))
    else:
        return return_value(32)

'''
表示32位浮点转定点的量化系数（暂时不管）
'''
def get_Pooling_quan_code_out(layernum):
    pass

'''
scaling放大倍数:
00：bilinearx2   01：bilinearx4
10：nearestx2    11: nearestx4
'''
def get_scaling_mode(layernum):
    pass

'''
全连接模式使能信号，返回1有效
'''
def get_fc_mode_en(layernum):
    fc_mode = find_type(layernum, "layer type:")
    if "fc" in fc_mode:
        return '{:01b}'.format(1)
    else:
        return return_value(1)

############################################################################
'''
获取当前指令的ID
'''
def get_Inst_id_01():
    return '{:02b}'.format(0b01)

'''
获取各个sub core之间拼tile的模式
'''
def get_Tile_mode():
    tile_mode = nnbaton_Xc * nnbaton_Yc
    if tile_mode == 1:
        return '{:02b}'.format(0b00)
    elif tile_mode == 2:
        return '{:02b}'.format(0b01)
    elif tile_mode == 4:
        return '{:02b}'.format(0b10)
    elif tile_mode == 8:
        return '{:02b}'.format(0b11)

'''
获取当前指令中，需要计算的tile的个数
'''
def get_tile_num(layernum):
    return '{:05b}'.format(nnbaton_X1 + nnbaton_Y1)


def get_Padding_mode(layernum):
    pass

def get_Padding_num(layernum):
    pass

def get_Repeat_send_num(layernum):
    pass

def get_act_inst_bypass(layernum):
    pass

def get_weight_inst_bypass(layernum):
    pass

def get_act_chl_one_inst_real(layernum):
    pass

def get_act_sub_ch(layernum):
    pass

def get_act_chl_one_inst(layernum):
    pass

def get_inst_invalid(layernum):
    pass

'''
获取一个sub_tile中水平方向上mini_tile的数量
'''
def get_mini_ver_num(layernum):
    return '{:07b}'.format(nnbaton_X1)

'''
获取一个sub_tile中垂直方向上mini_tile的数量
'''
def get_mini_hor_num(layernum):
    return '{:07b}'.format(nnbaton_Y1)


############################################################################
'''
获取当前指令的ID
'''
def get_Inst_id_11():
    return '{:02b}'.format(0b11)

'''
获取当前执行计算模式：
00：卷积模式；    01：pooling模式；
10：element wise模式；  11：scaling模式；
'''
def get_Run_mode(layernum):
    if "conv" in type:
        return '{:02b}'.format(0b00)
    elif "pooling" in type:
        return '{:02b}'.format(0b01)
    elif "element wise" in type:
        return '{:02b}'.format(0b10)
    elif "scaling" in type:
        return '{:02b}'.format(0b11)

def get_weight_addr(layernum):
    pass

def get_weight_output_chl(layernum):
    if "conv" in find_type(layernum, "layer type:"):
        output_chl = int(find_count(layernum, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        return '{:011b}'.format(output_chl - 1)
    else:
        return return_value(11)

def get_weight_updata_n(layernum):
    pass

def get_LLC_w_ping_pong(layernum):
    pass

'''
从存储器读取每一个tile的数据存储起始地址
'''
def get_act_addr(layernum):
    pass

def get_act_str_chl(layernum):
    pass

def get_act_str_line(layernum):
    pass

def get_act_updata_n(layernum):
    pass

def get_LLC_a_ping_pong(layernum):
    pass

def get_Out_tile_start_addr(layernum):
    pass

def get_Out_tile_stride(layernum):
    pass

def get_Out_feature_map_ver(layernum):
    pass

def get_act_addr_element(layernum):
    pass


def get_nnbaton(layernum, pool_num):
    if layernum == 1:
        return 2, 2, 2, 7, 1, 2, 1, 4, 1, 4, 2, 1, 8, 8, 8
    elif layernum == 2:
        return 2, 1, 1, 7, 7, 1, 1, 4, 4, 1, 2, 8, 8, 8, 8
    elif layernum == 4:
        return 1, 1, 1, 7, 7, 1, 4, 1, 2, 2, 2, 2, 8, 8, 8
    elif layernum == 5:
        return 1, 1, 1, 7, 7, 1, 4, 1, 2, 2, 2, 4, 8, 8, 8
    elif layernum == 7:
        return 1, 1, 1, 4, 7, 1, 4, 1, 4, 1, 2, 4, 8, 8, 8
    elif layernum == 8:
        return 1, 1, 1, 4, 7, 1, 4, 1, 4, 1, 2, 8, 8, 8, 8
    elif layernum == 9:
        return 1, 1, 1, 4, 7, 1, 4, 1, 4, 1, 2, 8, 8, 8, 8
    elif layernum == 11:
        return 1, 1, 1, 2, 4, 2, 4, 1, 4, 1, 2, 8, 8, 8, 8
    elif layernum == 12:
        return 1, 1, 4, 2, 1, 2, 4, 1, 1, 4, 2, 16, 8, 8, 8
    elif layernum == 13:
        return 1, 1, 4, 2, 1, 2, 4, 1, 1, 4, 2, 16, 8, 8, 8
    elif layernum == 15:
        return 1, 1, 2, 1, 1, 2, 4, 1, 2, 2, 2, 16, 8, 8, 8
    elif layernum == 16:
        return 1, 1, 2, 1, 1, 2, 4, 1, 2, 2, 2, 16, 8, 8, 8
    elif layernum == 17:
        return 1, 1, 2, 1, 1, 2, 4, 1, 2, 2, 2, 16, 8, 8, 8
    elif i in pool_num:
        return 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    elif layernum == 21 or layernum == 22 or layernum == 23:
        return 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

def get_layer_inst(layernum, tile_cnt):
    for i in range(1, tile_cnt+1):
        inst_00, inst_01, inst_11 = get_tile_inst(i, layernum)
        insts[i-1] = [f'layer{layernum} tile{i} instruction', [inst_00], [inst_01], [inst_11]]
        insts.append(insts[i-1])

def get_tile_inst(tile_num, layer_num):
    Inst_id_00 = get_Inst_id_00()
    act_tile_hor = get_act_tile_hor(layer_num)
    act_tile_ver = get_act_tile_ver(layer_num)
    act_tile_chl = get_act_tile_chl(layer_num)
    act_tile_str = get_act_tile_str(layer_num)
    act_tile_sub_chl = get_act_tile_sub_chl(layer_num)
    sub_tile_hor = get_sub_tile_hor(layer_num)
    sub_tile_ver = get_sub_tile_ver(layer_num)
    out_chl_num = get_out_chl_num(layer_num)
    Mini_tile_hor = get_Mini_tile_hor(layer_num)
    Mini_tile_ver = get_Mini_tile_ver(layer_num)
    in_chl_num = get_in_chl_num(layer_num)
    Out_mini_tile_hor = get_Out_mini_tile_hor(layer_num)
    Out_mini_tile_ver = get_Out_mini_tile_ver(layer_num)
    Out_tile_hor = get_Out_tile_hor(layer_num)
    Out_tile_ver = get_Out_tile_ver(layer_num)
    Out_tile_chl = get_Out_tile_chl(layer_num)
    weight_total_size = get_weight_total_size(layer_num)
    Weight_sub_size = get_Weight_sub_size(layer_num)
    weight_ram_output_chl_num = get_weight_ram_output_chl_num(layer_num)
    conv_type = get_conv_type(layer_num)
    Chiplet_mode = get_Chiplet_mode(layer_num)
    Chiplet_tile_size = get_Chiplet_tile_size(layer_num)
    Kernel_num = get_Kernel_num(layer_num)
    Kernel_width = get_Kernel_width(layer_num)
    Kernel_height = get_Kernel_height(layer_num)
    Kernel_str = get_Kernel_str(layer_num)
    deconv_hor_str = get_deconv_hor_str(layer_num)
    deconv_ver_str = get_deconv_ver_str(layer_num)
    Dilation_rate = get_Dilation_rate(layer_num)
    pooling_carry_sel = get_pooling_carry_sel(layer_num)
    pool_mode = get_pool_mode(layer_num)
    Pooling_quan_code_in = get_Pooling_quan_code_in(layer_num)
    pool_total_size = get_pool_total_size(layer_num)
    pool_size = get_pool_size(layer_num)
    Pooling_oprands = get_Pooling_oprands(layer_num)
    Pooling_quan_code_out = get_Pooling_quan_code_out(layer_num)
    scaling_mode = get_scaling_mode(layer_num)
    fc_mode_en = get_fc_mode_en(layer_num)

    Inst_id_01 = get_Inst_id_01()
    Tile_mode = get_Tile_mode()
    tile_num = get_tile_num(layer_num)
    Padding_mode = get_Padding_mode(layer_num)
    Padding_num = get_Padding_num(layer_num)
    Repeat_send_num = get_Repeat_send_num(layer_num)
    act_inst_bypass = get_act_inst_bypass(layer_num)
    weight_inst_bypass = get_weight_inst_bypass(layer_num)
    act_chl_one_inst_real = get_act_chl_one_inst_real(layer_num)
    act_sub_ch = get_act_sub_ch(layer_num)
    act_chl_one_inst = get_act_chl_one_inst(layer_num)
    inst_invalid = get_inst_invalid(layer_num)
    mini_ver_num = get_mini_ver_num(layer_num)
    mini_hor_num = get_mini_hor_num(layer_num)

    Inst_id_11 = get_Inst_id_11()
    Run_mode = get_Run_mode(layer_num)
    weight_addr = get_weight_addr(layer_num)
    weight_output_chl = get_weight_output_chl(layer_num)
    weight_updata_n = get_weight_updata_n(layer_num)
    LLC_w_ping_pong = get_LLC_w_ping_pong(layer_num)
    act_addr = get_act_addr(layer_num)
    act_str_chl = get_act_str_chl(layer_num)
    act_str_line = get_act_str_line(layer_num)
    act_updata_n = get_act_updata_n(layer_num)
    LLC_a_ping_pong = get_LLC_a_ping_pong(layer_num)
    Out_tile_start_addr = get_Out_tile_start_addr(layer_num)
    Out_tile_stride = get_Out_tile_stride(layer_num)
    Out_feature_map_ver = get_Out_feature_map_ver(layer_num)
    act_addr_element = get_act_addr_element(layer_num)

    inst_00 = [
        ["Inst_id_00", Inst_id_00],
        ["act_tile_hor", act_tile_hor],
        ["act_tile_ver", act_tile_ver],
        ["act_tile_chl", act_tile_chl],
        ["act_tile_str", act_tile_str],
        ["act_tile_sub_chl", act_tile_sub_chl],
        ["sub_tile_hor", sub_tile_hor],
        ["sub_tile_ver", sub_tile_ver],
        ["out_chl_num", out_chl_num],
        ["Mini_tile_hor", Mini_tile_hor],
        ["Mini_tile_ver", Mini_tile_ver],
        ["in_chl_num", in_chl_num],
        ["Out_mini_tile_hor", Out_mini_tile_hor],
        ["Out_mini_tile_ver", Out_mini_tile_ver],
        ["Out_tile_hor", Out_tile_hor],
        ["Out_tile_ver", Out_tile_ver],
        ["Out_tile_chl", Out_tile_chl],
        ["weight_total_size", weight_total_size],
        ["Weight_sub_size", Weight_sub_size],
        ["weight_ram_output_chl_num", weight_ram_output_chl_num],
        ["conv_type", conv_type],
        ["Chiplet_mode  ", Chiplet_mode],
        ["Chiplet_tile_size", Chiplet_tile_size],
        ["Kernel_num", Kernel_num],
        ["Kernel_width", Kernel_width],
        ["Kernel_height", Kernel_height],
        ["Kernel_str", Kernel_str],
        ["deconv_hor_str", deconv_hor_str],
        ["deconv_ver_str", deconv_ver_str],
        ["Dilation_rate", Dilation_rate],
        ["pooling_carry_sel", pooling_carry_sel],
        ["pool_mode", pool_mode],
        ["Pooling_quan_code_in", Pooling_quan_code_in],
        ["pool_total_size", pool_total_size],
        ["pool_size", pool_size],
        ["Pooling_oprands", Pooling_oprands],
        ["Pooling_quan_code_out", Pooling_quan_code_out],
        ["scaling_mode", scaling_mode],
        ["fc_mode_en", fc_mode_en],
    ]

    inst_01 = [
        ["Inst_id_01", Inst_id_01],
        ["Tile_mode", Tile_mode],
        ["tile_num", tile_num],
        ["Padding_mode", Padding_mode],
        ["Padding_num", Padding_num],
        ["Repeat_send_num", Repeat_send_num],
        ["act_inst_bypass", act_inst_bypass],
        ["weight_inst_bypass", weight_inst_bypass],
        ["act_chl_one_inst_real", act_chl_one_inst_real],
        ["act_sub_ch", act_sub_ch],
        ["act_chl_one_inst", act_chl_one_inst],
        ["inst_invalid", inst_invalid],
        ["mini_ver_num", mini_ver_num],
        ["mini_hor_num", mini_hor_num],
    ]

    inst_11 = [
        ["Inst_id_11", Inst_id_11],
        ["Run_mode", Run_mode],
        ["weight_addr", weight_addr],
        ["weight_output_chl", weight_output_chl],
        ["weight_updata_n", weight_updata_n],
        ["LLC_w_ping_pong", LLC_w_ping_pong],
        ["act_addr", act_addr],
        ["act_str_chl", act_str_chl],
        ["act_str_line", act_str_line],
        ["act_updata_n", act_updata_n],
        ["LLC_a_ping_pong", LLC_a_ping_pong],
        ["Out_tile_start_addr", Out_tile_start_addr],
        ["Out_tile_stride", Out_tile_stride],
        ["Out_feature_map_ver", Out_feature_map_ver],
        ["act_addr_element", act_addr_element],
    ]

    return inst_00, inst_01, inst_11

def shex_to_int(shex):
    if shex.isdigit():
        return int(shex)
    elif ((shex=='a') or (shex=='A')):
        return 10
    elif ((shex=='b') or (shex=='B')):
        return 11
    elif ((shex=='c') or (shex=='C')):
        return 12
    elif ((shex=='d') or (shex=='D')):
        return 13
    elif ((shex=='e') or (shex=='E')):
        return 14
    elif ((shex=='f') or (shex=='F')):
        return 15
    else:
        print('idx data trans error!')
        return -1

def write_tile_inst(inst):
    inst_256 = ""

    for i in range(len(inst)):
        if i == 0:
            print(f'write_tile_inst show instruction: {inst[i]} detail')
            continue
        for j in range(len(inst[i])):
            for k in range(len(inst[i][j])):
                for l in range(len(inst[i][j][k])):
                    if l%2:
                        if inst[i][j][k][l] != None:
                            inst_256 = '{}{}'.format(inst_256, inst[i][j][k][l])
    return inst_256

def write_insts(fw, tile_cnt):
    for i in range(tile_cnt):
        inst_256 = write_tile_inst(insts[i])

        for j in range(BUS_WIDTH - len(inst_256)):
            inst_256 = '{}{}'.format(inst_256, '0')

        j=0
        for x in range(int(256/4)):
            dec = int(inst_256[j:j+4], 2)
            fw.write(st.pack('B', dec))
            j += 4

def write_datas(data_locate, fw):
    for i in range(len(data_locate)):
        if i == 0:
            print(f'write_datas show data_locate: {data_locate[i]} detail')
            continue
        for j in range(len(data_locate[i])):
            for k in range(len(data_locate[i][j])):
                for l in range(len(data_locate[i][j][k])):
                    ints = shex_to_int(data_locate[i][j][k][l])
                    fw.write(st.pack('B', ints))

def get_fstin_CHW():
    C = H = W = 0
    global layer_num

    with open(netpath, 'r') as file:
        for line in file:
            if "rand" in line:
                C = int(line.split("(")[1].split(",")[1])
                W = int(line.split("(")[1].split(",")[2])
                H = int(line.split("(")[1].split(",")[3].split(")")[0])

    return C, H, W

def fc_CHW(layer_num):
    global pool_fc_outh_bac
    global pool_fc_outw_bac
    global pool_fc_outc_bac
    H = W = 0

    with open(logpath, 'r') as file:
        for line in file:
            if f"layer_num:{layer_num + 1}" in line:
                break
            if line.startswith(f"layer_num:{layer_num-1}"):
                for line in file:
                    if f"layer_num:{layer_num + 1}" in line:
                        break
                    if "out_features_x" in line:
                        H = W = int(line.split("out_features_x:")[1].split(" ")[0])

    C = pool_fc_outc_bac

    return C, H, W

def pool_CHW(layer_num):
    global pool_fc_outh_bac
    global pool_fc_outw_bac
    global pool_fc_outc_bac
    C = 0
    feature_map_x = feature_map_y = kernel_size_x = kernel_size_y = 0

    with open(logpath, 'r') as file:
        for line in file:
            if f"layer_num:{layer_num + 1}" in line:
                break
            if line.startswith(f"layer_num:{layer_num-1}"):
                for line in file:
                    if f"layer_num:{layer_num + 1}" in line:
                        break
                    if "feature_map_size_x" in line:
                        feature_map_x = int(line.split("feature_map_size_x:")[1].split(" ")[0])
                    if "feature_map_size_y" in line:
                        feature_map_y = int(line.split("feature_map_size_y:")[1].split(" ")[0])
                    if "kernel_size_x" in line:
                        kernel_size_x = int(line.split("kernel_size_x:")[1].split(" ")[0])
                    if "kernel_size_y" in line:
                        kernel_size_y = int(line.split("kernel_size_y:")[1].split(" ")[0])
                    if "out_channels" in line:
                        C = int(line.split("out_channels:")[1].split(" ")[0])

    if C == 0:
        C = pool_fc_outc_bac
    if feature_map_x != 0 and feature_map_y !=0:
        H = align(int(feature_map_x/kernel_size_x), 2)
        W = align(int(feature_map_y/kernel_size_y), 2)
    else:
        H = int(align(pool_fc_outh_bac/kernel_size_x, 2))
        W = int(align(pool_fc_outw_bac/kernel_size_y, 2))

    pool_fc_outc_bac = C
    pool_fc_outh_bac = H
    pool_fc_outw_bac = W

    return C, H, W

def get_out_CHW(layer_num):
    C = H = W = 0
    with open(logpath, 'r') as file:
        for line in file:
            if f"layer_num:{layer_num + 1}" in line:
                break
            if line.startswith(f"layer_num:{layer_num}"):
                if "type:maxpool" in line or "type:avgpool" in line:
                    return pool_CHW(layer_num)
                if "layer type:fc" in line:
                    return fc_CHW(layer_num)
                for line in file:
                    if f"layer_num:{layer_num+1}" in line:
                        break
                    if "out_channels" in line:
                        C = int(line.split("out_channels:")[1].split(" ")[0])
                    if "feature_map_size_x" in line:
                        H = int(line.split("feature_map_size_x:")[1].split(" ")[0])
                        W = int(line.split("feature_map_size_y:")[1].split(" ")[0])

    return C, H, W

def inout_addr(N, C, H, W, div_size, layercnt):
    padding = 0
    global ADDRBLOCK
    word_address = 0
    print("layer", layercnt, "net start addr:", ADDRBLOCK, " bus_addr:", bus_address())

    for c in range(align(C, div_size)):
        for w in range(W):
            for h in range(H):
                for cch in range(div_size):
                    word_address, bus_addr = index_to_address(N, C, H, W, 0, c * div_size + cch, h, w, div_size)
                    word_address += padding 
            padding = align(word_address, BUS_WIDTH) - word_address
        ADDRBLOCK += word_address + padding
    print("layer", layercnt, "net end addr:", ADDRBLOCK, " bus_addr:", bus_address())
    ADDRBLOCK = align(ADDRBLOCK, BUS_WIDTH)

def netinout_addr():
    N = 1
    # C = H = W = 0

    #net的输入层
    C, H, W = get_fstin_CHW()
    print("netinout_addr start C H W:", C, H, W)
    inout_addr(N, C, H, W, div_size, 1)
    #net的输出层
    for i in range(loadpt.layer_cnts):
        C, H, W = get_out_CHW(i+1)
    print(loadpt.layer_cnts, "netinout_addr end C H W:", C, H, W)
    inout_addr(N, C, H, W, div_size, loadpt.layer_cnts)

def otherinout_addr():
    #for循环中间层
    N = 1
    cnt = 0
    global ADDRBLOCK
    for i in range(1, loadpt.layer_cnts-2):
        cnt += 1
        if cnt > 5:
            ADDRBLOCK = CALCULATE
            cnt = 0
        C, H, W = get_out_CHW(i+1)
        inout_addr(N, C, H, W, div_size, i+1)

def get_binsize():
    return 100000000

def binary_addr():
    global ADDRBLOCK
    ADDRBLOCK += get_binsize()

def gen_ddraddr():
    print("dram_capacity:", dram_capacity)
    netinout_addr()
    weight_addr()
    global CALCULATE
    CALCULATE = ADDRBLOCK
    otherinout_addr()
    binary_addr()
    print("run gen_ddraddr successfully")

def prints(cnt, total, msg):
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
        output = f'debug/{cleanlist[i]}'
        if "data_for_fpga" in cleanlist[i]:
            output = f'{outputpath}/{cleanlist[i]}'
        if os.path.exists(output):
            if os.path.isdir(output):
                shutil.rmtree(output)
            else:
                os.remove(output)
            print("%s clean success" % (output))

    if os.path.exists("debug"):
        shutil.rmtree("debug")
    os.mkdir("debug")

    print("clean_ups successfully")

def checkfile(filelist):
    """
    description:
                Check files in working directory
    parameters:
                checklist: List of files to check
    return code:
                None
    """
    for i in range(len(filelist)):
        file = "input/"
        if "imagenet" in filelist[i] or "ResNet" in filelist[i]:
            file = ""
        name = f'{file}{filelist[i]}'
        if not os.path.exists(name) and not os.path.exists(filelist[i]):
            print("%s not found...\nPlease try again!" % name)
            sys.exit(1)
        else:
            print("check %s success" % filelist[i])

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
    start = time.time()
    allstart = start

    parase = option_parse()
    ptpath = parase.pt_name
    netpath = parase.net_name
    confpath = parase.conf_dir
    ptdtpath = parase.ptdata_dir
    bmpdtpath = parase.bmpdata_dir
    outputpath = parase.output_dir

    make_bin = ["Make the bin file needed by fpga...",
                 "gen_fpga('imagenet')"]
    logname = "vggnet.log"
    if "resnet" in netpath.lower():
        logname = "resnet.log"
        make_bin = ["Make file to binary", "address()"]

    logpath = f'{os.getcwd()}/debug/{logname}'

    fileslist = ["config_gen_file.py",  "inout_print.py", "quant_layer.py",  "quantop.py",
                 "im6.bmp",              ptpath,           netpath]
    cleanlist = ["data_for_fpga",       logname,           "debug",
                  confpath,             ptdtpath,          outputpath]
    
    run_step = [["Check the necessary files...",
                 "checkfile(fileslist)"],
                ["Clean the necessary files...",
                 "clean_ups(cleanlist)"],
                ["Run the network model file...", 
                 f'os.system(\'python3 {netpath} > debug/{logname}\')',
                 f'print(\'run {netpath} successfully\')'],
                ["Load pt file and format output...", 
                 "os.system(f'mkdir -p {ptdtpath} {outputpath}')",
                 "loadpt = load_pt(ptpath, confpath, ptdtpath)",
                 "loadpt.layer_cnts = get_cnts()",
                 f'loadpt.net_name = \'{logname.split(".")[0]}\'',
                 "gen_txt(loadpt)"],
                ["Generate input and output from bmp...",
                 f'gen_bmp(\'debug/{logname}\')'],
                ["gen_address for DRAM...",
                 "gen_ddraddr()"],
                 make_bin]

    for i in range(len(run_step)):
        start = time.time()
        prints(i+1, len(run_step), run_step[i][0])
        for j in range(1, len(run_step[i])):
            s = run_step[i][j]
            r = compile(s, "<string>", "exec")
            exec(r)
        end = time.time()
        schedule(i+1, start, end)

    i = 1
    type = ""
    inst_00 = [
        ["Inst_id_00", get_Inst_id_00()],
        ["act_tile_hor", get_act_tile_hor(i)],
        ["act_tile_ver", get_act_tile_ver(i)],
        ["act_tile_chl", get_act_tile_chl(i)],
        ["act_tile_str", get_act_tile_str(i)],
        ["act_tile_sub_chl", get_act_tile_sub_chl(i)],
        ["sub_tile_hor", get_sub_tile_hor(i)],
        ["sub_tile_ver", get_sub_tile_ver(i)],
        ["out_chl_num", get_out_chl_num(i)],
        ["Mini_tile_hor", get_Mini_tile_hor(i)],
        ["Mini_tile_ver", get_Mini_tile_ver(i)],
        ["in_chl_num", get_in_chl_num(i)],
        ["Out_mini_tile_hor", get_Out_mini_tile_hor(i)],
        ["Out_mini_tile_ver", get_Out_mini_tile_ver(i)],
        ["Out_tile_hor", get_Out_tile_hor(i)],
        ["Out_tile_ver", get_Out_tile_ver(i)],
        ["Out_tile_chl", get_Out_tile_chl(i)],
        ["weight_total_size", get_weight_total_size(i)],
        ["Weight_sub_size", get_Weight_sub_size(i)],
        ["weight_ram_output_chl_num", get_weight_ram_output_chl_num(i)],
        ["conv_type", get_conv_type(i)],
        ["Chiplet_mode  ", get_Chiplet_mode(i)],
        ["Chiplet_tile_size", get_Chiplet_tile_size(i)],
        ["Kernel_num", get_Kernel_num(i)],
        ["Kernel_width", get_Kernel_width(i)],
        ["Kernel_height", get_Kernel_height(i)],
        ["Kernel_str", get_Kernel_str(i)],
        ["deconv_hor_str", get_deconv_hor_str(i)],
        ["deconv_ver_str", get_deconv_ver_str(i)],
        ["Dilation_rate", get_Dilation_rate(i)],
        ["pooling_carry_sel", get_pooling_carry_sel(i)],
        ["pool_mode", get_pool_mode(i)],
        ["Pooling_quan_code_in", get_Pooling_quan_code_in(i)],
        ["pool_total_size", get_pool_total_size(i)],
        ["pool_size", get_pool_size(i)],
        ["Pooling_oprands", get_Pooling_oprands(i)],
        ["Pooling_quan_code_out", get_Pooling_quan_code_out(i)],
        ["scaling_mode", get_scaling_mode(i)],
        ["fc_mode_en", get_fc_mode_en(i)],
    ]

    inst_01 = [
        ["Inst_id_01", get_Inst_id_01()],
        ["Tile_mode", get_Tile_mode()],
        ["tile_num", get_tile_num(i)],
        ["Padding_mode", get_Padding_mode(i)],
        ["Padding_num", get_Padding_num(i)],
        ["Repeat_send_num", get_Repeat_send_num(i)],
        ["act_inst_bypass", get_act_inst_bypass(i)],
        ["weight_inst_bypass", get_weight_inst_bypass(i)],
        ["act_chl_one_inst_real", get_act_chl_one_inst_real(i)],
        ["act_sub_ch", get_act_sub_ch(i)],
        ["act_chl_one_inst", get_act_chl_one_inst(i)],
        ["inst_invalid", get_inst_invalid(i)],
        ["mini_ver_num", get_mini_ver_num(i)],
        ["mini_hor_num", get_mini_hor_num(i)],
    ]

    inst_11 = [
        ["Inst_id_11", get_Inst_id_11()],
        ["Run_mode", get_Run_mode(i)],
        ["weight_addr", get_weight_addr(i)],
        ["weight_output_chl", get_weight_output_chl(i)],
        ["weight_updata_n", get_weight_updata_n(i)],
        ["LLC_w_ping_pong", get_LLC_w_ping_pong(i)],
        ["act_addr", get_act_addr(i)],
        ["act_str_chl", get_act_str_chl(i)],
        ["act_str_line", get_act_str_line(i)],
        ["act_updata_n", get_act_updata_n(i)],
        ["LLC_a_ping_pong", get_LLC_a_ping_pong(i)],
        ["Out_tile_start_addr", get_Out_tile_start_addr(i)],
        ["Out_tile_stride", get_Out_tile_stride(i)],
        ["Out_feature_map_ver", get_Out_feature_map_ver(i)],
        ["act_addr_element", get_act_addr_element(i)],
    ]

    insts = ["", [[], [], []]]
    j = tile_cnt = 0
    del insts[0]

    datadir = f'{outputpath}/datas/'
    instdir = f'{outputpath}/insts/'
    os.mkdir(instdir)
    os.mkdir(datadir)
    #insts是一层的inst(包含n个tile指令)insts[i]均由inst_00 inst_01 inst_11组成的tile指令
    pool_num = get_layer_num("pool")
    for i in range(1, loadpt.layer_cnts+1):
        type = find_type(i, "layer type:")
        nnbaton_X1, nnbaton_Y1, nnbaton_K1, nnbaton_X2, nnbaton_Y2, \
        nnbaton_K2, nnbaton_Kp, nnbaton_Yp, nnbaton_Kc, nnbaton_Yc, \
        nnbaton_Xc, nnbaton_C1, nnbaton_C0, nnbaton_X0, nnbaton_Y0 \
            = get_nnbaton(i, pool_num)
        tile_cnt = nnbaton_X2 * nnbaton_Y2 * nnbaton_K2
        get_layer_inst(i, tile_cnt)
        inst_name = f'{instdir}/layer.{i}.inst.bin'
        data_name = f'{datadir}/layer.{i}.data.bin'
        with open(inst_name, 'wb') as finst:
            write_insts(finst, tile_cnt)
            if i not in pool_num:
                with open(data_name, 'wb') as fdata:
                    write_datas(datas_locate[j], fdata)
                j += 1
        insts = ["", [[], [], []]]
        del insts[0]

    allend = time.time()
    print('Make data successfully! It costs %.2f Seconds'%(allend - allstart))
