import os
import numpy as np
import struct as st
from script.run_steps import prints
#from script.insts_fun import find_type, find_count

div_size = 8
dram_base = 0
ADDRBLOCK = 0
CALCULATE = 0
chiplet_num = 0
ALIGN_WIDTH = 64

datas_locate = ["", []]
BUS_WIDTH = int(256 / 8)
dram_capacity = 1 << 30
layers_act_addr = ["", []]
layers_wet_addr = ["", []]
del layers_act_addr[0]
del layers_wet_addr[0]

netpath = 0
layer_cnts = 0
fmain = logpath = 0

def send_calcuaddrvar(fw, log, net, cnt):
    global fmain
    global logpath
    global netpath 
    global layer_cnts

    fmain = fw
    logpath = log
    netpath = net
    layer_cnts = cnt

def align(address, factor):
    if address % factor == 0:
        return address
    else:
        return (address // factor + 1) * factor

def find_count(layer_num, find_name):
    '''
    获取相应的数据
    参数： layer_num：哪一层的数据；  find_name：获取标志(日志中数据的名称加：)
    '''
    with open(logpath, 'r') as file:
        for line in file:
            if line.startswith(f"layer_num:{layer_num + 1} "):
                break
            if line.startswith(f"layer_num:{layer_num} "):
                for line in file:
                    if line.startswith(f"layer_num:{layer_num + 1} "):
                        break
                    if find_name in line:
                        return (int(line.split(find_name)[1].split(" ")[0]))

def find_type(layer_num, find_name):
    '''
    获取相应的层的类型
    参数： layer_num：哪一层的类型；  find_name：获取标志(日志中数据的名称加：)
    '''
    with open(logpath, 'r') as file:
        for line in file:
            if line.startswith(f"layer_num:{layer_num} "):
                return (line.split(find_name)[1].split(" ")[0])

def pt_skip(name):
    if "alpha" in name or "num_batches" in name:
        return 1

def test_chip_relocate84(layer_locate, outchl, C):
    fmain.write(f'test_relocate...Start Rearrange weight...len(layer_locate):{len(layer_locate)}\n')
    weight_width = align(outchl, BUS_WIDTH)
    end = 0

    for x in range(int(C/2)):  # outchannel/2组bn+weight
        bn_locate = layer_locate[end + BUS_WIDTH * x:end + (x + 2) * BUS_WIDTH]
        for i in range(len(bn_locate)):
            if bn_locate[i] != '00':
                prints(f'test_chip_relocate84 bnlocate[{i}] failed value:{bn_locate[i]}')
                return
        # prints(f'test_chip_relocate84 {BUS_WIDTH*2} bn_locate:{len(bn_locate)} layer_locate[{end+BUS_WIDTH*x}:{end+(x+2)*BUS_WIDTH-1}]={bn_locate}')
        # weight_locate = layer_locate[end + (x + 2) * BUS_WIDTH:end + weight_width*2 + (x + 2) * BUS_WIDTH]
        end += weight_width * 2 + BUS_WIDTH

    fmain.write(f'test_relocate...Endof Rearrange weight...len(layer_locate):{len(layer_locate)}\n')
    fmain.write(f'test_chip_relocate84 fc weight data success\n')

def test_relocate(layer_locate, outchl, C, H, W):
    fmain.write(f'test_relocate...Start Rearrange weight...len(layer_locate):{len(layer_locate)}\n')
    weight_width = align(C * H * W, BUS_WIDTH)
    if H == 1 and W == 1:
        weight_width = align(outchl, BUS_WIDTH)
        outchl = C
    end = 0

    for x in range(outchl):  # outchannel组bn+weight
        bn_locate = layer_locate[end + BUS_WIDTH * x:end + (x + 1) * BUS_WIDTH]
        if H == 1 and W == 1:
            for i in range(len(bn_locate)):
                if bn_locate[i] != '00':
                    prints(f'test_relocatetest bnlocate[{i}] failed value:{bn_locate[i]}')
                    return
        # fmain.write(f'test_relocate {BUS_WIDTH} bn_locate:{len(bn_locate)} layer_locate[{end+BUS_WIDTH*x}:{end+(x+1)*BUS_WIDTH-1}]={bn_locate}\n')
        # weight_locate = layer_locate[end + (x + 1) * BUS_WIDTH:end + weight_width + (x + 1) * BUS_WIDTH]
        # fmain.write(f'test_relocate {weight_width} weight_locate:{len(weight_locate)} layer_locate[{end+(x+1)*BUS_WIDTH}:{end+weight_width+(x+1)*BUS_WIDTH-1}]={weight_locate}\n')
        end += weight_width

    fmain.write(f'test_relocate...Endof Rearrange weight...len(Rearr_locate):{len(layer_locate)}\n')
    if H == 1 and W == 1:
        fmain.write(f'test_relocate fc weight data success\n')

def chip_relocate84(layer_locate, outchl, C):
    fmain.write(f'chip_relocate...Start Rearrange weight...len(layer_locate):{len(layer_locate)}\n')
    weight_width = align(outchl, BUS_WIDTH)
    Rearrange_locate = []
    end = 0

    for x in range(int(C/2)):  # outchannel/2组bn+weight
        bn_locate = layer_locate[end + BUS_WIDTH * x:end + (x + 2) * BUS_WIDTH]
        # fmain.write(f'chip_relocate84 {BUS_WIDTH*2} bn_locate:{len(bn_locate)} layer_locate[{end+BUS_WIDTH*x}:{end+(x+2)*BUS_WIDTH-1}]={bn_locate}\n')
        weight_locate = layer_locate[end + (x + 2) * BUS_WIDTH:end + weight_width * 2 + (x + 2) * BUS_WIDTH]
        # fmain.write(f'chip_relocate84 {weight_width*2} weight_locate:{len(weight_locate)} layer_locate[{end+(x+2)*BUS_WIDTH}:{end+weight_width*2+(x+2)*BUS_WIDTH-1}]\n')
        index = int(len(weight_locate) / (2 ** chiplet_num))
        weight_relocate = []  # 重排weight数据
        for i in range(index):
            for j in range(2 ** chiplet_num):
                weight_relocate.append(weight_locate[i + index * j])
        Rearrange_locate += bn_locate + weight_relocate
        end += weight_width * 2 + BUS_WIDTH

    fmain.write(f'chip_relocate...Endof Rearrange weight...len(Rearr_locate):{len(Rearrange_locate)}\n')
    return Rearrange_locate

def relocate(layer_locate, outchl, C, H, W):
    fmain.write(f'relocate...Start Rearrange weight...len(layer_locate):{len(layer_locate)}\n')
    weight_width = align(C * H * W, BUS_WIDTH)
    if H == 1 and W == 1:
        weight_width = align(outchl, BUS_WIDTH)
        outchl = C
    Rearrange_locate = []
    end = 0

    for x in range(outchl):  # outchannel组bn+weight
        bn_locate = layer_locate[end + BUS_WIDTH * x:end + (x + 1) * BUS_WIDTH]
        # fmain.write(f'relocate {BUS_WIDTH} bn_locate:{len(bn_locate)} layer_locate[{end+BUS_WIDTH*x}:{end+(x+1)*BUS_WIDTH-1}]={bn_locate}\n')
        weight_locate = layer_locate[end + (x + 1) * BUS_WIDTH:end + weight_width + (x + 1) * BUS_WIDTH]
        # fmain.write(f'relocate {weight_width} weight_locate:{len(weight_locate)} layer_locate[{end+(x+1)*BUS_WIDTH}:{end+weight_width+(x+1)*BUS_WIDTH-1}]\n')
        index = int(len(weight_locate) / (2 ** chiplet_num))
        weight_relocate = []  # 重排weight数据
        for i in range(index):
            for j in range(2 ** chiplet_num):
                weight_relocate.append(weight_locate[i + index * j])
        Rearrange_locate += bn_locate + weight_relocate
        end += weight_width

    fmain.write(f'relocate...Endof Rearrange weight...len(Rearr_locate):{len(Rearrange_locate)}\n')
    return Rearrange_locate

def fc_relocate88(layer_locate, W, outchl):
    fmain.write(f'fc_relocate88...Start Rearrange weight...len(layer_locate):{len(layer_locate)}\n')
    Rearrange_locate = []
    end = 0

    for x in range(outchl):  # outchannel组bn+weight
        # 每一组outchannel的数据：bn数据+weight数据
        bn_locate = layer_locate[end + BUS_WIDTH * x:end + (x + 1) * BUS_WIDTH]
        fmain.write(
            f'bn_k&_blocate:{len(bn_locate)} layer_locate[{end+BUS_WIDTH*x}:{end+(x+1)*BUS_WIDTH-1}]={bn_locate}\n')
        weight_locate = layer_locate[end + (x + 1) * BUS_WIDTH:end + W + (x + 1) * BUS_WIDTH]
        fmain.write(
            f'weight_locate:{len(weight_locate)} layer_locate[{end+(x+1)*BUS_WIDTH}:{end+W+(x+1)*BUS_WIDTH-1}]={weight_locate}\n')

        weight_relocate = []  # 重排weight数据
        for i in range(int(len(weight_locate)/(4 * 4))):
            for j in range(4 * 4):
                if j % 2 == 0:
                    weight_relocate.append(weight_locate[j + 4 * 4 * i])
            for j in range(4 * 4):
                if j % 2:
                    weight_relocate.append(weight_locate[j + 4 * 4 * i])
        Rearrange_locate += bn_locate + weight_relocate
        end += W

    fmain.write(f'fc_relocate88...Endof Rearrange weight...len(Rearr_locate):{len(Rearrange_locate)}\n')
    return Rearrange_locate

def fc_relocate84(layer_locate, W, outchl):
    fmain.write(f'fc_relocate...Start Rearrange weight...len(layer_locate):{len(layer_locate)}\n')
    Rearrange_locate = []
    scale = 4 * 4 * 2
    end = 0

    for x in range(int(outchl/2)):  # outchannel/2组bn+weight
        # 每一组outchannel的数据：bn数据+weight数据
        bn_locate1 = layer_locate[end + BUS_WIDTH * x:end + (x + 1) * BUS_WIDTH]
        weight_locate1 = layer_locate[end + (x + 1) * BUS_WIDTH:end + W + (x + 1) * BUS_WIDTH]
        # fmain.write(f'bn_k&_blocate1:{len(bn_locate1)} layer_locate[{end+BUS_WIDTH*x}:{end+(x+1)*BUS_WIDTH-1}]\n')
        # fmain.write(f'weight_locate1:{len(weight_locate1)} layer_locate[{end+(x+1)*BUS_WIDTH}:{end+W+(x+1)*BUS_WIDTH-1}]\n')
        end += W
        bn_locate2 = layer_locate[end + (x + 1) * BUS_WIDTH:end + (x + 2) * BUS_WIDTH]
        weight_locate2 = layer_locate[end + (x + 2) * BUS_WIDTH:end + W + (x + 2) * BUS_WIDTH]
        # fmain.write(f'bn_k&_blocate2:{len(bn_locate2)} layer_locate[{end+(x+1)*BUS_WIDTH}:{end+(x+2)*BUS_WIDTH-1}]\n')
        # fmain.write(f'weight_locate2:{len(weight_locate2)} layer_locate[{end+(x+2)*BUS_WIDTH}:{end+W+(x+2)*BUS_WIDTH-1}]\n')

        weight_relocate = []  # 重排fc8*4的weight数据
        for i in range(int(len(weight_locate1)/scale)):
            for j in range(4 * 4 * 2):
                if j % 2 == 0:
                    weight_relocate.append(weight_locate1[j + scale * i])
                    weight_relocate.append(weight_locate2[j + scale * i])
                    # prints(f'weight_relocate1[{j + scale * i}]:{weight_locate1[j + scale * i]}')
                    # prints(f'weight_relocate2[{j + scale * i}]:{weight_locate2[j + scale * i]}')
            for j in range(4 * 4 * 2):
                if j % 2:
                    weight_relocate.append(weight_locate1[j + scale * i])
                    weight_relocate.append(weight_locate2[j + scale * i])
                    # prints(f'weight_relocate1[{j + scale * i}]:{weight_locate1[j + scale * i]}')
                    # prints(f'weight_relocate2[{j + scale * i}]:{weight_locate2[j + scale * i]}')
        Rearrange_locate += bn_locate1 + bn_locate2 + weight_relocate
        end += W + BUS_WIDTH  # 为什么加BUS_WIDTH没想通，但加上是对的

    fmain.write(f'fc_relocate...Endof Rearrange weight...len(Rearr_locate):{len(Rearrange_locate)}\n')
    return Rearrange_locate

def add_padding(layer_locate):
    padding = align(len(layer_locate), BUS_WIDTH) - len(layer_locate)
    for i in range(padding):  # data补齐256bits
        layer_locate.append("00")

    return layer_locate

def get_layer_num(layer_name):
    pool_num = []
    downsample = []
    fc_flag = False
    if "fc" in layer_name:
        fc_flag = True

    with open(logpath, 'r') as file:
        for line in file:
            if fc_flag:
                if line.startswith("layer_num:") and layer_name in line:
                    return (int(line.split(" ")[0].split(":")[1]))
            elif layer_name in line:
                if line.startswith("layer_num:"):
                    poolname = line.split(":", 5)[2].split(" ")[0]
                    if "pool" not in poolname:
                        continue
                    pool_num.append(int(line.split(" ")[0].split(":")[1]))
            elif "downsample" in line and fc_flag == False:
                downsample.append(int(line.split(" ")[0].split(":")[1]))
    if fc_flag:
        return pool_num
    else:
        return pool_num, downsample

def bus_address():
    return int(ADDRBLOCK // (BUS_WIDTH / div_size))

def index_to_address(N, C, H, W, n, c, h, w, bit_mode):
    assert ((bit_mode == 4) | (bit_mode == 8) | (bit_mode == 16)), "bit_mode should be 4, 8 or 16."

    sub_channel = ALIGN_WIDTH / bit_mode
    bus_width_byte = BUS_WIDTH / bit_mode

    C = C // sub_channel * sub_channel
    C0 = sub_channel

    c0 = np.mod(c, sub_channel)
    c_ = c // C0

    word_address = int(c0 + w * C0 + h * W * C0 + c_ * H * W * C0)
    bus_address = int(word_address // bus_width_byte)

    return word_address, bus_address

def HexProcess(decimal):
    if decimal <= 15 and decimal >= 0:
        hex_num = '0' + hex(decimal & 0xFF)[-1:]
    else:
        hex_num = hex(decimal & 0xFF)[-2:]
    return hex_num

def weight_addr(name_list, data_list, active_bit, weight_bit):
    """
    description:
                Load pt file and format output
    parameters:
                loadpt: The Class of load_pt
    return code:
                None
    """
    k = scale = 0
    fc_flag = False
    word_address = 0
    weight_bn_k = []
    weight_bn_b = []
    weight_data = []
    global ADDRBLOCK
    start = ADDRBLOCK
    layer_locate = []
    N = C = H = W = 0
    global datas_locate
    global layers_wet_addr
    layer_cnt = fc_cnt = 0
    running_mean = []
    running_var = []
    bn_weight = []
    bn_bias = []

    poollist,downsample = get_layer_num("pool")
    fmain.write(f'weight_addr start:{ADDRBLOCK} bus_addr:{bus_address()} chiplet:{2**chiplet_num}\n')
    for i in range(len(name_list)):
        name = name_list[i]
        data = data_list[i]
        if pt_skip(name):
            continue
        # 从pt中获取running_mean running_var bn_bias scale和weight
        elif "scale" in name:
            scale = data.tolist()
        elif "weight" in name:
            if "classifier" in name or "fc" in name:
                fc_flag = True
            elif "bn" in name or "downsample.1" in name:
                bn_weight = data
                continue
            weight_data = data
            layer_cnt += 1
            if layer_cnt in poollist:
                layer_cnt += 1
        elif "bias" in name:
            bn_bias = data
        elif "running_mean" in name:
            running_mean = data
        elif "running_var" in name:
            running_var = data
        # 通过running_mean running_var bn_bias bn_weight计算bn_k和bn_b
        if len(running_mean) and len(running_var) and len(bn_bias) and len(bn_weight):
            for i in range(len(bn_bias)):
                bn_k = bn_weight[i] / (running_var[i].sqrt() + 1e-6)
                bn_b = -bn_weight[i] * running_mean[i] / \
                       (running_var[i].sqrt() + 1e-6) + bn_bias[i]
                weight_bn_k.append(bn_k.half().float())
                weight_bn_b.append(bn_b.half().float())
            running_mean = []
            running_var = []
            bn_weight = []
            bn_bias = []
        # activation地址排布和weight数据保存
        if scale and (len(weight_bn_k) and len(weight_bn_b) and len(weight_data) or fc_flag):
            if "classifier" in name or "fc" in name:
                fc_cnt += 1
                layer_cnt = get_layer_num(f'fc{fc_cnt}')
            layer_name = f'layer{str(layer_cnt)}'
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
                if len(weight_bn_k) and len(weight_bn_b):  # pt中的bnkb存入layer_locate
                    hexdata_k = '%X' % st.unpack('I', st.pack('f', weight_bn_k[n]))[0]
                    hexdata_b = '%X' % st.unpack('I', st.pack('f', weight_bn_b[n]))[0]
                    for x in range(len(str(hexdata_k))):
                        if (x % 2) == 0:
                            layer_locate.append(hexdata_k[x] + hexdata_k[x + 1])
                    for y in range(len(str(hexdata_b))):
                        if (y % 2) == 0:
                            layer_locate.append(hexdata_b[y] + hexdata_b[y + 1])
                layer_locate = add_padding(layer_locate)
                for c in range(C):
                    for h in range(H):
                        if dim_lens == 2:
                            for i in range(BUS_WIDTH):
                                layer_locate.append("00")
                        for w in range(W):
                            if dim_lens == 4:
                                word_addr = w + h * W + c * H * W
                                rounds = round(weight_data[n][c][h][w].tolist() / scale)
                            elif dim_lens == 2:
                                word_addr = w + h * W
                                rounds = round(weight_data[h][w].tolist() / scale)
                            else:
                                prints(f'Unknown weight:{name} length:{dim_lens} dim:{dim_list}')
                                continue
                            hexdata_w = HexProcess(rounds)
                            layer_locate.append(hexdata_w)
                word_address += BUS_WIDTH  # bn预留(256bit)
                word_address += align(word_addr, BUS_WIDTH)
                layer_locate = add_padding(layer_locate)
            end = ADDRBLOCK + word_address
            layers_wet_addr.append([f'{layer_name} weight start and end:', [start, end]])
            start = end
            if active_bit == 16 or active_bit == 4:  # weight 16*8 4*4不重排
                pass
            elif fc_flag == False:  # active_bit=8 conv weight重排
                if chiplet_num:
                    fmain.write(f'{layer_name} relocate conv\'s weight...\n')
                    layer_locate = relocate(layer_locate, N, C, H, W)
                    # test_relocate(layer_locate, N, C, H, W)
            else:  # fc weight重排
                if weight_bit == 8:  # fc weight 8*8重排
                    layer_locate = fc_relocate88(layer_locate, W, H)
                elif weight_bit == 4:  # fc weight 8*4重排
                    layer_locate = fc_relocate84(layer_locate, W, H)
                    test_chip_relocate84(layer_locate, W, H)
                if chiplet_num:
                    fmain.write(f'{layer_name} relocate fc\'s weight...\n')
                    if weight_bit == 8:  # chiplet mode fc weight 8*8重排
                        layer_locate = relocate(layer_locate, W, H, 1, 1)
                        test_relocate(layer_locate, W, H, 1, 1)
                    elif weight_bit == 4:  # chiplet mode fc weight 8*4重排
                        layer_locate = chip_relocate84(layer_locate, W, H)
                        test_chip_relocate84(layer_locate, W, H)
            fmain.write("%7s bn_k+bn_b+conv_weight data save data success\n" % layer_name)
            datas_locate[k] = [f'layer {str(layer_cnt)} data', [layer_locate]]
            datas_locate.append(datas_locate[k])
            layer_locate = []
            weight_bn_k = []
            weight_bn_b = []
            weight_data = []
            fc_flag = 0
            scale = 0
            k += 1
            if layer_cnts <= k:
                break

    ADDRBLOCK += word_address
    return poollist, downsample

def gen_ddraddr(name_list, data_list, act_bit, wet_bit):
    fmain.write(f'dram_capacity:{dram_capacity}\n')
    netinout_addr(netpath)
    poollist, downsample = weight_addr(name_list, data_list, act_bit, wet_bit)
    if layer_cnts > 1:
        global CALCULATE
        CALCULATE = ADDRBLOCK
        otherinout_addr()
    prints("run gen_ddraddr successfully")
    return layers_act_addr,layers_wet_addr,datas_locate,poollist,downsample

def get_layer_addr(layernum, addr_list):
    for i in range(1, len(addr_list) + 1):
        layer_addr = addr_list[i]
        if f'layer{layernum} ' in layer_addr[0]:
            del addr_list[i]
            return layer_addr

def get_fstin_CHW(netpath):
    C = H = W = 0
    global layer_num

    with open(netpath, 'r') as file:
        for line in file:
            if "rand" in line:
                C = int(line.split("(")[1].split(",")[1])
                W = int(line.split("(")[1].split(",")[2])
                H = int(line.split("(")[1].split(",")[3].split(")")[0])

    return C, H, W

def get_out_fc_CHW(layer_num):
    fc_C = find_count(layer_num, "out_features_y:")
    fc_H = fc_W = find_count(layer_num, "out_features_x:")

    return fc_C, fc_H, fc_W

def get_out_pool_CHW(layer_num):
    C = H = W = 0
    layer_type = find_type(layer_num, "layer type:")
    if "conv" in layer_type:
        C = find_count(layer_num, "in_channels:")
        H = find_count(layer_num, "feature_map_size_y:")
        W = find_count(layer_num, "feature_map_size_x:")
    elif "AdaptAvgpool" in layer_type:
        C = find_count(layer_num - 1, "out_channels:")
        H = find_count(layer_num, "output_size_x:")
        W = find_count(layer_num, "output_size_y:")
    elif "fc" in layer_type:
        C, H, W = get_out_pool_CHW(layer_num - 1)
    elif "pool" in layer_type:
        prev_layer_type = find_type(layer_num - 1, "layer type:")
        if "pool" in prev_layer_type:
            C, pool_H, pool_W = get_out_pool_CHW(layer_num - 1)
            pool_kernel_size_x = find_count(layer_num, "kernel_size_x:")
            pool_kernel_size_y = find_count(layer_num, "kernel_size_y:")
            H = int(pool_H / pool_kernel_size_y)
            W = int(pool_W / pool_kernel_size_x)
        elif "conv" in prev_layer_type:
            feature_map_x = find_count(layer_num - 1, "feature_map_size_x:")
            feature_map_y = find_count(layer_num - 1, "feature_map_size_y:")
            kernel_size_x = find_count(layer_num, "kernel_size_x:")
            kernel_size_y = find_count(layer_num, "kernel_size_y:")
            C = find_count(layer_num - 1, "out_channels:")
            H = align(int(feature_map_y / kernel_size_y), 2)
            W = align(int(feature_map_x / kernel_size_x), 2)

    return C, H, W

def get_out_CHW(layer_num):
    if layer_num <= 0 or layer_num > layer_cnts:
        print(f'get_out_CHW run error. {layer_num} is illegal(<= 0 or >{layer_cnts})')
        return 1

    layer_type = find_type(layer_num, "layer type:")
    if "conv" in layer_type:
        conv_out_C = find_count(layer_num, "out_channels:")
        conv_out_H = find_count(layer_num, "feature_map_size_y:")
        conv_out_W = find_count(layer_num, "feature_map_size_x:")
        return conv_out_C, conv_out_H, conv_out_W
    elif "pool" in layer_type or "AdaptAvgool" in layer_type:
        return get_out_pool_CHW(layer_num)
    elif "fc" in layer_type:
        return get_out_fc_CHW(layer_num)

def inout_addr(N, C, H, W, div_size, layercnt, str):
    global ADDRBLOCK
    word_address = 0
    start = ADDRBLOCK
    global layers_act_addr

    for c in range(int(align(C, div_size) / div_size)):
        for w in range(W):
            for h in range(H):
                for cch in range(div_size):
                    word_addr, bus_addr = index_to_address(N, C, H, W, 0, c * div_size + cch, h, w, div_size)
            word_address = align(word_addr, BUS_WIDTH)
            word_addr = 0

    ADDRBLOCK += word_address
    fmain.write(f'%7s %6s start and end:{[start, ADDRBLOCK]}\n' % (f'layer{layercnt}', str))
    layers_act_addr.append([f'layer{layercnt} {str} start and end:', [start, ADDRBLOCK]])

def netinout_addr(netpath):
    N = 1

    # net的输入层
    C, H, W = get_fstin_CHW(netpath)
    fmain.write("netinputs_addr start C:%3s   H:%3s   W:%3s\n" % (C, H, W))
    inout_addr(N, C, H, W, div_size, 1, "input")
    # net的输出层
    if layer_cnts > 1:
        C, H, W = get_out_CHW(layer_cnts)
        fmain.write("netoutput_addr endof C:%3s   H:%3s   W:%3s\n" % (C, H, W))

    inout_addr(N, C, H, W, div_size, layer_cnts, "output")

def otherinout_addr():
    # for循环中间层
    N = 1
    cnt = 0
    global ADDRBLOCK

    for i in range(1, layer_cnts):
        cnt += 1
        if cnt > 5:
            ADDRBLOCK = CALCULATE
            cnt = 0
        C, H, W = get_out_CHW(i)
        layername = f'layer{i} out_addr'
        fmain.write("%16s C:%3s   H:%3s   W:%3s\n" % (layername, C, H, W))
        inout_addr(N, C, H, W, div_size, i, "output")

def binary_addr(instdir, datadir, output):
    binsize = 0
    binsize = get_binsize(os.listdir(instdir), binsize, output)
    binsize = get_binsize(os.listdir(datadir), binsize, output)

    fmain.write(f'all binary size:{binsize} bytes\n')

    global ADDRBLOCK
    ADDRBLOCK += binsize
    fmain.write(f'DDR ADDRBLOCK size :{ADDRBLOCK} bytes\n')
    prints("run binary_addr successfully")

def get_binsize(file, binsize, output):
    dir = "inst"
    if "data" in file[0]:
        dir = "data"

    j = 1
    for i in range(layer_cnts):
        name = f'layer.{j}.{dir}.bin'
        path = f'{output}/{dir}s/{name}'
        if os.path.exists(path):
            size = os.path.getsize(path)
            binsize += size
            fmain.write(f'{path}\'s size={size} bytes binsize:{binsize} bytes\n')
        j += 1

    return binsize
