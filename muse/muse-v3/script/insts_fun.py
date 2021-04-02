import struct as st
from script.calcu_addr import get_out_CHW, index_to_address, get_fstin_CHW

# nnbaton参数
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
nnbaton_divcase = 0

fmain = 0
logpath = 0
netpath = 0
act_bit = 0
weight_bit = 0

def send_instsfunvar(fw, path):
    global fmain
    global logpath

    fmain = fw
    logpath = path

def send_instsbit(actbit, wetbit, net):
    global act_bit
    global netpath
    global weight_bit

    netpath = net
    act_bit = actbit
    weight_bit = wetbit

def align(address, factor):
    if address % factor == 0:
        return address
    else:
        return (address // factor + 1) * factor

def return_value(countnum):
    '''
    获取无效的参数
    参数： countnum:返回多少位无效的数字
    '''
    return_value = []
    for i in range(countnum):
        return_value.append('0')
    return ''.join(return_value)

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

#################  inst_00  #################
'''
获取当前指令的ID
'''
def get_Inst_id_00():
    return '{:02b}'.format(00)

'''
获取act  tile的水平(hor)和垂直(var)方向的大小，通过output推算input
'''
def get_act_tile_hor_var(layer_num, tile_num):
    out_H = find_count(layer_num, "feature_map_size_y:")
    out_W = find_count(layer_num, "feature_map_size_x:")
    stride = find_count(layer_num, "stride_x:")
    kernel_size = find_count(layer_num, "kernel_size_x:")
    padding_num = find_count(layer_num, "padding_num:")
    tile_case_hor = tile_num // nnbaton_X2 % (nnbaton_Yp * nnbaton_Y2)  # 当前tile是哪一行
    tile_case_ver = tile_num % nnbaton_X2  # 当前tile是哪一列

    out_tail_hor = out_W / nnbaton_X2
    out_tail_ver = out_H / nnbaton_Yp / nnbaton_Y2

    act_tail_hor = out_tail_hor * stride - 1 + kernel_size
    act_tail_ver = out_tail_ver * stride - 1 + kernel_size

    # 通过output推算出来的input带上了padding，因此需要将padding减掉
    # if判断第一行或者是最后一行
    if tile_case_hor == 0 or tile_case_hor == nnbaton_Yp * nnbaton_Y2 - 1:
        # if判断是否为边角上的四个tile
        if tile_case_ver == 0 or tile_case_ver == nnbaton_X2 - 1:
            act_tail_hor = act_tail_hor - padding_num
            act_tail_ver = act_tail_ver - padding_num
        # elif里面判断第一行或者最后一行中间部分
        elif tile_case_ver > 0 and tile_case_ver < nnbaton_X2 - 1:
            act_tail_ver = act_tail_ver - padding_num
    # elif判断第一列或者是最后一列
    elif tile_case_ver == nnbaton_X2 - 1 or tile_case_ver == 0:
        # if判断是否为中间列的部分
        if tile_case_hor > 0 and tile_case_hor < nnbaton_Yp * nnbaton_Y2 - 1:
            act_tail_hor = act_tail_hor - padding_num

    return act_tail_hor, act_tail_ver

'''
获取act  tile的水平方向的大小
'''
def get_act_tile_hor(layer_num, tile_num):
    if "conv" in find_type(layer_num, "layer type:"):
        act_tile_hor = int(get_act_tile_hor_var(layer_num, tile_num)[0])
        return '{:07b}'.format(act_tile_hor - 1)
    else:
        return return_value(7)

'''
获取act  tile的垂直方向的大小
'''
def get_act_tile_ver(layer_num, tile_num):
    if "conv" in find_type(layer_num, "layer type:"):
        act_tile_ver = int(get_act_tile_hor_var(layer_num, tile_num)[1])
        return '{:07b}'.format(act_tile_ver - 1)
    else:
        return return_value(7)

'''
获取act tile在input channel方向上有多少个stride大小的数据块
'''
def get_act_tile_chl(layer_num):
    # 8bit模式
    input_channel = find_count(layer_num, "in_channels:")
    if input_channel != None:
        return '{:012b}'.format(int(align(input_channel, 16) / 16))
    else:
        return return_value(12)

    # 16bit模式
    input_channel = find_count(layer_num, "in_channels:")
    if input_channel != None:
        return '{:012b}'.format(int(align(input_channel, 8) / 8))
    else:
        return return_value(12)

    # 4bit模式
    input_channel = find_count(layer_num, "in_channels:")
    if input_channel != None:
        return '{:012b}'.format(int(align(input_channel, 16) / 16))
    else:
        return return_value(12)

'''
获取act tile的多少bit为一组，第一层为64bit，其它层为128bit
'''
def get_act_tile_str(layer_num):
    if layer_num == 1:
        return '{:01b}'.format(0b1)
    else:
        return '{:01b}'.format(0b0)

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
def get_act_tile_sub_chl(layer_num):
    # if "conv" in find_type(layer_num, "layer type:"):
    #     act_subtail = get_act_subtile_hor_ver(layer_num)
    #     return '{:09b}'.format(act_subtail[0] * act_subtail[1])
    # else:
    #     return return_value(9)
    return '{:012b}'.format(int(get_act_tile_chl(layer_num)))

'''
获取sub_core处理tile的水平方向上大小
'''
def get_sub_tile_hor(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        sub_tile_hor = get_act_subtile_hor_ver(layer_num)[0]
        return '{:07b}'.format(sub_tile_hor - 1)
    else:
        return return_value(7)

'''
获取sub_core处理tile的垂直方向上大小
'''
def get_sub_tile_ver(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        sub_tile_ver = get_act_subtile_hor_ver(layer_num)[1]
        return '{:07b}'.format(sub_tile_ver - 1)
    else:
        return return_value(7)

'''
任务块在output channel方向上需要计算的次数
'''
def get_out_chl_num(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        output_chl = find_count(layer_num, "out_channels:")
        output_tile_chl = int(output_chl / nnbaton_Kp / nnbaton_K2)
        out_chl_num = int(output_tile_chl / (nnbaton_Kc * 16) - 1)
        return '{:04b}'.format(out_chl_num)
    else:
        return return_value(4)

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
def get_Mini_tile_hor(layer_num):
    layertype = find_type(layer_num, "layer type:")
    if "conv" in layertype:
        mini_tile_hor = get_act_minitile_hor_ver(layer_num)[0]
        return '{:05b}'.format(mini_tile_hor - 1)
    elif "pool" in layertype:
        mini_tile_hor = find_count(layer_num, "pool_size:")
        return '{:05b}'.format(mini_tile_hor - 1)
    else:
        return return_value(5)

'''
获取Mini tile的高度
'''
def get_Mini_tile_ver(layer_num):
    layertype = find_type(layer_num, "layer type:")
    if "conv" in layertype:
        mini_tile_ver = get_act_minitile_hor_ver(layer_num)[1]
        return '{:05b}'.format(mini_tile_ver - 1)
    elif "pool" in layertype:
        mini_tile_hor = find_count(layer_num, "pool_size:")
        return '{:05b}'.format(mini_tile_hor - 1)
    else:
        return return_value(5)

'''
获取任务块中需要计算的chl数据
'''
def get_in_chl_num(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        act_input_channel = find_count(layer_num, "out_channels:")
        act_input_channel = int(act_input_channel / nnbaton_Kp / nnbaton_K2)
        return '{:09b}'.format(act_input_channel - 1)
    else:
        return return_value(9)

'''
sub_core计算出来的mini tile的水平方向大小
'''
def get_Out_mini_tile_hor(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        out_hor = find_count(layer_num, "feature_map_size_x:")
        mini_tile_hor = int(out_hor / nnbaton_X2 / nnbaton_Xc / nnbaton_X1)
        return '{:07b}'.format(mini_tile_hor - 1)
    else:
        return return_value(7)

'''
sub_core计算出来的mini tile的垂直方向大小
'''
def get_Out_mini_tile_ver(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        out_ver = find_count(layer_num, "feature_map_size_y:")
        mini_tile_hor = int(out_ver / nnbaton_Yp / nnbaton_Y2 / nnbaton_Y1)
        return '{:07b}'.format(mini_tile_hor - 1)
    else:
        return return_value(7)

'''
获取output tile的水平方向大小
'''
def get_Out_tile_hor(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        out_W = find_count(layer_num, "feature_map_size_x:")
        out_tile_hor = int(out_W / nnbaton_X2)
        return '{:07b}'.format(out_tile_hor - 1)
    else:
        return return_value(7)

'''
获取output tile的垂直方向大小
'''
def get_Out_tile_ver(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        out_H = find_count(layer_num, "feature_map_size_y:")
        out_tail_ver = int(out_H / nnbaton_Yp / nnbaton_Y2)
        return '{:07b}'.format(out_tail_ver - 1)
    else:
        return return_value(7)

'''
获取output tile的通道方向大小
'''
def get_Out_tile_chl(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        out_C = find_count(layer_num, "out_channels:")
        out_tile_chl = int(out_C / nnbaton_K2)
        return '{:011b}'.format(out_tile_chl - 1)
    else:
        return return_value(11)

# get_weight_total_size 获取weight_size值的函数
def get_weight_total_size_num(layer_num):
    input_chl = find_count(layer_num, "in_channels:")
    kernel_size = find_count(layer_num, "kernel_size_x:") * find_count(layer_num, "kernel_size_y:")
    weight_size = int(input_chl * kernel_size * weight_bit / 64)
    return weight_size

'''
获取一个output channel的weight大小
in_channel和out_channel相等，不需要反推
'''
def get_weight_total_size(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        weight_size = get_weight_total_size_num(layer_num)
        return '{:017b}'.format(weight_size)
    else:
        return return_value(17)

'''
表示当前一个ping ram或pong ram能存下的一个output channel的weight大小
'''
def get_Weight_sub_size(layer_num):
    weitght_bit = 8
    tile_mode = int(8 / (nnbaton_Xc * nnbaton_Yc))
    if "conv" in find_type(layer_num, "layer type:"):
        kernel_size_x = find_count(layer_num, "kernel_size_x:")
        kernel_size_y = find_count(layer_num, "kernel_size_y:")
        input_channel = find_count(layer_num, "in_channels:")
        Weight_sub_size = int ((kernel_size_x * kernel_size_y) * input_channel * weitght_bit * tile_mode / 64)
        return '{:011b}'.format(Weight_sub_size)
    else:
        return return_value(11)

'''
获取weight ram可以存下output_channel/tile_mode的个数
'''
def get_weight_ram_output_chl_num(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        output_channel = int(find_count(layer_num, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        tile_mode = int(8 / (nnbaton_Xc * nnbaton_Yc))
        weight_ram_output_chl_num = int(output_channel / tile_mode) - 1
        return '{:07b}'.format(weight_ram_output_chl_num)
    else:
        return return_value(7)

'''
获取卷积类型
00：普通卷积
01：空洞卷积
10：反卷积
11：保留
'''
def get_conv_type(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        return '{:02b}'.format(0)
    else:
        return return_value(2)

'''
获取chiplet的工作模式
kp = 4时为1（四块chiplet），kp = 1时为0（chiplet各自工作）
'''
def get_Chiplet_mode():
    if nnbaton_Kp == 4:
        return '{:01b}'.format(1)
    elif nnbaton_Kp == 1:
        return '{:01b}'.format(0)

'''
获取多个chiplet协同工作时，每块input channel对应的weight存储
等于weight_total_size/chiplet_mode
'''
def get_Chiplet_tile_size(layer_num):
    if nnbaton_Kp == 4:
        Chiplet_tile_size = get_weight_total_size_num(layer_num) // 4
        return '{:010b}'.format(Chiplet_tile_size)
    else:
        return return_value(10)

'''
获取kernel宽度乘以高度的大小
FC的kernel_size为1*1
'''
def get_Kernel_num(layer_num):
    if "fc" in find_type(layer_num, "layer type:"):
        kernel_num = 1
        return '{:08b}'.format(kernel_num - 1)
    else:
        kernel_width = find_count(layer_num, "kernel_size_x:")
        kernel_height = find_count(layer_num, "kernel_size_y:")
        kernel_num = kernel_height * kernel_width
        return '{:08b}'.format(kernel_num - 1)

'''
获取kernel宽度的大小
'''
def get_Kernel_width(layer_num):
    if "fc" in find_type(layer_num, "layer type:"):
        Kernel_width = 1
        return '{:04b}'.format(Kernel_width - 1)
    elif find_count(layer_num, "kernel_size_x:") == None:
        return return_value(4)
    else:
        Kernel_width = find_count(layer_num, "kernel_size_x:")
        return '{:04b}'.format(Kernel_width - 1)

'''
获取kernel高度的大小
'''
def get_Kernel_height(layer_num):
    if "fc" in find_type(layer_num, "layer type:"):
        kernel_height = 1
        return '{:04b}'.format(kernel_height - 1)
    elif find_count(layer_num, "kernel_size_y:") == None:
        return return_value(4)
    else:
        kernel_height = find_count(layer_num, "kernel_size_y:")
        return '{:04b}'.format(kernel_height - 1)

'''
获取步长大小
000：跨度为1；   001：跨度为2；   010：跨度为3；   011：跨度为4；
100：跨度为5；   101：跨度为6；   110：跨度为7；   111：跨度为8；
'''
def get_Kernel_str(layer_num):
    Kernel_str = find_count(layer_num, "stride_x:")
    if Kernel_str == None:
        return return_value(3)
    else:
        return '{:03b}'.format(Kernel_str - 1)

'''
获取反卷积在水平方向上的跨度
'''
def get_deconv_hor_str(layer_num):
    return '{:02b}'.format(0)

'''
获取反卷积在垂直方向上的跨度
'''
def get_deconv_ver_str(layer_num):
    return '{:02b}'.format(0)

'''
获取反卷空洞卷积，插入零的个数
'''
def get_Dilation_rate(layer_num):
    return '{:03b}'.format(0)

'''
取整方式选择
取整方式先按照1获取，后期在pt文件中获取
'''
def get_pooling_carry_sel(layer_num):
    return '{:01b}'.format(1)

'''
返回0：平均池化；(avgpooling)
返回1：最大池化；(maxpooling)
非pool返回0
'''
def get_pool_mode(layer_num):
    pool_mode = find_type(layer_num, "layer type:")
    if "avgpool" in pool_mode:
        return '{:01b}'.format(0)
    elif "maxpool" in pool_mode:
        return '{:01b}'.format(1)
    else:
        return return_value(1)

'''
表示定点转32位浮点的量化系数（暂时先不管）
'''
def get_Pooling_quan_code_in(layer_num):
    return return_value(4)

'''
池化的尺寸乘积结果
'''
def get_pool_total_size(layer_num):
    pool_total_size = find_count(layer_num, "pool_size:")
    if pool_total_size == None:
        return return_value(10)
    else:
        return '{:010b}'.format(pool_total_size * pool_total_size - 1)

'''
获取池化的size
'''
def get_pool_size(layer_num):
    pool_size = find_count(layer_num, "pool_size:")
    if pool_size == None:
        return return_value(5)
    else:
        return '{:05b}'.format(pool_size)

'''
给平均池化使用的任意尺寸的倒数
'''
def get_Pooling_oprands(layer_num):
    type = find_type(layer_num, "layer type:")
    if "avgpool" in type:
        pool_size = find_count(layer_num, "pool_size:")
        return '{:032b}'.format(int(st.unpack('I', st.pack('f', 1 / (pool_size * pool_size)))[0]))
    else:
        return return_value(32)

'''
表示32位浮点转定点的量化系数（暂时不管）
'''
def get_Pooling_quan_code_out(layer_num):
    return return_value(4)

'''
scaling放大倍数:
00：bilinearx2   01：bilinearx4
10：nearestx2    11: nearestx4
'''
def get_scaling_mode(layer_num):
    return return_value(2)

'''
全连接模式使能信号，返回1有效
'''
def get_fc_mode_en(layer_num):
    if "fc" in find_type(layer_num, "layer type:"):
        return '{:01b}'.format(1)
    else:
        return 0

#################  inst_01  #################
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
def get_tile_num():
    return '{:05b}'.format(nnbaton_X1 * nnbaton_Y1)

def padding_mode_cout(layer_num, tile_num):
    tile_case_hor = tile_num // nnbaton_X2 % (nnbaton_Yp * nnbaton_Y2)  # 当前tile是哪一行
    # tile_num // nnbaton_X2表示当前tile所在的行数(取值范围为[0 , +∞)， 即第一行为0)
    # nnbaton_Yp * nnbaton_Y2表示一面墙有多少行tile
    # tile_num // nnbaton_X2 % (nnbaton_Yp * nnbaton_Y2)表示排列的是哪一面墙的哪一行tile
    tile_case_ver = tile_num % nnbaton_X2  # 当前tile是哪一列
    return tile_case_hor, tile_case_ver


'''
获取当前指令中，tile的padding模式
'''
def get_Padding_mode(layer_num, tile_num):
    if "conv" in find_type(layer_num, "layer type:"):
        tile_case_hor, tile_case_ver = padding_mode_cout(layer_num, tile_num)

        if tile_case_hor == 0:
        # 表示第一行tile
            if tile_case_ver == 0:
            # 表示第一列tile
                return '{:04b}'.format(0b0011)
            elif tile_case_ver == nnbaton_X2 - 1:
            # 最后一列tile
                return '{:04b}'.format(0b0110)
            elif (tile_case_ver > 0) and (tile_case_ver < nnbaton_X2 - 1):
            # 中间的tile
                return '{:04b}'.format(0b0010)
            else:
                return '{:04b}'.format(0)
        elif tile_case_hor == nnbaton_Yp * nnbaton_Y2 - 1:
        # 最后一行
            if tile_case_ver == 0:
            # 表示第一列tile
                return '{:04b}'.format(0b1001)
            elif tile_case_ver == nnbaton_X2 - 1:
            # 最后一列tile
                return '{:04b}'.format(0b1100)
            elif (tile_case_ver > 0) and (tile_case_ver < nnbaton_X2 - 1):
                return '{:04b}'.format(0b1000)
            else:
                return '{:04b}'.format(0)
        elif (tile_case_hor > 0) and (tile_case_hor < nnbaton_Yp * nnbaton_Y2 - 1):
        # 第二行到倒数第二行tile
            if tile_case_ver == 0:
            # 表示第一列tile
                return '{:04b}'.format(0b0001)
            elif tile_case_ver == nnbaton_X2 - 1:
            # 最后一列tile
                return '{:04b}'.format(0b0100)
            else:
                return '{:04b}'.format(0)
        else:
            return '{:04b}'.format(0)
    else:
        return '{:04b}'.format(0)

'''
获取当前指令中，padding的数量
'''
def get_Padding_num(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        padding_num = find_count(layer_num, "padding_num:")
        return '{:04b}'.format(padding_num - 1)
    else:
        return '{:04b}'.format(0)

'''
获取当前指令中，每块L1数据需要重复发送的次数
'''
def get_Repeat_send_num():
    return return_value(4)

'''
获取当前指令中，当前指令不做rd_dma不做任何读取activation的操作
'''
def get_act_inst_bypass():
    return return_value(1)

'''
获取当前指令中，当前指令不做rd_dma不做任何读取weight的操作
'''
def get_weight_inst_bypass():
    return return_value(1)

def get_out_pool_CHW(layer_num):
    C = H = W = 0
    layer_type = find_type(layer_num, "layer type:")
    if "conv" in layer_type:
        C = find_count(layer_num, "in_channels:")
        H = find_count(layer_num, "feature_map_size_y:")
        W = find_count(layer_num, "feature_map_size_x:")
    elif "fc" in layer_type:
        C, H, W = get_out_pool_CHW(layer_num - 1)
    elif "pool" in layer_type or "Pool" in layer_type:
        prev_layer_type = find_type(layer_num - 1, "layer type:")
        if "pool" in prev_layer_type or "Pool" in prev_layer_type:
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

'''
整个指令所需act数据channel方向上有多少个64bit,不与rd_dma统一；
'''
def get_act_chl_one_inst_real(layer_num):
    layer_type = find_type(layer_num, "layer type:")
    if "conv" in layer_type:
        input_channel = find_count(layer_num, "in_channels:")
        act_chl_one_inst_real = int(input_channel / (64 / act_bit))
        return '{:012b}'.format(act_chl_one_inst_real)
    elif "pool" in layer_type or "Pool" in layer_type:
        input_channel = get_out_pool_CHW(layer_num - 1)[0]
        act_chl_one_inst_real = int(input_channel / (64 / act_bit))
        return '{:012b}'.format(act_chl_one_inst_real)
    else:
        return return_value(12)

'''
一个ping/pong sram 存储数据在channel方向上有多少个64bit
'''
def get_act_sub_ch():
    return return_value(12)

'''
整个指令所需act数据channel方向上有多少个64bit，且与rd_dma统一；
'''
def get_act_chl_one_inst(layer_num):
    layer_type = find_type(layer_num, "layer type:")
    if "conv" in layer_type:
        input_channel = find_count(layer_num, "in_channels:")
        input_channel = align(input_channel, int(128 / act_bit))
        act_chl_one_inst = int(input_channel / (64 / act_bit))
        return '{:012b}'.format(act_chl_one_inst)
    elif "pool" in layer_type or "Pool" in layer_type:
        input_channel = get_out_pool_CHW(layer_num - 1)[0]
        input_channel = align(input_channel, int(128 / act_bit))
        act_chl_one_inst = int(input_channel / (64 / act_bit))
        return '{:012b}'.format(act_chl_one_inst)
    else:
        return return_value(12)

'''
指令无效标志:1：act和上一次一样   0:act和上一次不一样
'''
def get_inst_invalid(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        if nnbaton_divcase % 2 == 1:
            # nnbaton_divcase为1、3时，先走channel方向，即需要更新activation
            return '{:01b}'.format(0)
        else:
            # nnbaton_divcase为2、4时，先走平面方向，即需要更新weight
            return '{:01b}'.format(1)
    else:
        return '{:01b}'.format(1)

'''
获取一个sub_tile中水平方向上mini_tile的数量
'''
def get_mini_ver_num():
    return '{:07b}'.format(nnbaton_X1)

'''
获取一个sub_tile中垂直方向上mini_tile的数量
'''
def get_mini_hor_num():
    return '{:07b}'.format(nnbaton_Y1)

#################  inst_11  #################
'''
获取当前指令的ID
'''
def get_Inst_id_11():
    return '{:02b}'.format(0b11)

'''
output tile整块偏移量
'''
def get_Out_feature_map_ver(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        out_W = find_count(layer_num, "feature_map_size_x:")
        out_tile_W = int(out_W / nnbaton_X2)

        out_H = find_count(layer_num, "feature_map_size_y:")
        out_tile_H = int(out_H / nnbaton_Yp /nnbaton_Y2)

        Out_feature_map_ver = 16 * out_tile_W * out_tile_H * act_bit

        return '{:032b}'.format(Out_feature_map_ver)
    else:
        return return_value(32)

'''
Output Tile地址跳变结束当前tile的行的跳变后，转到下一行所需要给地址加的偏移量
'''
def get_Out_tile_stride(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        out_W = find_count(layer_num, "feature_map_size_x:")
        out_tile_W = int(out_W / nnbaton_X2)

        Out_tile_stride = 16 * out_tile_W * act_bit
        return '{:020b}'.format(Out_tile_stride)
    else:
        return return_value(20)

'''
表示rd_dma当前收到的读DDR指令是否写入LLC_act，低电平有效，如果需要写入，则为0
(和上一个tile相比，不需要更新act，拉高跳过读取操作)
'''
def get_act_updata_n(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        if nnbaton_divcase % 2 == 1:
            # nnbaton_divcase为1、3时，先走channel方向，即需要更新activation
            return '{:01b}'.format(0)
        else:
            # nnbaton_divcase为2、4时，先走平面方向，即需要更新weight
            return '{:01b}'.format(1)
    else:
        return '{:01b}'.format(1)

def get_LLC_a_ping_pong(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        if nnbaton_divcase % 2 == 1:
            # nnbaton_divcase为1、3时，先走channel方向，即需要更新activation
            return '{:01b}'.format(0)
        else:
            # nnbaton_divcase为2、4时，先走平面方向，即需要更新weight
            return '{:01b}'.format(1)
    else:
        return '{:01b}'.format(1)


'''
一行act在DDR中连续存储的大小（一行数据的大小）
'''
def get_act_str_line(layer_num):
    layer_type = find_type(layer_num, "layer type:")
    input_W = 0

    if "conv" in layer_type:
        prev_layer_type = find_type(layer_num - 1, "layer type:")
        if prev_layer_type == None:
            input_W = find_count(layer_num, "feature_map_size_x:")
        elif "conv" in prev_layer_type:
            input_W = find_count(layer_num - 1, "feature_map_size_x:")
        elif "pool" in prev_layer_type or "Pool" in prev_layer_type:
            input_W = get_out_pool_CHW(layer_num - 1)[2]
    elif "pool" in layer_type or "Pool" in layer_type:
        input_W = get_out_pool_CHW(layer_num - 1)[2]
    elif "fc" in layer_type:
        input_W = find_count(layer_num, "in_features_x:")

    act_str_line = input_W * 16 * act_bit
    return '{:020b}'.format(act_str_line)

'''
一层activation在DDR中连续存储的大小（一面墙数据的大小）
'''
def get_act_str_chl(layer_num):
    layer_type = find_type(layer_num, "layer type:")
    input_W = 0
    input_C = 0
    if "conv" in layer_type:
        input_C = find_count(layer_num, "in_channels:")
        if layer_num == 1 or "pool" in find_type(layer_num - 1, "layer type:"):
            input_W = find_count(layer_num, "feature_map_size_x:")
        else:
            input_W = find_count(layer_num - 1, "feature_map_size_x:")

    elif "Pool" in layer_type or "pool" in layer_type:
        return return_value(32)

    elif "fc" in layer_type:
        input_W = input_C = find_count(layer_num, "in_features_x:")

    act_str_chl = input_W * input_C * 16 * act_bit
    return '{:032b}'.format(act_str_chl)

'''
表示rd_dma当前收到的读DDR指令是否写入weight_LLC，低电平有效，如果需要写入，则为0
(和上一个tile相比，不需要更新weight，拉高跳过读取操作)
'''
def get_weight_updata_n(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        if nnbaton_divcase % 2 == 1:
            # nnbaton_divcase为1、3时，先走channel方向，即需要更新activation
            return '{:01b}'.format(1)
        else:
            # nnbaton_divcase为2、4时，先走平面方向，即需要更新weight
            return '{:01b}'.format(0)
    else:
        return '{:01b}'.format(1)


def get_LLC_w_ping_pong(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        if nnbaton_divcase % 2 == 1:
            # nnbaton_divcase为1、3时，先走channel方向，即需要更新activation
            return '{:01b}'.format(1)
        else:
            # nnbaton_divcase为2、4时，先走平面方向，即需要更新weight
            return '{:01b}'.format(0)
    else:
        return '{:01b}'.format(1)

'''
表示当前指令需要给LLC_weight传输一共多少个output_channel
'''
def get_weight_output_chl(layer_num):
    if "conv" in find_type(layer_num, "layer type:"):
        output_chl = int(find_count(layer_num, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        return '{:011b}'.format(output_chl - 1)
    else:
        return return_value(11)

def get_act_addr_element(layer_num):
    return return_value(32)

def get_Out_tile_start_addr(layer_num, output_address_start, tile_num):
    '''
    output当前tile左上角第一个激活的地址
    '''
    if "conv" in find_type(layer_num, "layer type:"):
        out_W = find_count(layer_num, "feature_map_size_x:")
        out_tile_W = int(out_W / nnbaton_X2)

        out_H = find_count(layer_num, "feature_map_size_y:")
        out_tile_H = int(out_H / nnbaton_Yp / nnbaton_Y2)

        out_C = find_count(layer_num, "out_channels:")
        out_tail_C = int(out_C / nnbaton_Kp / nnbaton_K2)

        out_tile_start_addr = output_address_start + out_tail_C * (
                    tile_num // (nnbaton_Yp * nnbaton_Y2 * nnbaton_X2)) * out_H * out_W * act_bit \
                              + 16 * out_W * out_tile_H * (tile_num % nnbaton_X2) * act_bit \
                              + 16 * out_tile_W * act_bit * (tile_num % (nnbaton_Yp * nnbaton_Y2 * nnbaton_X2))
        # output_address_start + out_tail_C * (tile_num // (nnbaton_Yp * nnbaton_Y2 * nnbaton_X2)) * out_H * out_W * act_bit \
        #   + 16 * out_W * out_tile_H * (tile_num % nnbaton_X2) * act_bit  @每一行tile左上角的第一个点
        #   out_tail_C * (tile_num // (nnbaton_Yp * nnbaton_Y2 * nnbaton_X2)) * out_H * out_W * act_bit  @前面墙的大小
        #   16 * out_W * out_tile_H * (tile_num % nnbaton_X2) * act_bit  @以16chl为单位的前几行tile的大小
        # tile_num // (nnbaton_Yp * nnbaton_Y2 * nnbaton_X2) 确定是哪一面墙的output tile
        # tile_num % (nnbaton_Yp * nnbaton_Y2 * nnbaton_X2) 确定是平面上哪一个output tile
        # 16 * out_tile_W * act_bit * (tile_num % (nnbaton_Yp * nnbaton_Y2 * nnbaton_X2)) @以16chal为单位下一个output所需要的偏移量
        return '{:032b}'.format(out_tile_start_addr)
    else:
        return '{:032b}'.format(output_address_start)

'''
校对小于0的坐标，返回0
'''
def proofreading(pro_num):
    if pro_num < 0:
        return 0
    elif pro_num > 223:
        return 223
    else:
        return pro_num

'''
从存储器读取每一个tile的数据存储起始地址
'''
def get_act_addr(layer_num, tile_num, act_start):
    if "conv" in find_type(layer_num, "layer type:"):
        output_C = find_count(layer_num, "out_channels:")
        output_H = find_count(layer_num, "feature_map_size_x:")
        output_W = find_count(layer_num, "feature_map_size_y:")
        output_tile_C = int(output_C / nnbaton_Kp / nnbaton_K2)
        output_tile_H = int(output_H / nnbaton_Yp / nnbaton_Y2)
        output_tile_W = int(output_W / nnbaton_X2)

        if layer_num == 1:
            input = get_fstin_CHW(netpath)
            input_C = input[0]
            input_H = input[1]
            input_W = input[2]
        else:
            input = get_out_CHW(layer_num - 1)
            input_C = input[0]
            input_H = input[1]
            input_W = input[2]

        stride_size = find_count(layer_num, "stride_x:")

        '''
        output_tile和intput_tile左上角的点的坐标
        '''
        output_leftup_h = output_tile_H * int(tile_num // nnbaton_X2 % (nnbaton_Yp * nnbaton_Y2))
        # output_tile点的h方向应该使用output_tile_H的值乘(当前tile所在的行数 - 1)
        # tile_num // nnbaton_X2表示当前tile所在的行数
        # nnbaton_Yp * nnbaton_Y2表示一面墙有多少行tile
        # tile_num // nnbaton_X2 % (nnbaton_Yp * nnbaton_Y2)表示排列的是哪一面墙的哪一行tile
        output_leftup_w = output_tile_W * (tile_num % nnbaton_X2)
        # output_tile点的w方向使用output_tile_W的值乘(当前tile所在的列数 - 1)
        # tile_num % nnbaton_X2表示当前tile在哪一列
        # print("output left up point : ", output_leftup_c, output_leftup_w, output_leftup_h)

        input_leftup_fx = input_leftup_fy = 0
        input_leftup_c = 0
        input_leftup_h = proofreading(output_leftup_h * stride_size + input_leftup_fy - 1)
        input_leftup_w = proofreading(output_leftup_w * stride_size + input_leftup_fx - 1)
        # input_tile点的channel方向和input块的channel大小相同，H和W方向根据公式推算

        act_addr = act_start + index_to_address(1, input_C, input_H, input_W, 0, input_leftup_c, input_leftup_h, input_leftup_w, 8)[0]
        return '{:032b}'.format(act_addr)

    else:
        return '{:032b}'.format(act_start)

def get_weight_addr(layer_num):
    return return_value(32)

def get_Run_mode(layer_num):
    return return_value(2)

def get_nnbaton(layer_num):
    if layer_num == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if layer_num == 1:
        return 2, 2, 2, 7, 1, 2, 1, 4, 1, 4, 2, 1, 8, 8, 8, 1
    elif layer_num == 2:
        return 2, 1, 1, 7, 7, 1, 1, 4, 4, 1, 2, 8, 8, 8, 8, 4
    elif layer_num == 4:
        return 1, 1, 1, 7, 7, 1, 4, 1, 2, 2, 2, 2, 8, 8, 8, 1
    elif layer_num == 5:
        return 1, 1, 1, 7, 7, 1, 4, 1, 2, 2, 2, 4, 8, 8, 8, 4
    elif layer_num == 7:
        return 1, 1, 1, 4, 7, 1, 4, 1, 4, 1, 2, 4, 8, 8, 8, 4
    elif layer_num == 8:
        return 1, 1, 1, 4, 7, 1, 4, 1, 4, 1, 2, 8, 8, 8, 8, 1
    elif layer_num == 9:
        return 1, 1, 1, 4, 7, 1, 4, 1, 4, 1, 2, 8, 8, 8, 8, 4
    elif layer_num == 11:
        return 1, 1, 1, 2, 4, 2, 4, 1, 4, 1, 2, 8, 8, 8, 8, 2
    elif layer_num == 12:
        return 1, 1, 4, 2, 1, 2, 4, 1, 1, 4, 2, 16, 8, 8, 8, 4
    elif layer_num == 13:
        return 1, 1, 4, 2, 1, 2, 4, 1, 1, 4, 2, 16, 8, 8, 8, 1
    elif layer_num == 15:
        return 1, 1, 2, 1, 1, 2, 4, 1, 2, 2, 2, 16, 8, 8, 8, 4
    elif layer_num == 16:
        return 1, 1, 2, 1, 1, 2, 4, 1, 2, 2, 2, 16, 8, 8, 8, 3
    elif layer_num == 17:
        return 1, 1, 2, 1, 1, 2, 4, 1, 2, 2, 2, 16, 8, 8, 8, 1
    else: # pool fc or undefine return code
        return 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0

'''
获取所有input块的地址
'''
def get_layer_input_addr(act_layer_addr, layer_cnt):
    input_addr = []
    del act_layer_addr[0]
	
    for i in range(len(act_layer_addr)):
        if layer_cnt == 1 and "input" in act_layer_addr[i][0]:
            return [act_layer_addr[i][1]]
        if f"layer{layer_cnt}" not in act_layer_addr[i][0]:
            input_addr.append(act_layer_addr[i][1])

    return input_addr

'''
获取所有output块的地址
'''
def get_layer_outpu_addr(layers_act_addr):
    output_addr = []

    del layers_act_addr[0]
    layers_act_addr.append(layers_act_addr[0])
    del layers_act_addr[0]

    for i in range(len(layers_act_addr)):
        output_addr.append(layers_act_addr[i][1])

    return output_addr


def get_layer_inst(insts, layer_num, tile_cnt, act_start, out_start, wet_addr):
    fmain.write(f'{act_start}\n')
    if len(wet_addr):
        fmain.write(f'{wet_addr}\n')

    for i in range(1, tile_cnt + 1):
        if len(wet_addr):
            wet_start = wet_addr[1][0]
            wet_end = wet_addr[1][1]
            inst_00, inst_01, inst_11 = get_tile_inst(layer_num, i-1, act_start, out_start, wet_start, wet_end)
        else:
            inst_00, inst_01, inst_11 = get_tile_inst(layer_num, i-1, act_start, out_start, "", "")
        insts[i-1] = [f'layer{layer_num} tile{i} instruction', [inst_00], [inst_01], [inst_11]]
        insts.append(insts[i-1])

    return insts

def refresh_nnbaton(X1, Y1, K1, X2, Y2, K2, Kp, Yp, Kc, Yc, Xc, C1, C0, X0, Y0, divcase):
    global nnbaton_X1
    global nnbaton_Y1
    global nnbaton_K1
    global nnbaton_X2
    global nnbaton_Y2
    global nnbaton_K2
    global nnbaton_Kp
    global nnbaton_Yp
    global nnbaton_Kc
    global nnbaton_Yc
    global nnbaton_Xc
    global nnbaton_C1
    global nnbaton_C0
    global nnbaton_X0
    global nnbaton_Y0
    global nnbaton_divcase

    nnbaton_X1 = X1
    nnbaton_Y1 = Y1
    nnbaton_K1 = K1
    nnbaton_X2 = X2
    nnbaton_Y2 = Y2
    nnbaton_K2 = K2
    nnbaton_Kp = Kp
    nnbaton_Yp = Yp
    nnbaton_Kc = Kc
    nnbaton_Yc = Yc
    nnbaton_Xc = Xc
    nnbaton_C1 = C1
    nnbaton_C0 = C0
    nnbaton_X0 = X0
    nnbaton_Y0 = Y0
    nnbaton_divcase = divcase

def pass_fun():
    pass

def get_tile_inst(layer_num, tile_num, act_start, out_start, wet_start, wet_end):
    fmain.write("get_tile_inst layer%2s %2s %8s %8s %8s %8s\n" %
        (layer_num, tile_num, act_start, out_start, wet_start, wet_end))

    fc_mode_en = get_fc_mode_en(layer_num)
    scaling_mode = get_scaling_mode(layer_num)
    Pooling_quan_code_out = get_Pooling_quan_code_out(layer_num)
    Pooling_oprands = get_Pooling_oprands(layer_num)
    pool_size = get_pool_size(layer_num)
    pool_total_size = get_pool_total_size(layer_num)
    Pooling_quan_code_in = get_Pooling_quan_code_in(layer_num)
    pool_mode = get_pool_mode(layer_num)
    pooling_carry_sel = get_pooling_carry_sel(layer_num)
    Dilation_rate = get_Dilation_rate(layer_num)
    deconv_ver_str = get_deconv_ver_str(layer_num)
    deconv_hor_str = get_deconv_hor_str(layer_num)
    Kernel_str = get_Kernel_str(layer_num)
    Kernel_height = get_Kernel_height(layer_num)
    Kernel_width = get_Kernel_width(layer_num)
    Kernel_num = get_Kernel_num(layer_num)
    Chiplet_tile_size = get_Chiplet_tile_size(layer_num)
    Chiplet_mode = get_Chiplet_mode()
    conv_type = get_conv_type(layer_num)
    weight_ram_output_chl_num = get_weight_ram_output_chl_num(layer_num)
    Weight_sub_size = get_Weight_sub_size(layer_num)
    weight_total_size = get_weight_total_size(layer_num)
    Out_tile_chl = get_Out_tile_chl(layer_num)
    Out_tile_ver = get_Out_tile_ver(layer_num)
    Out_tile_hor = get_Out_tile_hor(layer_num)
    Out_mini_tile_ver = get_Out_mini_tile_ver(layer_num)
    Out_mini_tile_hor = get_Out_mini_tile_hor(layer_num)
    in_chl_num = get_in_chl_num(layer_num)
    Mini_tile_ver = get_Mini_tile_ver(layer_num)
    Mini_tile_hor = get_Mini_tile_hor(layer_num)
    out_chl_num = get_out_chl_num(layer_num)
    sub_tile_ver = get_sub_tile_ver(layer_num)
    sub_tile_hor = get_sub_tile_hor(layer_num)
    act_tile_sub_chl = get_act_tile_sub_chl(layer_num)
    act_tile_str = get_act_tile_str(layer_num)
    act_tile_chl = get_act_tile_chl(layer_num)
    act_tile_ver = get_act_tile_ver(layer_num, tile_num)
    act_tile_hor = get_act_tile_hor(layer_num, tile_num)
    Inst_id_00 = get_Inst_id_00()

    mini_hor_num = get_mini_hor_num()
    mini_ver_num = get_mini_ver_num()
    inst_invalid = get_inst_invalid(layer_num)
    act_chl_one_inst = get_act_chl_one_inst(layer_num)
    act_sub_ch = get_act_sub_ch()
    act_chl_one_inst_real = get_act_chl_one_inst_real(layer_num)
    weight_inst_bypass = get_weight_inst_bypass()
    act_inst_bypass = get_act_inst_bypass()
    Repeat_send_num = get_Repeat_send_num()
    Padding_num = get_Padding_num(layer_num)
    Padding_mode = get_Padding_mode(layer_num, tile_num)
    inst_tile_num = get_tile_num()
    Tile_mode = get_Tile_mode()
    Inst_id_01 = get_Inst_id_01()

    act_addr_element = get_act_addr_element(layer_num)
    Out_feature_map_ver = get_Out_feature_map_ver(layer_num)
    Out_tile_stride = get_Out_tile_stride(layer_num)
    Out_tile_start_addr = get_Out_tile_start_addr(layer_num, out_start, tile_num)
    LLC_a_ping_pong = get_LLC_a_ping_pong(layer_num)
    act_updata_n = get_act_updata_n(layer_num)
    act_str_line = get_act_str_line(layer_num)
    act_str_chl = get_act_str_chl(layer_num)
    act_addr = get_act_addr(layer_num, tile_num, act_start)
    LLC_w_ping_pong = get_LLC_w_ping_pong(layer_num)
    weight_updata_n = get_weight_updata_n(layer_num)
    weight_output_chl = get_weight_output_chl(layer_num)
    weight_addr = get_weight_addr(layer_num)
    Run_mode = get_Run_mode(layer_num)
    Inst_id_11 = get_Inst_id_11()

    inst_00 = [["fc_mode_en", fc_mode_en], ["scaling_mode", scaling_mode],
               ["Pooling_quan_code_out", Pooling_quan_code_out],
               ["Pooling_oprands", Pooling_oprands],
               ["pool_size", pool_size], ["pool_total_size", pool_total_size],
               ["Pooling_quan_code_in", Pooling_quan_code_in],
               ["pool_mode", pool_mode], ["pooling_carry_sel", pooling_carry_sel],
               ["Dilation_rate", Dilation_rate], ["deconv_ver_str", deconv_ver_str],
               ["deconv_hor_str", deconv_hor_str], ["Kernel_str", Kernel_str],
               ["Kernel_height", Kernel_height], ["Kernel_width", Kernel_width],
               ["Kernel_num", Kernel_num], ["Chiplet_tile_size", Chiplet_tile_size],
               ["Chiplet_mode  ", Chiplet_mode], ["conv_type", conv_type],
               ["weight_ram_output_chl_num", weight_ram_output_chl_num],
               ["Weight_sub_size", Weight_sub_size], ["weight_total_size", weight_total_size],
               ["Out_tile_chl", Out_tile_chl], ["Out_tile_ver", Out_tile_ver],
               ["Out_tile_hor", Out_tile_hor], ["Out_mini_tile_ver", Out_mini_tile_ver],
               ["Out_mini_tile_hor", Out_mini_tile_hor], ["in_chl_num", in_chl_num],
               ["Mini_tile_ver", Mini_tile_ver], ["Mini_tile_hor", Mini_tile_hor],
               ["out_chl_num", out_chl_num], ["sub_tile_ver", sub_tile_ver],
               ["sub_tile_hor", sub_tile_hor], ["act_tile_sub_chl", act_tile_sub_chl],
               ["act_tile_str", act_tile_str], ["act_tile_chl", act_tile_chl],
               ["act_tile_ver", act_tile_ver], ["act_tile_hor", act_tile_hor],
               ["Inst_id_00", Inst_id_00]]
    inst_01 = [["mini_hor_num", mini_hor_num], ["mini_ver_num", mini_ver_num],
               ["inst_invalid", inst_invalid], ["act_chl_one_inst", act_chl_one_inst],
               ["act_sub_ch", act_sub_ch], ["act_chl_one_inst_real", act_chl_one_inst_real],
               ["weight_inst_bypass", weight_inst_bypass], ["act_inst_bypass", act_inst_bypass],
               ["Repeat_send_num", Repeat_send_num],
               ["Padding_num", Padding_num], ["Padding_mode", Padding_mode],
               ["tile_num", inst_tile_num], ["Tile_mode", Tile_mode],
               ["Inst_id_01", Inst_id_01]]
    inst_11 = [["act_addr_element", act_addr_element], ["Out_feature_map_ver", Out_feature_map_ver],
               ["Out_tile_stride", Out_tile_stride], ["Out_tile_start_addr", Out_tile_start_addr],
               ["LLC_a_ping_pong", LLC_a_ping_pong], ["act_updata_n", act_updata_n],
               ["act_str_line", act_str_line], ["act_str_chl", act_str_chl],
               ["act_addr", act_addr], ["LLC_w_ping_pong", LLC_w_ping_pong],
               ["weight_updata_n", weight_updata_n], ["weight_output_chl", weight_output_chl],
               ["weight_addr", weight_addr], ["Run_mode", Run_mode],
               ["Inst_id_11", Inst_id_11]]

    return inst_00, inst_01, inst_11
