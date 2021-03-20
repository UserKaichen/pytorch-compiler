import struct as st

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

layercnts = 0
fmain = logpath = 0

def send_instsfunvar(fw, path, cnt):
    global fmain
    global logpath
    global layercnts

    fmain = fw
    logpath = path
    layercnts = cnt

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

def find_count(layernum, find_name):
    '''
    获取相应的数据
    参数： layernum：哪一层的数据；  find_name：获取标志(日志中数据的名称加：)
    '''
    with open(logpath, 'r') as file:
        for line in file:
            if line.startswith(f"layer_num:{layernum + 1} "):
                break
            if line.startswith(f"layer_num:{layernum} "):
                for line in file:
                    if line.startswith(f"layer_num:{layernum + 1} "):
                        break
                    if find_name in line:
                        return (int(line.split(find_name)[1].split(" ")[0]))

def find_type(layernum, find_name):
    '''
    获取相应的层的类型
    参数： layernum：哪一层的类型；  find_name：获取标志(日志中数据的名称加：)
    '''
    with open(logpath, 'r') as file:
        for line in file:
            if line.startswith(f"layer_num:{layernum} "):
                return (line.split(find_name)[1].split(" ")[0])

def get_Inst_id_00():
    '''
    获取当前指令的ID
    '''
    return '{:02b}'.format(00)

def get_act_tile_hor_var(layernum):
    '''
    获取act  tile的水平(hor)和垂直(var)方向的大小，通过output推算input
    '''
    out_H = find_count(layernum, "feature_map_size_y:")
    out_W = find_count(layernum, "feature_map_size_x:")
    stride = find_count(layernum, "stride_x:")
    kernel_size = find_count(layernum, "kernel_size_x:")
    out_tail_hor = out_W / nnbaton_X2
    out_tail_ver = out_H / nnbaton_Yp / nnbaton_Y2
    act_tail_hor = out_tail_hor * stride - 1 + kernel_size
    act_tail_ver = out_tail_ver * stride - 1 + kernel_size
    return act_tail_hor, act_tail_ver

def get_act_tile_hor(layernum):
    '''
    获取act  tile的水平方向的大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        act_tile_hor = int(get_act_tile_hor_var(layernum)[0])
        return '{:07b}'.format(act_tile_hor - 1)
    else:
        return return_value(7)

def get_act_tile_ver(layernum):
    '''
    获取act  tile的垂直方向的大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        act_tile_ver = int(get_act_tile_hor_var(layernum)[0])
        return '{:07b}'.format(act_tile_ver - 1)
    else:
        return return_value(7)

def get_act_tile_chl(layernum):
    '''
    获取act tile在input channel方向上有多少个stride大小的数据块
    '''
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

def get_act_tile_str(layernum):
    '''
    获取act tile的多少bit为一组，第一层为64bit，其它层为128bit
    '''
    if layernum == 1:
        return '{:02b}'.format(0b1)
    else:
        return '{:02b}'.format(0b0)

def get_act_subtile_hor_ver(layer_num):
    """
    获取act sub_tile水平和垂直方向大小
    """
    out_H = find_count(layer_num, "feature_map_size_y:")
    out_W = find_count(layer_num, "feature_map_size_x:")
    stride = find_count(layer_num, "stride_x:")
    kernel_size = find_count(layer_num, "kernel_size_x:")
    out_tail_hor = int(out_W / nnbaton_X2 / nnbaton_Xc)
    out_tail_ver = int(out_H / nnbaton_Yp / nnbaton_Y2 / nnbaton_Yc)
    act_tail_hor = out_tail_hor * stride - 1 + kernel_size
    act_tail_ver = out_tail_ver * stride - 1 + kernel_size
    return act_tail_hor, act_tail_ver

def get_act_tile_sub_chl(layernum):
    '''
    获取act tile向下游切换的ping pang新号的数据量大小，相当于input的sub_tile大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        act_subtail = get_act_subtile_hor_ver(layernum)
        return '{:09b}'.format(act_subtail[0] * act_subtail[1])
    else:
        return return_value(9)

def get_sub_tile_hor(layernum):
    '''
    获取sub_core处理tile的水平方向上大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        sub_tile_hor = get_act_subtile_hor_ver(layernum)[0]
        return '{:07b}'.format(sub_tile_hor - 1)
    else:
        return return_value(7)

def get_sub_tile_ver(layernum):
    '''
    获取sub_core处理tile的垂直方向上大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        sub_tile_ver = get_act_subtile_hor_ver(layernum)[1]
        return '{:07b}'.format(sub_tile_ver - 1)
    else:
        return return_value(7)

def get_out_chl_num(layernum):
    '''
    任务块在output channel方向上需要计算的次数
    '''
    pass

def get_act_minitile_hor_ver(layer_num):
    '''
    获取act mini_tile水平和垂直方向大小
    '''
    out_H = find_count(layer_num, "feature_map_size_y:")
    out_W = find_count(layer_num, "feature_map_size_x:")
    stride = find_count(layer_num, "stride_x:")
    kernel_size = find_count(layer_num, "kernel_size_x:")
    out_tail_hor = int(out_W / nnbaton_X2 / nnbaton_Xc / nnbaton_X1)
    out_tail_ver = int(out_H / nnbaton_Yp / nnbaton_Y2 / nnbaton_Yc / nnbaton_Y1)
    act_tail_hor = out_tail_hor * stride - 1 + kernel_size
    act_tail_ver = out_tail_ver * stride - 1 + kernel_size
    return act_tail_hor, act_tail_ver

def get_Mini_tile_hor(layernum):
    '''
    获取Mini tile的宽度
    '''
    if "conv" in find_type(layernum, "layer type:"):
        mini_tile_hor = get_act_minitile_hor_ver(layernum)[0]
        return '{:05b}'.format(mini_tile_hor - 1)
    else:
        return return_value(5)

def get_Mini_tile_ver(layernum):
    '''
    获取Mini tile的高度
    '''
    if "conv" in find_type(layernum, "layer type:"):
        mini_tile_ver = get_act_minitile_hor_ver(layernum)[1]
        return '{:05b}'.format(mini_tile_ver - 1)
    else:
        return return_value(5)

def get_in_chl_num(layernum):
    '''
    获取任务块中需要计算的chl数据
    '''
    if "conv" in find_type(layernum, "layer type:"):
        act_input_channel = find_count(layernum, "out_channels:")
        act_input_channel = int(act_input_channel / nnbaton_Kp / nnbaton_K2)
        return '{:09b}'.format(act_input_channel - 1)
    else:
        return return_value(9)

def get_Out_mini_tile_hor(layernum):
    '''
    sub_core计算出来的mini tile的水平方向大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        out_hor = find_count(layernum, "feature_map_size_x:")
        mini_tile_hor = int(out_hor / nnbaton_X2 / nnbaton_Xc / nnbaton_X1)
        return '{:07b}'.format(mini_tile_hor - 1)
    else:
        return return_value(7)

def get_Out_mini_tile_ver(layernum):
    '''
    sub_core计算出来的mini tile的垂直方向大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        out_ver = find_count(layernum, "feature_map_size_y:")
        mini_tile_hor = int(out_ver / nnbaton_Yp / nnbaton_Y2 / nnbaton_Y1)
        return '{:07b}'.format(mini_tile_hor - 1)
    else:
        return return_value(7)

def get_Out_tile_hor(layernum):
    '''
    获取output tile的水平方向大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        out_W = find_count(layernum, "feature_map_size_x:")
        out_tile_hor = int(out_W / nnbaton_X2)
        return '{:07b}'.format(out_tile_hor - 1)
    else:
        return return_value(7)

def get_Out_tile_ver(layernum):
    '''
    获取output tile的垂直方向大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        out_H = find_count(layernum, "feature_map_size_y:")
        out_tail_ver = int(out_H / nnbaton_Yp / nnbaton_Y2)
        return '{:07b}'.format(out_tail_ver - 1)
    else:
        return return_value(7)

def get_Out_tile_chl(layernum):
    '''
    获取output tile的通道方向大小
    '''
    if "conv" in find_type(layernum, "layer type:"):
        out_C = find_count(layernum, "out_channels:")
        out_tile_chl = int(out_C / nnbaton_K2)
        return '{:07b}'.format(out_tile_chl - 1)
    else:
        return return_value(7)

def get_weight_total_size(layernum):
    '''
    获取一个output channel的weight大小
    in_channel和out_channel相等，不需要反推
    '''
    if "conv" in find_type(layernum, "layer type:"):
        output_channel = int(find_count(layernum, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        kernel_size = find_count(layernum, "kernel_size_x:") * find_count(layernum, "kernel_size_y:")
        weight_size = kernel_size * int(align(output_channel, 8) / 8)
        return '{:017b}'.format(weight_size)
    else:
        return return_value(17)

def get_Weight_sub_size(layernum):
    pass

def get_weight_ram_output_chl_num(layernum):
    '''
    获取weight ram可以存下output_channel/tile_mode的个数
    '''
    if "conv" in find_type(layernum, "layer type:"):
        output_channel = int(find_count(layernum, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        tile_mode = int(8 / (nnbaton_Xc * nnbaton_Yc))
        return '{:07b}'.format(int(output_channel / tile_mode) - 1)
    else:
        return return_value(7)

def get_conv_type(layernum):
    '''
    获取卷积类型
    00：普通卷积
    01：空洞卷积
    10：反卷积；
    11：保留
    '''
    if "conv" in find_type(layernum, "layer type:"):
        return '{:02b}'.format(0)
    else:
        return return_value(2)

def get_Chiplet_mode(layernum):
    '''
    获取chiplet的工作模式
    kp = 4时为10（四块chiplet），kp = 1时为00（chiplet各自工作）
    '''

    if nnbaton_Kp == 4:
        return '{:02b}'.format(0b10)
    elif nnbaton_Kp == 1:
        return '{:02b}'.format(0)

def get_Chiplet_tile_size(layernum):
    '''
    获取多个chiplet协同工作时，每块input channel对应的weight存储
    等于weight_total_size/chiplet_mode
    '''
    if "conv" in find_type(layernum, "layer type:"):
        output_channel = int(find_count(layernum, "out_channels:") / nnbaton_Kp / nnbaton_K2)
        kernel_size = find_count(layernum, "kernel_size_x:") * find_count(layernum, "kernel_size_y:")
        weight_size = kernel_size * int(align(output_channel, 8) / 8 / nnbaton_Kp)
        return '{:017b}'.format(weight_size)
    else:
        return return_value(17)

def get_Kernel_num(layernum):
    '''
    获取kernel宽度乘以高度的大小
    FC的kernel_size为1*1
    '''
    if "fc" in find_type(layernum, "layer type:"):
        kernel_num = 1
        return '{:08b}'.format(kernel_num - 1)
    else:
        kernel_width = find_count(layernum, "kernel_size_x:")
        kernel_height = find_count(layernum, "kernel_size_y:")
        kernel_num = kernel_height * kernel_width
        return '{:08b}'.format(kernel_num - 1)

def get_Kernel_width(layernum):
    '''
    获取kernel宽度的大小
    '''
    if "fc" in find_type(layernum, "layer type:"):
        Kernel_width = 1
        return '{:04b}'.format(Kernel_width - 1)
    else:
        Kernel_width = find_count(layernum, "kernel_size_y:")
        return '{:04b}'.format(Kernel_width - 1)

def get_Kernel_height(layernum):
    '''
    获取kernel高度的大小
    '''
    if "fc" in find_type(layernum, "layer type:"):
        kernel_height = 1
        return '{:04b}'.format(kernel_height - 1)
    else:
        kernel_height = find_count(layernum, "kernel_size_y:")
        return '{:04b}'.format(kernel_height - 1)

def get_Kernel_str(layernum):
    '''
    获取步长大小
    000：跨度为1；   001：跨度为2；   010：跨度为3；   011：跨度为4；
    100：跨度为5；   101：跨度为6；   110：跨度为7；   111：跨度为8；
    '''
    Kernel_str = find_count(layernum, "stride_x:")
    if Kernel_str == None:
        return return_value(3)
    else:
        return '{:03b}'.format(Kernel_str - 1)

def get_deconv_hor_str(layernum):
    '''
    获取反卷积在水平方向上的跨度
    '''
    return '{:02b}'.format(0)

def get_deconv_ver_str(layernum):
    '''
    获取反卷积在垂直方向上的跨度
    '''
    return '{:02b}'.format(0)

def get_Dilation_rate(layernum):
    '''
    获取反卷空洞卷积，插入零的个数
    '''
    return '{:03b}'.format(0)

def get_pooling_carry_sel(layernum):
    '''
    取整方式选择
    取整方式先按照1获取，后期在pt文件中获取
    '''
    return '{:01b}'.format(1)

def get_pool_mode(layernum):
    '''
    返回0：平均池化；(avgpooling)
    返回1：最大池化；(maxpooling)
    非pool返回0
    '''
    pool_mode = find_type(layernum, "layer type:")
    if "avgpool" in pool_mode:
        return '{:01b}'.format(0)
    elif "maxpool" in pool_mode:
        return '{:01b}'.format(1)
    else:
        return return_value(1)

def get_Pooling_quan_code_in(layernum):
    '''
    表示定点转32位浮点的量化系数（暂时先不管）
    '''
    pass

def get_pool_total_size(layernum):
    '''
    池化的尺寸乘积结果
    '''
    pool_total_size = find_count(layernum, "pool_size:")
    if pool_total_size == None:
        return return_value(10)
    else:
        return '{:010b}'.format(pool_total_size * pool_total_size - 1)

def get_pool_size(layernum):
    '''
    获取池化的size
    '''
    pool_size = find_count(layernum, "pool_size:")
    if pool_size == None:
        return return_value(5)
    else:
        return '{:05b}'.format(pool_size)

def get_Pooling_oprands(layernum, types):
    '''
    给平均池化使用的任意尺寸的倒数
    '''
    types = find_type(layernum, "layer type:")
    if "avgpool" in types:
        pool_size = find_count(layernum, "pool_size:")
        return '{:032b}'.format(int(st.unpack('I', st.pack('f', 1 / (pool_size * pool_size)))[0]))
    else:
        return return_value(32)

def get_Pooling_quan_code_out(layernum):
    '''
    表示32位浮点转定点的量化系数（暂时不管）
    '''
    pass

def get_scaling_mode(layernum):
    '''
    scaling放大倍数:
    00：bilinearx2   01：bilinearx4
    10：nearestx2    11: nearestx4
    '''
    pass

def get_fc_mode_en(layernum):
    '''
    全连接模式使能信号，返回1有效
    '''
    fc_mode = find_type(layernum, "layer type:")
    if "fc" in fc_mode:
        return '{:01b}'.format(1)
    else:
        return return_value(1)

def get_Inst_id_01():
    '''
    获取当前指令的ID
    '''
    return '{:02b}'.format(0b01)

def get_Tile_mode():
    '''
    获取各个sub core之间拼tile的模式
    '''
    tile_mode = nnbaton_Xc * nnbaton_Yc
    if tile_mode == 1:
        return '{:02b}'.format(0b00)
    elif tile_mode == 2:
        return '{:02b}'.format(0b01)
    elif tile_mode == 4:
        return '{:02b}'.format(0b10)
    elif tile_mode == 8:
        return '{:02b}'.format(0b11)

def get_tile_num(layernum):
    '''
    获取当前指令中，需要计算的tile的个数
    '''
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

def get_mini_ver_num(layernum):
    '''
    获取一个sub_tile中水平方向上mini_tile的数量
    '''
    return '{:07b}'.format(nnbaton_X1)

def get_mini_hor_num(layernum):
    '''
    获取一个sub_tile中垂直方向上mini_tile的数量
    '''
    return '{:07b}'.format(nnbaton_Y1)

def get_Inst_id_11():
    '''
    获取当前指令的ID
    '''
    return '{:02b}'.format(0b11)

def get_Run_mode(layernum, types):
    '''
    获取当前执行计算模式：
    00：卷积模式；    01：pooling模式；
    10：element wise模式；  11：scaling模式；
    '''
    if "conv" in types:
        return '{:02b}'.format(0b00)
    elif "pooling" in types:
        return '{:02b}'.format(0b01)
    elif "element wise" in types:
        return '{:02b}'.format(0b10)
    elif "scaling" in types:
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

def get_act_addr(layernum):
    '''
    从存储器读取每一个tile的数据存储起始地址
    '''
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
    elif layernum in pool_num:
        return 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    elif layernum == layercnts - 2 or \
            layernum == layercnts - 1 or \
            layernum == layercnts:
        return 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

def get_layer_inst(insts, types, layernum, tile_cnt, act_layer_addr, wet_layer_addr):
    fmain.write(f'{act_layer_addr}\n')
    if len(wet_layer_addr):
        fmain.write(f'{wet_layer_addr}\n')
    for i in range(1, tile_cnt + 1):
        act_start = act_layer_addr[1][0]
        act_end = act_layer_addr[1][1]
        if len(wet_layer_addr):
            wet_start = wet_layer_addr[1][0]
            wet_end = wet_layer_addr[1][1]
            inst_00, inst_01, inst_11 = get_tile_inst(layernum, types, i, act_start, act_end, wet_start, wet_end)
        else:
            inst_00, inst_01, inst_11 = get_tile_inst(layernum, types, i, act_start, act_end, "", "")
        insts[i-1] = [f'layer{layernum} tile{i} instruction', [inst_00], [inst_01], [inst_11]]
        insts.append(insts[i-1])

    return insts

def refresh_nnbaton(X1, Y1, K1, X2, Y2, K2, Kp, Yp, Kc, Yc, Xc, C1, C0, X0, Y0):
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

def get_tile_inst(layer_num, types, tile_num, act_start, act_end, wet_start, wet_end):
    fmain.write("get_tile_inst layer%2s %2s %8s %8s %8s %8s\n" %
        (layer_num, tile_num, act_start, act_end, wet_start, wet_end))

    fc_mode_en = get_fc_mode_en(layer_num)
    scaling_mode = get_scaling_mode(layer_num)
    Pooling_quan_code_out = get_Pooling_quan_code_out(layer_num)
    Pooling_oprands = get_Pooling_oprands(layer_num, types)
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
    Chiplet_mode = get_Chiplet_mode(layer_num)
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
    act_tile_ver = get_act_tile_ver(layer_num)
    act_tile_hor = get_act_tile_hor(layer_num)
    Inst_id_00 = get_Inst_id_00()

    mini_hor_num = get_mini_hor_num(layer_num)
    mini_ver_num = get_mini_ver_num(layer_num)
    inst_invalid = get_inst_invalid(layer_num)
    act_chl_one_inst = get_act_chl_one_inst(layer_num)
    act_sub_ch = get_act_sub_ch(layer_num)
    act_chl_one_inst_real = get_act_chl_one_inst_real(layer_num)
    weight_inst_bypass = get_weight_inst_bypass(layer_num)
    act_inst_bypass = get_act_inst_bypass(layer_num)
    Repeat_send_num = get_Repeat_send_num(layer_num)
    Padding_num = get_Padding_num(layer_num)
    Padding_mode = get_Padding_mode(layer_num)
    tile_num = get_tile_num(layer_num)
    Tile_mode = get_Tile_mode()
    Inst_id_01 = get_Inst_id_01()

    act_addr_element = get_act_addr_element(layer_num)
    Out_feature_map_ver = get_Out_feature_map_ver(layer_num)
    Out_tile_stride = get_Out_tile_stride(layer_num)
    Out_tile_start_addr = get_Out_tile_start_addr(layer_num)
    LLC_a_ping_pong = get_LLC_a_ping_pong(layer_num)
    act_updata_n = get_act_updata_n(layer_num)
    act_str_line = get_act_str_line(layer_num)
    act_str_chl = get_act_str_chl(layer_num)
    act_addr = get_act_addr(layer_num)
    LLC_w_ping_pong = get_LLC_w_ping_pong(layer_num)
    weight_updata_n = get_weight_updata_n(layer_num)
    weight_output_chl = get_weight_output_chl(layer_num)
    weight_addr = get_weight_addr(layer_num)
    Run_mode = get_Run_mode(layer_num, types)
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
               ["Repeat_send_num", Repeat_send_num], ["Repeat_send_num", Repeat_send_num],
               ["Padding_num", Padding_num], ["Padding_mode", Padding_mode],
               ["tile_num", tile_num], ["Tile_mode", Tile_mode],
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
