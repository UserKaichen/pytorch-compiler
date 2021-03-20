import os
import re
import sys
import math
import threading
from input.config_gen_file import *
from script.calcu_addr import get_fstin_CHW, get_out_CHW

fmain = 0
layer_cnts = 0
BUS_WIDTH = int(256 / 8)

def send_inoutbinvar(fw, cnt):
    global fmain
    global layer_cnts

    fmain = fw
    layer_cnts = cnt

def write_inout_bin(txtfile, binfile, H, W, C, in_q):
    gen_act_file(txtfile, "", binfile, H, W, C, in_q, 1, 0, 0, 1)
    fmain.write(f'{binfile} generate success\n')

def gen_inout_bin(output, bmpdt, net, pool_num, name_list, data_list):
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

    bindir = f'{output}/inout'
    os.mkdir(bindir)

    for i in range(len(name_list)):
        name = name_list[i]
        if ".alpha" in name:
            alphalist.append(data_list[i].tolist())

    j = 0
    bit = 7
    in_q = 5
    C_in, H_in, W_in = get_fstin_CHW(net)
    C_out, H_out, W_out = get_out_CHW(1)

    for i in range(1, layer_cnts - len(fc_inputs) + 1):
        if i > 1:
            C_out, H_out, W_out = get_out_CHW(i)
        if i in pool_num:
            out_q = in_q
            C_in, H_in, W_in = get_out_CHW(i)
            continue
        out_q = bit - math.ceil(math.log2(0.5 * alphalist[j]))
        input_txt = f'{bmpdt}/{conv_inputs[j]}'
        output_txt = f'{bmpdt}/{conv_output[j]}'
        input_bin = f'{bindir}/layer{i-1}_input_act0_file.bin'
        output_bin = f'{bindir}/layer{i-1}_output_file.bin'
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
        C_in = C_out
        H_in = H_out
        W_in = W_out

    for i in range(len(fc_inputs)):
        layernum = i + len(conv_inputs) + len(pool_num) + 1
        if layernum == layer_cnts:
            C_in, H_in, W_in, C_out = get_out_CHW(layernum)
            C_out, H_out, W_out, C_out = get_out_CHW(layernum)
        else:
            C_in, H_in, W_in = get_out_CHW(layernum)
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

def write_tile_inst(inst_256, insts):
    for i in range(len(insts)):
        for j in range(len(insts[i])):
            if insts[i][j][1] != None:
                inst_256 = '{}{}'.format(inst_256, insts[i][j][1])
    return inst_256

def fillings(c, cnt):
    inst_zero = ""
    for i in range(cnt):
        inst_zero = '{}{}'.format(inst_zero, c)
    return inst_zero

def write_insts(fw, tile_cnt, insts):
    for i in range(tile_cnt):
        fmain.write(f'write_insts {insts[i][0]}...\n')
        print(f'write_insts {insts[i][0]}...')
        for j in range(1, len(insts[i])):
            if j == 1:
                cnt = 8
                inst_name = "inst_id=2'h00"
            elif j == 2:
                cnt = 192
                inst_name = "inst_id=2'h01"
            elif j == 3:
                cnt = 5
                inst_name = "inst_id=2'h11"
            inst_256 = fillings('0', cnt)
            inst_256 = write_tile_inst(inst_256, insts[i][j])

            for k in range(BUS_WIDTH*8 - len(inst_256)):
                inst_256 = '{}{}'.format(inst_256, '0')

            for k in range(int(BUS_WIDTH*8/4)):
                dec = int(inst_256[k*4:(k+1)*4], 2)
                # print(f'inst_256[{k*4}:{(k+1)*4-1}]={inst_256[k*4:(k+1)*4]}')
                fw.write(st.pack('B', dec))
            fmain.write(f'{insts[i][0]} {inst_name} write data success\n')
            print(f'{insts[i][0]} {inst_name} write data success')

def write_datas(data_locate, fw):
    for i in range(len(data_locate)):
        if i == 0:
            fmain.write(f'write_datas show data_locate: {data_locate[i]} detail\n')
            continue
        for j in range(len(data_locate[i])):
            for k in range(len(data_locate[i][j])):
                ints = int(data_locate[i][j][k], 16)
                fw.write(st.pack('B', ints))
