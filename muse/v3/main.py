import os
import sys
import time
from script.run_steps import *
from script.insts_fun import *
from script.gen_files import *
from script.inout_bin import *
from script.calcu_addr import *
from script.load_pt import load_pt


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

    name_list = []
    data_list = []
    datas_locate = ["", []]
    layers_act_addr = ["", []]
    layers_wet_addr = ["", []]
    del layers_act_addr[0]
    del layers_wet_addr[0]

    parase = muse_parse()
    ptpath = parase.pt_name
    netpath = parase.net_name
    confpath = parase.conf_dir
    ptdtpath = parase.ptdata_dir
    bmpdtpath = parase.bmpdata_dir
    outputpath = parase.output_dir

    make_bin = ["Make the bin file needed by fpga...", "gen_fpga(f'{os.getcwd()}/imagenet')"]
    logname = "vggnet.log"
    if "resnet" in netpath.lower():
        logname = "resnet.log"
        make_bin = ["gen_address for DRAM...", "gen_ddraddr(name_list, data_list, netpath)"]

    debug = f'{os.getcwd()}/.debug/'
    logpath = f'{debug}/{logname}'
    mainlog = f'{debug}/main.log'
    info = f'{os.getcwd()}/.info'

    fileslist = ["config_gen_file.py", "inout_print.py", "quant_layer.py", "quantop.py",
                 "im6.bmp", ptpath, netpath]
    cleanlist = ["data_for_fpga",      "imagenet",        logname,          outputpath,
                  confpath, ptdtpath]

    if len(sys.argv) < 2 or sys.argv[1] != "clean":
        wr_flag = True
    elif len(sys.argv) == 2 and sys.argv[1] == "clean":
        wr_flag = False
        clean_ups(cleanlist, os.getcwd(), wr_flag, info, debug)
        if os.path.exists(info):
            os.remove(info)
        exit(0)

    if os.path.exists(debug):
        os.system(f'rm -rf {debug}')
    os.mkdir(debug)

    run_step1 = [["Clean the necessary files...", "clean_ups(cleanlist, os.getcwd(), wr_flag, info, debug)"],
                 ["Check the necessary files...", "checkfile(fileslist)"],
                 ["Run the network model file...", f'os.system(\'python3 {netpath} > {logpath}\')',
                  f'prints(\'run {netpath} successfully\')'],
                 ["Load pt file and format output...", "os.system(f'mkdir -p {ptdtpath} {outputpath}')",
                  "loadpt = load_pt(fmain, confpath, ptdtpath, netpath, ptpath, logpath)",
                  f'loadpt.net_name = \'{logname.split(".")[0]}\'',
                  "name_list, data_list = loadpt.gen_txt()"],
                 ["Generate input and output from bmp...", 
                  "send_genfilevar(ptpath, logpath, netpath, bmpdtpath, confpath, fmain, ptdtpath, outputpath)",
                 f'gen_bmp(\'{logpath}\')'],
                 ["gen_address for DRAM...",
                  "send_calcuaddrvar(fmain, logpath, loadpt.layer_cnts)", 
                  "layers_act_addr, layers_wet_addr, datas_locate = gen_ddraddr(name_list, data_list, netpath)"],
                  make_bin]

    run_step2 = [["calculate the size of all bin files...", "binary_addr(instdir, datadir, outputpath)"],
                 ["generate input and output bin...",
                  "gen_inout_bin(outputpath, bmpdtpath, netpath, pool_num, name_list, data_list)",
                  f'prints(\'run gen_inout_bin successfully\')']]

    step_cnt = len(run_step1) + len(run_step2) + 1

    with open(mainlog, 'w') as fmain:
        send_runstepvar(fmain)

        for i in range(len(run_step1)):
            start = time.time()
            outputs(i + 1, step_cnt, run_step1[i][0])
            for j in range(1, len(run_step1[i])):
                s = run_step1[i][j]
                r = compile(s, "<string>", "exec")
                exec(r)
            end = time.time()
            schedule(i + 1, start, end)

        start = time.time()
        # insts是一层的inst(包含n个tile指令)insts[i]
        # 均由inst_00 inst_01 inst_11组成的tile指令
        send_instsfunvar(fmain, logpath, loadpt.layer_cnts)
        send_inoutbinvar(fmain, loadpt.layer_cnts)
        insts = ["", [[], [], []]]
        j = tile_cnt = 0
        del insts[0]
        types = ""
        i = 1
        inst_00 = [["fc_mode_en", get_fc_mode_en(i)], ["scaling_mode", get_scaling_mode(i)],
                   ["Pooling_quan_code_out", get_Pooling_quan_code_out(i)],
                   ["Pooling_oprands", get_Pooling_oprands(i, types)],
                   ["pool_size", get_pool_size(i)], ["pool_total_size", get_pool_total_size(i)],
                   ["Pooling_quan_code_in", get_Pooling_quan_code_in(i)],
                   ["pool_mode", get_pool_mode(i)], ["pooling_carry_sel", get_pooling_carry_sel(i)],
                   ["Dilation_rate", get_Dilation_rate(i)], ["deconv_ver_str", get_deconv_ver_str(i)],
                   ["deconv_hor_str", get_deconv_hor_str(i)], ["Kernel_str", get_Kernel_str(i)],
                   ["Kernel_height", get_Kernel_height(i)], ["Kernel_width", get_Kernel_width(i)],
                   ["Kernel_num", get_Kernel_num(i)], ["Chiplet_tile_size", get_Chiplet_tile_size(i)],
                   ["Chiplet_mode  ", get_Chiplet_mode(i)], ["conv_type", get_conv_type(i)],
                   ["weight_ram_output_chl_num", get_weight_ram_output_chl_num(i)],
                   ["Weight_sub_size", get_Weight_sub_size(i)], ["weight_total_size", get_weight_total_size(i)],
                   ["Out_tile_chl", get_Out_tile_chl(i)], ["Out_tile_ver", get_Out_tile_ver(i)],
                   ["Out_tile_hor", get_Out_tile_hor(i)], ["Out_mini_tile_ver", get_Out_mini_tile_ver(i)],
                   ["Out_mini_tile_hor", get_Out_mini_tile_hor(i)], ["in_chl_num", get_in_chl_num(i)],
                   ["Mini_tile_ver", get_Mini_tile_ver(i)], ["Mini_tile_hor", get_Mini_tile_hor(i)],
                   ["out_chl_num", get_out_chl_num(i)], ["sub_tile_ver", get_sub_tile_ver(i)],
                   ["sub_tile_hor", get_sub_tile_hor(i)], ["act_tile_sub_chl", get_act_tile_sub_chl(i)],
                   ["act_tile_str", get_act_tile_str(i)], ["act_tile_chl", get_act_tile_chl(i)],
                   ["act_tile_ver", get_act_tile_ver(i)], ["act_tile_hor", get_act_tile_hor(i)],
                   ["Inst_id_00", get_Inst_id_00()]]
        inst_01 = [["mini_hor_num", get_mini_hor_num(i)], ["mini_ver_num", get_mini_ver_num(i)],
                   ["inst_invalid", get_inst_invalid(i)], ["act_chl_one_inst", get_act_chl_one_inst(i)],
                   ["act_sub_ch", get_act_sub_ch(i)], ["act_chl_one_inst_real", get_act_chl_one_inst_real(i)],
                   ["weight_inst_bypass", get_weight_inst_bypass(i)], ["act_inst_bypass", get_act_inst_bypass(i)],
                   ["Repeat_send_num", get_Repeat_send_num(i)], ["Repeat_send_num", get_Repeat_send_num(i)],
                   ["Padding_num", get_Padding_num(i)], ["Padding_mode", get_Padding_mode(i)],
                   ["tile_num", get_tile_num(i)], ["Tile_mode", get_Tile_mode()],
                   ["Inst_id_01", get_Inst_id_01()]]
        inst_11 = [["act_addr_element", get_act_addr_element(i)], ["Out_feature_map_ver", get_Out_feature_map_ver(i)],
                   ["Out_tile_stride", get_Out_tile_stride(i)], ["Out_tile_start_addr", get_Out_tile_start_addr(i)],
                   ["LLC_a_ping_pong", get_LLC_a_ping_pong(i)], ["act_updata_n", get_act_updata_n(i)],
                   ["act_str_line", get_act_str_line(i)], ["act_str_chl", get_act_str_chl(i)],
                   ["act_addr", get_act_addr(i)],  ["LLC_w_ping_pong", get_LLC_w_ping_pong(i)],
                   ["weight_updata_n", get_weight_updata_n(i)],  ["weight_output_chl", get_weight_output_chl(i)],
                   ["weight_addr", get_weight_addr(i)], ["Run_mode", get_Run_mode(i, types)],
                   ["Inst_id_11", get_Inst_id_11()]]

        datadir = f'{outputpath}/datas/'
        instdir = f'{outputpath}/insts/'
        os.mkdir(instdir)
        os.mkdir(datadir)
        outputs(len(run_step1) + 1, step_cnt, "generate inst and data bin...")

        pool_num = get_layer_num("pool")
        for i in range(1, loadpt.layer_cnts + 1):
            types = find_type(i, "layer type:")
            nnbaton_X1, nnbaton_Y1, nnbaton_K1, nnbaton_X2, nnbaton_Y2, \
            nnbaton_K2, nnbaton_Kp, nnbaton_Yp, nnbaton_Kc, nnbaton_Yc, \
            nnbaton_Xc, nnbaton_C1, nnbaton_C0, nnbaton_X0, nnbaton_Y0 \
                = get_nnbaton(i, pool_num)
            refresh_nnbaton(nnbaton_X1, nnbaton_Y1, nnbaton_K1, nnbaton_X2,
                            nnbaton_Y2, nnbaton_K2, nnbaton_Kp, nnbaton_Yp,
                            nnbaton_Kc, nnbaton_Yc, nnbaton_Xc, nnbaton_C1,
                            nnbaton_C0, nnbaton_X0, nnbaton_Y0)
            tile_cnt = (nnbaton_Kp * nnbaton_Yp) * (nnbaton_K2 * nnbaton_Y2 * nnbaton_X2)
            act_layer_addr = get_layer_addr(i, layers_act_addr)
            if i not in pool_num:
                wet_layer_addr = get_layer_addr(i, layers_wet_addr)
                insts = get_layer_inst(insts, types, i, tile_cnt, act_layer_addr, wet_layer_addr)
            else:
                insts = get_layer_inst(insts, types, i, tile_cnt, act_layer_addr, [])
            inst_name = f'{instdir}/layer.{i}.inst.bin'
            data_name = f'{datadir}/layer.{i}.data.bin'
            with open(inst_name, 'wb') as finst:
                write_insts(finst, tile_cnt, insts)
                if i not in pool_num:
                    with open(data_name, 'wb') as fw:
                        write_datas(datas_locate[j], fw)
                    j += 1
            fmain.write("write layer%-2s datas and insts successfully\n" % i)
            insts = ["", [[], [], []]]
            del insts[0]

        end = time.time()
        schedule(len(run_step1) + 1, start, end)

        for i in range(len(run_step2)):
            start = time.time()
            outputs(len(run_step1) + 1 + i + 1, step_cnt, run_step2[i][0])
            for j in range(1, len(run_step2[i])):
                s = run_step2[i][j]
                r = compile(s, "<string>", "exec")
                exec(r)
            end = time.time()
            schedule(len(run_step1) + 1 + i + 1, start, end)

        allend = time.time()
        prints('Make data successfully! It costs %.2f Seconds' % (allend - allstart))

        save_params(parase, info)
