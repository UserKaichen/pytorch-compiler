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

    parase,mkbin,logname = muse_parse()
    ptpath = parase.pt_name
    netpath = parase.net_name
    confpath = parase.conf_dir
    ptdtpath = parase.ptdata_dir
    bmpdtpath = parase.bmpdata_dir
    outputpath = parase.output_dir

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
    if len(sys.argv) == 2:
        if sys.argv[1] == "clean":
            wr_flag = False
            clean_ups(cleanlist, os.getcwd(), wr_flag, info, debug)
            if os.path.exists(info):
                os.remove(info)
            exit(0)
        elif sys.argv[1] == "debug":
            netpath = run_debug(os.getcwd())
            logpath = f'{debug}/simulator.log'
        else:
            print(f'Unknown {sys.argv[1]} in toolchains.')

    if os.path.exists(debug):
        os.system(f'rm -rf {debug}')
    os.mkdir(debug)

    run_step1 = [
    ["Clean the necessary files...", "clean_ups(cleanlist, os.getcwd(), wr_flag, info, debug)"],
    ["Check the necessary files...", "checkfile(fileslist)"],
    ["Run the network model file...", f'os.system(\'python3 {netpath} > {logpath}\')',
    f'prints(\'run {netpath} successfully\')'],
    ["Load pt file and format output...", "os.system(f'mkdir -p {outputpath}')",
     "loadpt = load_pt(fmain, confpath, ptdtpath, netpath, ptpath, logpath)",
    f'loadpt.net_name = \'{logname.split(".")[0]}\'', 
     "name_list, data_list, actbit, wetbit = loadpt.gen_txt()"],
    ["Generate input and output from bmp...",
     "send_genfilevar(ptpath, logpath, netpath, bmpdtpath, confpath, fmain, ptdtpath, outputpath)",
    f'gen_bmp(\'{logpath}\')'],
    ["gen_address for DRAM...", "send_calcuaddrvar(fmain, logpath, netpath, loadpt.layer_cnts)",
    "layers_act_addr,layers_wet_addr,datas_locate,pool,downs=gen_ddraddr(name_list,data_list,actbit,wetbit)"],
    ]
    run_step2 = [
    ["calculate the size of all bin files...", "binary_addr(instdir, datadir, outputpath)"],
    ["generate input and output bin...",
    f'{mkbin}(outputpath, bmpdtpath, netpath, pool, downs, name_list, data_list)',
    f'prints(\'run gen_inout_bin successfully\')']]

    step_cnt = len(run_step1) + len(run_step2) + 1

    with open(mainlog, 'w') as fmain:
        send_runstepvar(fmain)
        send_instsfunvar(fmain, logpath)

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

        send_inoutbinvar(fmain, loadpt.layer_cnts)
        insts = ["", [[], [], []]]
        j = tile_cnt = 0
        del insts[0]
        types = ""

        inst_00 = ["inst_00", 39]
        inst_01 = ["inst_01", 14]
        inst_11 = ["inst_11", 15]
        insts = [inst_00, inst_01, inst_11]

        for i in range(len(insts)):
            for k in range(insts[i][1]):
                fuzhi = f'{insts[i][0]}.append(["", pass_fun()])'
                r = compile(fuzhi, "<string>", "exec")
                exec(r)

        inst_00 = inst_00[2:]
        inst_01 = inst_01[2:]
        inst_11 = inst_11[2:]

        datadir = f'{outputpath}/datas/'
        instdir = f'{outputpath}/insts/'
        os.mkdir(instdir)
        os.mkdir(datadir)
        outputs(len(run_step1) + 1, step_cnt, "generate inst and data bin...")

        send_instsbit(actbit, wetbit, netpath)
        act_addr = get_layer_input_addr(layers_act_addr, loadpt.layer_cnts)
        out_addr = get_layer_outpu_addr(layers_act_addr)
        for i in range(1, loadpt.layer_cnts + 1):
            nnbaton_X1, nnbaton_Y1, nnbaton_K1, nnbaton_X2, nnbaton_Y2, \
            nnbaton_K2, nnbaton_Kp, nnbaton_Yp, nnbaton_Kc, nnbaton_Yc, \
            nnbaton_Xc, nnbaton_C1, nnbaton_C0, nnbaton_X0, nnbaton_Y0, \
            nnbaton_divcase = get_nnbaton(i)
            refresh_nnbaton(nnbaton_X1, nnbaton_Y1, nnbaton_K1, nnbaton_X2,
                            nnbaton_Y2, nnbaton_K2, nnbaton_Kp, nnbaton_Yp,
                            nnbaton_Kc, nnbaton_Yc, nnbaton_Xc, nnbaton_C1,
                            nnbaton_C0, nnbaton_X0, nnbaton_Y0, nnbaton_divcase)
            tile_cnt = (nnbaton_Kp * nnbaton_Yp) * (nnbaton_K2 * nnbaton_Y2 * nnbaton_X2)
            if i not in pool:
                wet_addr = get_layer_addr(i, layers_wet_addr)
                insts = get_layer_inst(insts, i, tile_cnt, act_addr[i-1][0], out_addr[i-1][0], wet_addr)
            else:
                insts = get_layer_inst(insts, i, tile_cnt, act_addr[i-1][0], out_addr[i-1][0], [])
            inst_name = f'{instdir}/layer.{i-1}.inst.bin'
            data_name = f'{datadir}/layer.{i-1}.data.bin'
            with open(inst_name, 'wb') as finst:
                write_insts(finst, tile_cnt, insts)
                if i not in pool:
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
