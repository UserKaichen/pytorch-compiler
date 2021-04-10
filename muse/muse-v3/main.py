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
    weights = ["", []]
    lys_act = ["", []]
    lys_wet = ["", []]
    del lys_act[0]
    del lys_wet[0]

    # Toolchains for parameter analysis
    parase,mkbin,logname = muse_parse()
    ptpath = parase.pt_name
    netpath = parase.net_name
    confpath = parase.conf_dir
    ptdtpath = parase.ptdata_dir
    bmpdtpath = parase.bmpdata_dir
    outputpath = parase.output_dir

    pool = []
    debug = ".debug"
    actbit = wetbit = 0
    laycnts = chipid = 0
    logpath = f'{debug}/{logname}'
    mainlog = f'{debug}/main.log'
    info = f'{os.getcwd()}/.info'

    fileslist = ["config_gen_file.py", "inout_print.py", "quant_layer.py", "quantop.py",
                 "im6.bmp", ptpath, netpath]
    cleanlist = ["imagenet",     logname,     outputpath,     confpath,     ptdtpath]

    # Select the operating mode (normal or debug mode)
    if len(sys.argv) < 2 or sys.argv[1] != 'clean':
        wr_flag = True
    if len(sys.argv) == 2:
        if sys.argv[1] == 'clean':
            wr_flag = False
            clean_ups(cleanlist, os.getcwd(), wr_flag, info, debug)
            if os.path.exists(info):
                os.remove(info)
            exit(0)
        elif sys.argv[1] == 'debug':
            netpath = run_debug(os.getcwd())
            logpath = f'{debug}/simulator.log'
        else:
            print(f'Unknown {sys.argv[1]} in toolchains.')
            exit(0)

    # Detection toolchains operating platform
    substep = 'linux'
    if sys.platform == 'linux':
        substep = "os.system(f'python3 {netpath} >{logpath} 2>/dev/null')"
    elif sys.platform == 'win32':
        substep = "os.system(f'copy imagenet\\{logname} {debug}')"
    else:
        print(f'{sys.platform} muse-v3 toolchains not adapted. Stay tuned')
        exit(0)

    if os.path.exists(debug):
        shutil.rmtree(debug)
    os.mkdir(debug)

    # Toolchains define operation steps
    run_step1 = [
    ["Clean the necessary files...",
     "clean_ups(cleanlist, os.getcwd(), wr_flag, info, debug)"],
    ["Check the necessary files...", "checkfile(fileslist)"],
    ["Run the network model file...", substep,
    f'prints(\'run {netpath} successfully\')'],
    ["Load pt file and format output...", "os.mkdir(outputpath)",
     "loadpt = load_pt(fmain, confpath, ptdtpath, netpath, ptpath, logpath)",
    f'loadpt.net_name = \'{logname.split(".")[0]}\'',
    f'laycnts = loadpt.layer_cnts',
     "name_list, data_list, actbit, wetbit, chipid = loadpt.gen_txt()"],
    ["Generate input and output from bmp...",
     "send_genfilevar(ptpath,logpath,netpath,bmpdtpath,confpath,fmain,ptdtpath,outputpath)",
    f'gen_bmp(laycnts, \'{logpath}\')'],
    ["gen_address for DRAM...", "send_calcuaddrvar(fmain, logpath, netpath, laycnts, chipid)",
    "lys_act,lys_wet,weights,pool,downs=gen_ddraddr(name_list,data_list,actbit,wetbit)"]]
    run_step2 = [
    ["calculate the size of all bin files...", "binary_addr(instdir, datadir, outputpath)"],
    ["generate input and output bin...",
    f'{mkbin}(inoudir, bmpdtpath, netpath, pool, downs, name_list, data_list)',
    f'prints(\'run gen_inout_bin successfully\')']]

    # Toolchains run step1
    step_cnt = len(run_step1) + len(run_step2) + 1
    with open(mainlog, 'w') as fmain:
        send_runstepvar(fmain)
        send_instsfunvar(fmain, logpath)

        for i in range(len(run_step1)):
            start = time.time()
            outputs(i + 1, step_cnt, run_step1[i][0])
            for j in range(1, len(run_step1[i])):
                s = run_step1[i][j]
                exec(compile(s, '<string>', 'exec'))
            end = time.time()
            schedule(i + 1, start, end)

        start = time.time()
        send_inoutbinvar(fmain, laycnts)
        insts = ["", [[], [], []]]
        j = tile_cnt = 0
        del insts[0]
        types = ""

        # Initialize instruction
        inst_00 = ['inst_00', 39]
        inst_01 = ['inst_01', 14]
        inst_11 = ['inst_11', 15]
        insts = [inst_00, inst_01, inst_11]

        for i in range(len(insts)):
            for k in range(insts[i][1]):
                assign = f'{insts[i][0]}.append(["", pass_fun()])'
                exec(compile(assign, '<string>', 'exec'))

        inst_00 = inst_00[2:]
        inst_01 = inst_01[2:]
        inst_11 = inst_11[2:]

        # Create output directory
        datadir = f'{outputpath}/datas/'
        instdir = f'{outputpath}/insts/'
        confdir = f'{outputpath}/confs/'
        inoudir = f'{outputpath}/inout/'

        outname = [datadir, instdir, confdir, inoudir]
        for i in range(len(outname)):
            os.mkdir(outname[i])

        outputs(len(run_step1)+1, step_cnt, "generate inst and data bin...")
        act_addr = get_layer_input_addr(lys_act, laycnts)
        out_addr = get_layer_outpu_addr(lys_act)
        num_type = get_layernum_to_type(logpath)
        send_instsbit(actbit, wetbit, netpath)
        nnbatons = get_nnbaton(num_type)
        netname = logname.split('.')[0]
        act_addr.insert(0, [])
        out_addr.insert(0, [])

        # Write instructions and data of each layer
        for i in range(1, laycnts + 1):
            baton = nnbatons[i-1]
            refresh_nnbaton(baton)
            tile_cnt = (baton[3] * baton[4] * baton[5]) * (baton[6] * baton[7])
            if i not in pool:
                wet_addr = get_layer_addr(i, lys_wet)
                insts = get_layer_inst(insts, i, tile_cnt, act_addr[i][0], out_addr[i][0], wet_addr)
            else:
                insts = get_layer_inst(insts, i, tile_cnt, act_addr[i][0], out_addr[i][0], [])
            inst_name = f'{instdir}/layer{i}_inst.bin'
            data_name = f'{datadir}/layer{i}_data.bin'
            conf_name = f'{confdir}/config_{i}.txt'
            with open(inst_name, 'wb') as finst:
                write_inst(finst, tile_cnt, insts, wetbit)
                if i not in pool:
                    with open(data_name, 'wb') as fdata:
                        write_data(weights[j], fdata)
                    j += 1
            with open(conf_name, 'w') as fconf:
                lenread = int(os.path.getsize(inst_name)/BUS_WIDTH)
                write_conf(num_type[i], actbit, wetbit, chipid, tile_cnt, fconf, netname, lenread)
            fmain.write("write layer%-2s datas, insts and confs successfully\n" % i)
            insts = ["", [[], [], []]]
            del insts[0]

        end = time.time()
        schedule(len(run_step1) + 1, start, end)

        # Toolchains run step2
        for i in range(len(run_step2)):
            start = time.time()
            outputs(len(run_step1) + 1 + i + 1, step_cnt, run_step2[i][0])
            for j in range(1, len(run_step2[i])):
                s = run_step2[i][j]
                exec(compile(s, '<string>', 'exec'))
            end = time.time()
            schedule(len(run_step1) + 1 + i + 1, start, end)

        allend = time.time()
        prints('Make data successfully! It costs %.2f Seconds' % (allend - allstart))

        save_params(parase, info)
