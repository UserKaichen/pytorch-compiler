import os
import threading
from script.run_steps import prints

fmain = logpath = 0
netpath = ptpath = 0
confpath = ptdtpath = 0
bmpdtpath = outputpath = 0

def send_genfilevar(pt, log, net, bmpd, conf, fw, ptdt, output):
    global fmain
    global ptpath
    global netpath
    global logpath
    global confpath
    global ptdtpath
    global bmpdtpath
    global outputpath

    fmain = fw
    ptpath = pt
    netpath = net
    logpath = log
    confpath = conf
    ptdtpath = ptdt
    bmpdtpath = bmpd
    outputpath = output

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
    cmd_list = ["rm -rf con*txt cfg*txt *bn* *bias* *alpha* *weight* *input* *output* \
                  *k* data_for_fpga",
                f'cp -af {confpath}/* {ptdtpath}/* {bmpdtpath}/* imagenet/',
                "python ../input/config_gen_file.py -d ../imagenet/ -n imagenet_img6 -f > ../.debug/genfile.log",
                f'mv {filepath}/data_for_fpga {outputpath}']

    for i in range(len(cmd_list)):
        if "cp -af " in cmd_list[i] or "mv " in cmd_list[i]:
            os.chdir("..")
        elif "python" in cmd_list[i]:
            os.chdir(filepath)
        os.system(cmd_list[i])

    prints("run gen_fpga successfully")

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
    fc_k_list = []
    data_list = []
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

    fc_k_list = sorted(fc_k_list)

    for i in range(len(fc_k_list)):
        with open(f'{bmpdtpath}/{fc_k_list[i]}', 'r') as fd:
            line = fd.readline()
            data_list.append(line)

    for i in range(len(fc_k_list)):
        with open(f'{bmpdtpath}/{fc_k_list[i]}', 'w') as fw:
            write_fc_k = threading.Thread(target=write_fck_data,
                                          args=(fw, data_list[i], int(fc_channel[i])))
            write_fc_k.start()
            write_fc_k.join()
        fmain.write(f'write {data_list[i].strip()} to {fc_k_list[i]} file {fc_channel[i]} times success\n')

def gen_bmp(vgglog):
    """
    description:
                Generate model files
    parameters:
                vgglog: Path to vgglog file
    return code:
                None
    """
    cmd_list = [f'python3 input/inout_print.py input/im6.bmp {ptpath} {bmpdtpath} > .debug/genbmp.log']

    for i in range(len(cmd_list)):
        os.system(cmd_list[i])

    deal_fc_k(vgglog)
    prints("run gen_bmp successfully")

def align(address, factor):
    if address % factor == 0:
        return address
    else:
        return (address // factor + 1) * factor
