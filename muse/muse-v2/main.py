import io
import os
import sys
import time
import torch
import shutil
import threading
from optparse import OptionParser
from script.load_pt import load_pt

def option_parse():
    usage = "Usage: %prog [options] arg1 arg2 ... "
    parser = OptionParser(usage)

    parser.add_option("-p", "--pt_dir",      dest = "pt_name",    help = "PT file directory",      action = "store",
                      type = "string",       default = "input/vgg_imagenet.pt")
    parser.add_option("-n", "--net_dir",     dest ="net_name",    help = "Vggnet file directory",  action = "store",
                      type = "string",       default = "input/vgg_imagenet.py")
    parser.add_option("-o", "--output_dir",  dest ="output_dir",  help = "Output file directory",  action = "store",
                      type = "string",       default = "output")
    parser.add_option("-c", "--conf_dir",    dest ="conf_dir",    help = "Config file directory",  action = "store",
                      type = "string",       default = "debug/config")
    parser.add_option("-t", "--ptdata_dir",  dest ="ptdata_dir",  help = "Ptdata file directory",  action = "store",
                      type = "string",       default = "debug/ptdata")
    parser.add_option("-m", "--bmpdata_dir", dest ="bmpdata_dir", help ="Bmpdata file directory",  action = "store",
                      type = "string",       default = "debug/bmpdata")

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

    with open(f'debug/{logname}', 'r') as f:
        for lines in f.readlines():
            if "layer_num:" in lines:
                layer_cnts += 1

    return layer_cnts

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
    name_list = []
    data_list = []
    quant_list = []
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
    logpath = f'{os.getcwd()}/debug/{logname}'
    for i in range(loadpt.layer_cnts):
        layername = f'layer_num:{str(i + 1)}'
        loadpt.layermsg = loadpt.get_layer_info(logpath, layername)
        loadpt.splicing_output(int(onelayer_cnt[i]), counts, quant_list)
        counts += int(onelayer_cnt[i])

    scale = ""
    fcname = ""
    weight = ""
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
        make_bin = ["Resnet not run fpga", ""]

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

    allend = time.time()
    print('Make data successfully! It costs %.2f Seconds'%(allend - allstart))
