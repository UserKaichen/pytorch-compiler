import os
import sys
import shutil
import random
from optparse import OptionParser

fmain = 0

def send_runstepvar(fw):
    global fmain
    fmain = fw

def prints(string):
    print(string)
    fmain.write('{}{}'.format(string, '\n'))

def save_params(parase, info):
    with open(info, 'w') as ftool:
        ftool.write('muse toolchains all parameter:\n')
        ftool.write(f'ptpath      =  {parase.pt_name}\n')
        ftool.write(f'netpath     =  {parase.net_name}\n')
        if len(parase.conf_dir):
            ftool.write(f'confpath    =  {parase.conf_dir}\n')
        if len(parase.ptdata_dir):
            ftool.write(f'ptdtpath    =  {parase.ptdata_dir}\n')
        ftool.write(f'bmpdtpath   =  {parase.bmpdata_dir}\n')
        ftool.write(f'outputpath  =  {parase.output_dir}\n')

def clean_ups(cleanlist, mainpwd, wr_flag, info, debug):
    """
    description:
                Clean up working directory
    parameters:
                cleanlist: List of files to be cleaned
    return code:
                None
    """
    cleanlist += os.listdir(mainpwd)
    for i in range(len(cleanlist)):
        output = cleanlist[i]
        if os.path.exists(output):
            if os.path.isdir(output):
                if output == "input" or output == "script" \
                     or output == ".debug" or output == "nnbaton"\
                        or output == "imagenet":
                    continue
                shutil.rmtree(output)
            elif output == info.rsplit("/",1)[1].strip():
                if wr_flag:
                    prints("Clean up the last temporary directory")
                delete_lastdir(info, wr_flag)
                continue
            elif output == "main.py":
                continue
            else:
                os.remove(output)
            if wr_flag:
                fmain.write(f'{output} clean success\n')

    if wr_flag:
        prints("clean_ups successfully")
    else:
        if os.path.exists(debug):
            shutil.rmtree(debug)
        print(f'Clean up the toolchains successfully')

def delete_lastdir(info, flag):
    with open(info, 'r') as fd:
        for line in fd.readlines():
            if "=" in line:
                if "netpath" in line or "ptpath" in line:
                    continue
                line = line.split('=', 1)[1].strip()
                if flag:
                    fmain.write(f'{line} clean success\n')
                if os.path.exists(line):
                    shutil.rmtree(line)

def outputs(cnt, total, msg):
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

    for i in range(46 + 25):
        filling.append("=")

    prints(''.join(filling))
    prints(f'[Step {cnt}/{total}] {msg}')
    prints(''.join(filling))

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
    prints('******************************')
    prints(f'Step {i} Running time: %.7s s' % (end - start))
    prints('******************************')

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
        if "im6.bmp" in filelist[i]:
            file = "imagenet/"
        elif "ResNet" in filelist[i]:
            file = ""
        name = f'{file}{filelist[i]}'
        if not os.path.exists(name) and not os.path.exists(filelist[i]):
            prints(f'{name} not found...\nPlease try again!')
            sys.exit(1)
        else:
            fmain.write(f'check {filelist[i]} success\n')

    prints("checkfile successfully")

def get_model_name(netpath):
    with open(netpath, 'r') as fd:
        for line in fd:
            if line.startswith("if __name__ =="):
                for line in fd:
                    line = line.lstrip()
                    if line.startswith("model = "):
                        return line.split('(')[0].split('=')[1].strip()

def muse_parse():
    usage = "Usage: %prog [options] arg1 arg2 ... "
    parser = OptionParser(usage)

    parser.add_option("-p", "--pt_dir", dest="pt_name", help="PT file directory", action="store",
                      type="string", default=f'input/vggnet16.ckpt')
    parser.add_option("-n", "--net_dir", dest="net_name", help="Vggnet file directory", action="store",
                      type="string", default=f'input/vgg_imagenet.py')
    parser.add_option("-o", "--output_dir", dest="output_dir", help="Output file directory", action="store",
                      type="string", default=f'{os.getcwd()}/output')
    parser.add_option("-c", "--conf_dir", dest="conf_dir", help="Config file directory", action="store",
                      type="string", default=f'{os.getcwd()}/.debug/config')
    parser.add_option("-t", "--ptdata_dir", dest="ptdata_dir", help="Ptdata file directory", action="store",
                      type="string", default=f'{os.getcwd()}/.debug/ptdata')
    parser.add_option("-m", "--bmpdata_dir", dest="bmpdata_dir", help="Bmpdata file directory", action="store",
                      type="string", default=f'{os.getcwd()}/.debug/bmpdata')

    (options, args) = parser.parse_args()

    make_bin = "gen_inout_bin_vgg"

    logname = "vggnet16.log"
    model_name = get_model_name(options.net_name)
    if "resnet" in options.net_name.lower():
        logname = f'{model_name}.log'
        options.conf_dir = options.ptdata_dir = ""
        options.pt_name = '{}{}{}{}'.format(options.pt_name.split('/')[0], "/", model_name, ".ckpt")
        make_bin = "gen_inout_bin_res"

    return options, make_bin, logname

def adwrap(fw, str):
    fw.write(f'{str}\n')

def rand_param():
    H = random.randint(1, 500)
    C = random.randint(1, int(H/2))

    out_chl = random.randint(1, 100)
    ker_size = random.randint(1, 10)
    stride = random.randint(1, ker_size)
    padding = random.randint(0, 10)

    return C, H, out_chl, ker_size, stride, padding

def run_debug(runpath):
    simulog = f'{runpath}/../simu_output/simuinfo'
    netpath = f'input/simulator.py'
    
    N = 1
    C, H, out_chl, ker_size, stride, padding = rand_param()
    in_chl = C
    W = H

    logline = f'N:{N} C:{C} H:{H} W:{W} in_channels:{in_chl} out_channels:{out_chl} '
    logline = '{}{}'.format(logline, f'kernel_size:{ker_size} stride:{stride} padding:{padding}\n')
 
    if not os.path.exists(simulog):
        with open(simulog, 'w') as fw:
            fw.write(logline)
    else:
        with open(simulog, 'r') as fd:
            for line in fd:
                line.strip()
                if logline == line:
                    print(f'repeat data{line} in {simulog}，rand params again')
                    C, H, out_chl, ker_size, stride, padding = rand_param()
                    in_chl = C
                    W = H
        
        with open(simulog, 'a') as fa:
            logline = f'N:{N} C:{C} H:{H} W:{W} in_channels:{in_chl} out_channels:{out_chl} '
            logline = '{}{}'.format(logline, f'kernel_size:{ker_size} stride:{stride} padding:{padding}\n')
            fa.write(logline)
        
    with open(netpath, 'w') as fw:
       adwrap(fw, 'import torch')
       adwrap(fw, 'import torch.nn as nn\n')
       adwrap(fw, 'class Net(nn.Module):')
       adwrap(fw, '    def __init__(self):')
       adwrap(fw, '        super(Net, self).__init__()')
       adwrap(fw, f'        self.conv = nn.Conv2d({in_chl}, {out_chl}, {ker_size}, {stride}, {padding})')
       adwrap(fw, '    def forward(self, x):')
       adwrap(fw, '        x = self.conv(x)')
       adwrap(fw, '        return x\n')
       adwrap(fw, 'n = Net()')
       adwrap(fw, f'input = torch.rand({N}, {C}, {H}, {W})')
       adwrap(fw, 'module = torch.jit.trace(n, input)')
       adwrap(fw, 'module._c._fun_compile()\n')
    
    return netpath
