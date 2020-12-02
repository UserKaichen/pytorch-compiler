import io
import os
import sys
import time
import torch
import shutil
import threading
from script.load_pt import load_pt
from script.makenet import makenet

def gen_fpga(filepath):
    os.chdir(filepath)
    cmd_list = ["rm -rf config*txt cfg_*txt *bn* *bias* *alpha* *weight* data_for_fpga/",
                "cp -af ../debug/output/* .",
                "python ../input/config_gen_file_v3_0.py -d ../imagenet/ -n imagenet_img6 -f",
                "mv data_for_fpga ../output"]

    for i in range(len(cmd_list)):
        os.system(cmd_list[i])

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

    loadpt.get_tensorinfo("debug/vggnet.py")
    loadpt.get_layercount("debug/layerinfo")

    with open("debug/output/img.input.q.txt", 'w') as fw:
        fw.write(loadpt.in_q)
        fw.write("\n")
        print("debug/output/img.input.q.txt write data success")

    with open(pt_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        dict = torch.load(buffer, map_location=torch.device('cpu'))
        for k, v in dict.items():
            if "quant_" in k or "classifier." in k:
                quant_list.append(k)
                quant_list.append(v)
            name_list.append(k)
            data_list.append(v)

    for i in range(loadpt.layer_cnts):
        layer = "{}{}{}".format("layers.", i, ".")
        for j in range(len(name_list)):
            if layer in name_list[j]:
                loadpt.layers.append([name_list[j], data_list[j]])
                counts += 1
        onelayer_cnt.append(str(counts))
        counts = 0

    del (loadpt.layers[0])
    logpath = "{}{}".format(os.getcwd(), "/debug/vggnet.log")
    for i in range(loadpt.layer_cnts):
        layername = "{}{}".format("layer_num:", str(i + 1))
        loadpt.layermsg = loadpt.get_layer_info(logpath, layername)
        loadpt.splicing_output(int(onelayer_cnt[i]), counts, quant_list)
        counts += int(onelayer_cnt[i])

    scale = fcname = weight = ""
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

def gen_net(mknet, filename):
    """
    description:
                Generate model files
    parameters:
                mknet:    The Class of makenet
                filename: The network name that the compiler can resolve
    return code:
                None
    """
    mknet.make_config(filename)
    mknet.make_block(filename)
    mknet.make_class(filename)
    mknet.make_layers(filename)
    mknet.make_padding(filename)
    mknet.make_forward(filename)
    mknet.make_weight(filename)
    mknet.make_main(filename)
    mknet.fmakenet.close()

    os.system("python3 debug/makenet.py > debug/layerinfo")
    mknet.splicing_layers("debug/layerinfo")
    mknet.bns[0] = mknet.get_op_code(filename, "bn")
    mknet.get_op_code(filename, "avgpool")
    mknet.get_op_code(filename, "weight")
    mknet.get_op_code(filename, "quant")
    mknet.get_op_code(filename, "fc")

    mknet._make_head()
    mknet._make_init()
    mknet._make_padding(filename)
    mknet._make_forward()
    mknet._make_avgpool()
    mknet._make_tail()
    mknet.fvggnet.close()
    print("make vggnet.py successfully")
        
def output(cnt, total, msg):
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
        if not os.path.exists("debug"):
            os.mkdir("debug")
        output = "{}{}".format("debug/", cleanlist[i])
        if "data_for_fpga" in cleanlist[i]:
            output = "{}{}".format("output/", cleanlist[i])
        if os.path.exists(output):
            if os.path.isdir(output):
                shutil.rmtree(output)
            else:
                os.remove(output)
            print("%s clean success" % (output))

    print("clean_ups successfully")

def checkfile(filelist):
    """
    description:
                Check files in working directory
    parameters:
                cleanlist: List of files to check
    return code:
                None
    """
    for i in range(len(filelist)):
        file = "input/"
        print("debug:", filelist[i], file)
        #if "imagenet" in filelist[i]:
        if "imagenet" == filelist[i]:
            file = ""
        name = "{}{}".format(file, filelist[i])
        if not os.path.exists(name):
            print("%s not found...\nPlease try again!" % name)
            sys.exit(1)

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
    fileslist = ["imagenet", "vgg_imagenet.py", "vgg_imagenet2.pt", "quant_layer.py"]
    cleanlist = ["makenet.py", "vggnet.py", "vggnet.log", "layerinfo", "output", "data_for_fpga"]

    if (len(sys.argv) != 2) or not os.path.exists(sys.argv[1]):
        pt_path = "input/vgg_imagenet2.pt"
    else:
        pt_path = sys.argv[1]

    start = time.time()
    allstart = start
    output(1, 6, "Check the necessary files...")
    checkfile(fileslist)
    end = time.time()
    schedule(1, start, end)

    start = time.time()
    output(2, 6, 'Clean the necessary files...')
    clean_ups(cleanlist)
    end = time.time()
    schedule(2, start, end)

    start = time.time()
    output(3, 6, 'Make network model file...')
    mknet = makenet()
    gen_net(mknet, "./input/vgg_imagenet.py")
    end = time.time()
    schedule(3, start, end)

    start = time.time()
    output(4, 6, 'Run the network model file...')
    os.system("python3 debug/vggnet.py > debug/vggnet.log")
    print("run vggnet.py successfully")
    end = time.time()
    schedule(4, start, end)

    start = time.time()
    os.system("mkdir -p debug/output output")
    output(5, 6, 'Load pt file and format output...')
    loadpt = load_pt()
    gen_txt(loadpt)
    print("pt data and config file successfully")
    end = time.time()
    schedule(5, start, end)

    start = time.time()
    output(6, 6, 'Make the bin file needed by fpga...')
    gen_fpga("imagenet")
    print("generate fpga data successfully")
    end = time.time()
    schedule(6, start, end)
    allend = end

    print('Make data successfully! It costs %.2f Seconds'%(allend - allstart))
