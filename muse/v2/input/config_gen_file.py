#!/bin/python3
# -*- coding: UTF-8 -*-
import struct as st
import binascii as bac
from optparse import OptionParser
import struct
import os
import os.path
import re


def option_parse():
    usage="Usage: %prog [options] arg1 arg2"
    parser = OptionParser(usage)
    parser.add_option("-d", "--file_dir", dest="dir_name", help="The dir which contain config file and data file", action = "store", type="string", default="")
    parser.add_option("-n", "--test_name", dest="test_name", help="The test name", action = "store", type="string", default="")
    #parser.add_option("-r", "--rtl", dest="rtl_flag", help="The number of test we want to run", action = "store", type="int", default="1")
    parser.add_option("-m", "--muse", dest="muse_flag", help="The enable of muse rtl verify file", action = "store_true", default=False)
    parser.add_option("-s", "--muse_dir", dest="muse_dir", help="The dir which contain  muse verify file and data file", action = "store", type="string", default="./")
    parser.add_option("-r", "--rtl", dest="rtl_flag", help="The enable of dma + muse verify file", action = "store_true", default=False)
    parser.add_option("-v", "--rtl_dir", dest="rtl_dir", help="The dir which contain dma+muse verify file and data file", action = "store", type="string", default="./")
    parser.add_option("-f", "--fpga", dest="fpga_flag", help="The enable of fpga file", action = "store_true", default=False)
    (options,args)=parser.parse_args()
    return options

class Layer :
    #bit9    wwp ;

    #int layer_num;

    #string dir_name              ;
    #string act0_fname            ;
    #string act1_fname            ;
    #string wei_fname             ;
    #string ord_fname             ;

    ## dma parameter
    #int  act_saddr_0             ;  
    #int  act_saddr_1             ;
    #int  act_stride              ;
    #int  last_out_channel        ;

    #int  wei_saddr               ;
    #int  wei_stride              ;

    #int  ord_saddr               ;
    #int  ord_stride              ;
    #int  order_burst_num         ;

    #int  out_saddr               ; 
    #int  out_stride              ; 
    #int  out_hor                 ; 
    #int  out_ver                 ; 
    #int  out_chl                 ; 
    #int  output_mode             ; 
    #int  output_mode1            ; 
    #bit  pooling_8x8_en          ; 


    ## muse parameter
    #int  order_ram_ver           ; 
    #int  ver_bank_size           ; 
    #int  rd_chl_mode             ; 
    #int  sel_8fi_enable          ; 

    #int  pattern_mode            ; 
    #int  filter_str              ; 
    #int  pooling_size            ; 
    #int  pooling_mode            ; 
    #bit  order_en                ; 

    #int  feature_ver             ; 
    #int  feature_hor             ; 

    #int  in_channel              ; 
    #int  alu_mode                ; 
    #int  alu_ab_value            ; 
    #bit  pooling_up_or_down      ; 

    #int  order_real_num          ; 

    #int  out_channel             ; 

    #int  reuse_chl               ; 
    #int  reuse_hor               ; 
    #int  reuse_ver               ; 

    #int  l1_ch_weight_size       ; 
    #int  l2_ch_weight_size       ; 

    #int  quan_code_27fi          ; 
    #int  quan_code_8fi           ; 

    #bit[8:0]  w_pattern_data0    ; 
    #bit[8:0]  w_pattern_data1    ; 
    #bit[8:0]  w_pattern_data2    ; 
    #bit[8:0]  w_pattern_data3    ; 
    #bit[8:0]  w_pattern_data4    ; 
    #bit[8:0]  w_pattern_data5    ; 
    #bit[8:0]  w_pattern_data6    ; 
    #bit[8:0]  w_pattern_data7    ; 
    #bit[8:0]  w_pattern_data8    ; 
    #bit[8:0]  w_pattern_data9    ; 
    #bit[8:0]  w_pattern_data10   ; 
    #bit[8:0]  w_pattern_data11   ; 
    #bit[8:0]  w_pattern_data12   ; 
    #bit[8:0]  w_pattern_data13   ; 
    #bit[8:0]  w_pattern_data14   ; 
    #bit[8:0]  w_pattern_data15   ; 

    #bit  mul_en                  ; 
    #bit  sum_en                  ; 
    #bit  overflow_sel            ; 
    #bit  relu_en                 ; 
    #int  quan_code_16fp          ; 

    #int  output_total_num        ; 

    #int  ker_pruning_num         ; 
    #bit  index_en                ; 
    #bit  clock_gate_en           ; 

    def __init__(self, name, num):
        self.wwp=[]
        #for wwpi in range(0,16) : 
        #    self.wwp.append(wwp[wwpi])
        self.layer_num = num

    def config_layer(self, dir_name, file_list, para_list, wwp_list, alone) :
        #para :  pattern_mode , filter_str, out_mode
        self.set_layer(para_list[0],       para_list[1],    para_list[2])
        #para :  file_dir, act0_file, act1_file, wei_file, ord_file
        self.set_file(dir_name, file_list[0],   file_list[1],   file_list[2],  file_list[3])
        #para :  act_addr0, feature_hor, feature_ver, in_channel, last_out_channel, quan_code_27fi
        self.set_input(para_list[3],   para_list[4],     para_list[5],     para_list[6],    para_list[7],          para_list[8])
        #para: weight_addr0, weight_stride, index_en, ker_pruning_num, w_pattern
        if (alone) :
            self.set_weight(0x40000,   para_list[10],      para_list[11], para_list[12],        wwp_list)
        else :
            self.set_weight(para_list[9],   para_list[10],      para_list[11], para_list[12],        wwp_list)
        #para : order_addr, order_en
        self.set_order(para_list[13],  para_list[14]       )
        #para : act_addr_1
        self.set_add( para_list[15]     )
        #para :   out_addr,  out_hor, out_ver, out_chl, quan_code_16fp, pool_8x8_en
        self.set_output(para_list[16],  para_list[17],para_list[18],para_list[19],para_list[20],       para_list[21])
        #para : alu_mode , alu_ab_value , quan_code_8fi
        self.set_conv(para_list[22],  para_list[23],      para_list[24]            )
        #para : mul_en , sum_en , relu_en , overflow_sel
        self.set_bn(  para_list[25],para_list[26],para_list[27], para_list[28]           )
        #para : pool_size , pool_mode , pool_up_or_down
        self.set_pool(para_list[29],   para_list[30],   para_list[31]              )

    #para : pattern_mode , filter_str, out_mode
    def set_layer(self, ptn_mode, f_str, mode) :
        self.pattern_mode = ptn_mode
        self.filter_str   = f_str
        self.output_mode  = mode 

    def set_file(self, file_dir, file0, file1, file2, file3) :
        self.dir_name = file_dir
        self.act0_fname = file0
        self.act1_fname = file1
        self.wei_fname = file2
        self.ord_fname = file3

    #para : alu_mode , alu_ab_value , quan_code_8fi
    def set_conv(self, a_mode, a_ab, quan) :
        self.alu_mode       = a_mode 
        self.alu_ab_value   = a_ab   
        self.quan_code_8fi  = quan  
        self.cal_enable_8fi(); 
        #cal_reuse();

    #para : mul_en , sum_en , relu_en , overflow_sel
    def set_bn(self, men, sen, ren, of_sel) :
        self.mul_en  = men 
        self.sum_en  = sen 
        self.relu_en = ren 
        self.overflow_sel = of_sel 
        
    #para : pool_size , pool_mode , pool_up_or_down
    def set_pool(self, size, mode, ud) :
        self.pooling_size = size 
        self.pooling_mode = mode 
        self.pooling_up_or_down = ud 

    #para: act_addr0, feature_hor, feature_ver, in_channel, last_out_channel, quan_code_27fi
    def set_input(self, saddr, hor, ver, chl, last_chl, quan) :
        self.act_saddr_0 = saddr
        self.feature_hor = hor
        if (self.pattern_mode==6) :
            self.feature_ver = ver * 2 
        else :
            self.feature_ver = ver
        if (chl == 3) :
            self.in_channel = chl
        else :
            self.in_channel  = (int((chl-1)/8) + 1) * 8 
        if (self.pattern_mode==4) :
            self.rtl_hor = self.in_channel/4
            self.rtl_ver = 4
            self.rtl_chl = 1
            self.order_real_num   = chl/16 
        else :
            self.rtl_hor = self.feature_hor
            self.rtl_ver = self.feature_ver
            self.rtl_chl = self.in_channel
            self.order_real_num   = chl 
        self.quan_code_27fi   = quan
        self.last_out_channel = last_chl 
        self.act_stride = hor * self.last_out_channel 
        #self.cal_rd_chl_mode()

    #para: weight_addr0, weight_stride, index_en, ker_pruning_num, w_pattern
    def set_weight(self, saddr, stride, idx_en, ker_num, w_pattern) :
        if(self.pattern_mode == 0) :
            pattern_num = 5
        elif(self.pattern_mode == 1) :
            pattern_num = 6 
        elif(self.pattern_mode == 2) :
            pattern_num = 9 
        else :
            pattern_num = 1 
        self.wei_saddr       = saddr 
        self.index_en        = idx_en
        if(self.pattern_mode==4) :
            self.ker_pruning_num = ker_num/16 
        elif(self.index_en==0) :
            self.ker_pruning_num = self.order_real_num
        else :
            self.ker_pruning_num = ker_num 
        
        self.set_w_pattern(w_pattern)
        self.cal_weight_size(pattern_num)
        self.cal_rd_chl_mode()

        #self.wei_stride = ((self.l1_ch_weight_size-1)/32+1)*32;
        if(((self.l1_ch_weight_size-1)/32+1)*16>stride) :
            print("weight stride is too narrow!")
        else :
            self.wei_stride   = stride
        

    def cosel_out_mode(self) :
        if ( min( (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) , (1024 / (self.in_channel)) )  > 8 ) :
            ram_bank = 8
        else :
            ram_bank = min( (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) , (1024 / (self.in_channel)) )
        if (self.pattern_mode > 4) :
            self.output_mode = 0
        elif ( ram_bank*32*(self.rd_chl_mode+1) >= self.out_channel ) :
            self.output_mode = 0
        else : 
            self.output_mode = 1

    #para : order_addr, order_en
    def set_order(self, saddr, ord_en):
        self.ord_saddr       = saddr 
        self.order_en        = ord_en 
        self.order_burst_num = (self.order_real_num-1)/14
        self.cal_ord_stride()

    #para : act_addr_1
    def set_add(self, saddr):
        self.act_saddr_1 = saddr 

    #para : out_addr, out_hor, out_ver, out_chl, quan_code_16fp, pool_8x8_en
    def set_output(self, saddr, hor, ver, chl, quan, pool_8x8) :
        self.out_saddr   = saddr 
        self.out_hor     = hor 
        self.out_ver     = ver 
        if(self.pattern_mode>=5) :
            self.out_channel = 64 
        else :
            self.out_channel = chl 
        self.out_chl     = chl 
        #self.out_stride  = hor * out_channel;
        self.pooling_8x8_en = pool_8x8 
        self.quan_code_16fp = quan 
        #cal_rd_chl_mode();
        self.cosel_out_mode()
        self.cal_ver_bank_size()
        self.cal_resue()
        self.cal_ram_ver()
        self.cal_out_total_num()

    # ***************************************
    def cal_resue(self) :
        if (self. pattern_mode >3) :
            self.reuse_hor      = 0
            self.reuse_ver      = 0
            self.reuse_chl      = 0
        elif(self.output_mode == 0) :
            self.reuse_hor      = 0
            self.reuse_ver      = 0
            self.reuse_chl      = (self.out_channel-1)/(32*(self.rd_chl_mode+1))
        elif(self.output_mode == 1) :
            self.reuse_hor      = (self.feature_hor-1)/(2*(self.filter_str+1))
            self.reuse_ver      = (self.feature_ver-1)/(2*(self.filter_str+1))
            #self.reuse_chl      = (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) - 1;#(self.out_channel-1)/(32*(self.rd_chl_mode+1));
            if(min( (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) , (1024 / (self.in_channel)) )  > 8) :
                self.reuse_chl      = 7
            else :
                self.reuse_chl      = min( (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) - 1 , (1024 / (self.in_channel) - 1) )
        elif(self.output_mode == 2) :
            self.reuse_hor      = (self.feature_hor-1)/(2*(self.filter_str+1))
            self.reuse_ver      = 0
            #self.reuse_chl      = (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) - 1;#(self.out_channel-1)/(32*(self.rd_chl_mode+1));
            if(min( (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) , (1024 / (self.in_channel)) )  > 8) :
                self.reuse_chl      = 7
            else :
                self.reuse_chl      = min( (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) - 1 , (1024 / (self.in_channel) - 1) )
        elif(self.output_mode == 3) :
            self.reuse_hor      = 0
            self.reuse_ver      = 0
            #self.reuse_chl      = (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) - 1;
            if(min( (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) , (1024 / (self.in_channel)) )  > 8) :
                self.reuse_chl      = 7
            else :
                self.reuse_chl      = min( (304*2/(((self.l1_ch_weight_size-1)/16+1)*(self.rd_chl_mode+1))) - 1 , (1024 / (self.in_channel) - 1) )

    def cal_ver_bank_size(self):
        if((self.pattern_mode>3) or (self.output_mode == 1)) :
            self.ver_bank_size = 0 
        else :
            self.ver_bank_size = (3072)/(((self.feature_hor-1)/5+1)*((self.in_channel-1)/8+1)*8) - 1
        #if((self.feature_ver/5)<(self.ver_bank_size+1))
        #    self.ver_bank_size = (self.feature_ver/5) - 1;
        if(self.ver_bank_size>7) :
            self.ver_bank_size = 7
        #self.ver_bank_size = 5;

    def set_w_pattern(self, w_pattern):
        self.w_pattern_data0  = w_pattern[ 0] 
        self.w_pattern_data1  = w_pattern[ 1] 
        self.w_pattern_data2  = w_pattern[ 2] 
        self.w_pattern_data3  = w_pattern[ 3] 
        self.w_pattern_data4  = w_pattern[ 4] 
        self.w_pattern_data5  = w_pattern[ 5] 
        self.w_pattern_data6  = w_pattern[ 6] 
        self.w_pattern_data7  = w_pattern[ 7] 
        self.w_pattern_data8  = w_pattern[ 8] 
        self.w_pattern_data9  = w_pattern[ 9] 
        self.w_pattern_data10 = w_pattern[10] 
        self.w_pattern_data11 = w_pattern[11] 
        self.w_pattern_data12 = w_pattern[12] 
        self.w_pattern_data13 = w_pattern[13] 
        self.w_pattern_data14 = w_pattern[14] 
        self.w_pattern_data15 = w_pattern[15] 

    def cal_weight_size(self, pattern_num) :
        if(self.pattern_mode>=4) :
            chl_num = self.in_channel
        elif (self.index_en==0):
            chl_num = self.in_channel
        else :
            chl_num = self.ker_pruning_num
        self.l2_ch_weight_size = pattern_num * chl_num + 8 

        if(self.index_en==1) :
            self.l1_ch_weight_size = self.l2_ch_weight_size + ((self.in_channel-1)/64+1)*64/4 
        else :
            self.l1_ch_weight_size = self.l2_ch_weight_size 

    def cal_enable_8fi(self) :
        if(self.quan_code_27fi != self.quan_code_8fi) :
            self.sel_8fi_enable = 1
        else :
            self.sel_8fi_enable = 0 

    def cal_ord_stride(self):
        self.ord_stride = ((self.order_real_num-1)/14 + 1) * 16

    def cal_ram_ver(self) :
        if(((self.in_channel<7) or (self.pattern_mode>=4)) and (self.order_en==1)) :
            print("order_en should be closed!")
        if(self.order_en==1) :
            self.order_ram_ver = self.reuse_chl
        else :
            self.order_ram_ver = 0

    def cal_rd_chl_mode(self) :
        #if((self.pattern_mode < 4) and (self.in_channel > 256)) :
        if((self.pattern_mode < 4) and (304/((self.l1_ch_weight_size-1)/16+1) < 2 )) :
            self.rd_chl_mode = 0
        else :
            self.rd_chl_mode = 1

    def cal_out_total_num(self) :
        #int hor, ver ;
        if(self.pattern_mode>=5) :
            chl = self.in_channel 
        else :
            chl = ((self.out_channel-1)/64+1)*64 
        #if(self.filter_str == 1) begin
        #    hor = self.feature_hor/2 ;
        #    ver = self.feature_ver/2 ;
        #end
        #else begin
        #    hor = self.feature_hor ;
        #    ver = self.feature_ver ;
        #end
        #self.output_total_num = hor * ver * ((chl-1)/64+1)*64 / 16 ;
        self.output_total_num = self.out_hor * self.out_ver * chl / 16 
        self.out_stride  = self.out_hor * chl

###################################
########### gen seq file ##########
###################################
def gen_seq_file(seq_lib, name, tmp_layer, in_file, elw_file, ord_file, wei_file, index_file, idx_file, k_file, b_file, out_file) :
    seq_name   = name+"_seq"
    seq_file_name = seq_lib+seq_name+".sv"
    seq_file   = open(seq_file_name,"w")
    class_text = "class " +seq_name+ " extends uvm_sequence #(in_trans);\n\n"
    class_text += "    `uvm_object_utils(" +seq_name+ ")\n"
    class_text += "    `uvm_declare_p_sequencer(virtual_sequencer)\n"
    class_text += "    in_trans tr;\n"
    class_text += "    function new(string name = \"" +seq_name+ "\");\n"
    class_text += "        super.new(name);\n"
    class_text += "    endfunction\n\n"
    class_text += "    virtual task pre_body();\n"
    class_text += "        if (get_starting_phase() != null)\n"
    class_text += "            get_starting_phase().raise_objection(this);\n"
    class_text += "    endtask\n"
    class_text += "    virtual task post_body();\n"
    class_text += "        if (get_starting_phase() != null)\n"
    class_text += "            get_starting_phase().drop_objection(this);\n"
    class_text += "    endtask\n\n"
    class_text += "    virtual task body();\n"
    class_text += "        wait(harness.rst == 1'b0);\n"
    class_text += "        wait(harness.rst == 1'b1);\n"
    class_text += "        for(int ll = 0; ll < 2; ll++)\n"
    class_text += "            send_one_layer(ll);\n"
    class_text += "        repeat(10) @(posedge p_sequencer.c_if.clk);\n"
    class_text += "        `uvm_info(get_type_name(), \"Send all master trans done!\", UVM_HIGH);\n"
    class_text += "    endtask\n\n"
    class_text += "    virtual task send_one_layer(input int layer_cnt);\n"
    class_text += "        tr = new();\n"
    class_text += "        `uvm_info(get_type_name(), $sformatf(\"Start send %d-th in_tran!\", layer_cnt), UVM_LOW);\n"
    class_text += "        `uvm_do_on_with(tr, p_sequencer.in_sqr,\n"
    class_text += "                       {tr.mode               == " + str((tmp_layer.pattern_mode<4)*(tmp_layer.output_mode+1))  + ";\n"
    class_text += "                        tr.exception_flag     == 0;\n"
    class_text += "                        tr.exception_delay    == 2000;\n"
    class_text += "                        tr.order_ram_ver      == " + str(tmp_layer.order_ram_ver       )  + ";\n"
    class_text += "                        tr.ver_bank_size      == " + str(tmp_layer.ver_bank_size       )  + ";\n"
    class_text += "                        tr.rd_chl_mode        == " + str(tmp_layer.rd_chl_mode         )  + ";\n"
    class_text += "                        tr.sel_8fi_enable     == " + str(tmp_layer.sel_8fi_enable      )  + ";\n"
    class_text += "                        tr.pattern_mode       == " + str(tmp_layer.pattern_mode        )  + ";\n"
    class_text += "                        tr.filter_str         == " + str(tmp_layer.filter_str          )  + ";\n"
    class_text += "                        tr.pooling_size       == " + str(tmp_layer.pooling_size        )  + ";\n"
    class_text += "                        tr.pooling_mode       == " + str(tmp_layer.pooling_mode        )  + ";\n"
    class_text += "                        tr.order_en           == " + str(tmp_layer.order_en            )  + ";\n"
    class_text += "                        tr.feature_ver        == " + str(tmp_layer.rtl_ver-1           )  + ";\n"
    class_text += "                        tr.feature_hor        == " + str(tmp_layer.rtl_hor-1           )  + ";\n"
    class_text += "                        tr.in_channel         == " + str(tmp_layer.rtl_chl-1           )  + ";\n"
    class_text += "                        tr.alu_mode           == " + str(tmp_layer.alu_mode            )  + ";\n"
    class_text += "                        tr.alu_ab_value       == " + str(tmp_layer.alu_ab_value        )  + ";\n"
    class_text += "                        tr.pooling_up_or_down == " + str(tmp_layer.pooling_up_or_down  )  + ";\n"
    class_text += "                        tr.order_real_num     == " + str(tmp_layer.order_real_num-1    )  + ";\n"
    class_text += "                        tr.out_channel        == " + str(tmp_layer.out_channel-1       )  + ";\n"
    class_text += "                        tr.reuse_chl          == " + str(tmp_layer.reuse_chl           )  + "; \n"
    class_text += "                        tr.reuse_hor          == " + str(tmp_layer.reuse_hor           )  + ";\n"
    class_text += "                        tr.reuse_ver          == " + str(tmp_layer.reuse_ver           )  + ";\n"
    class_text += "                        tr.l1_ch_weight_size  == " + str(tmp_layer.l1_ch_weight_size-1 )  + ";\n"
    class_text += "                        tr.l2_ch_weight_size  == " + str(tmp_layer.l2_ch_weight_size-1 )  + ";\n"
    class_text += "                        tr.quan_code_27fi     == " + str(tmp_layer.quan_code_27fi    )  + ";\n"
    class_text += "                        tr.quan_code_8fi      == " + str(tmp_layer.quan_code_8fi     )  + ";\n"
    class_text += "                        tr.w_pattern_data0    == 9'b" + tmp_layer.w_pattern_data0     + ";\n"
    class_text += "                        tr.w_pattern_data1    == 9'b" + tmp_layer.w_pattern_data1     + ";\n"
    class_text += "                        tr.w_pattern_data2    == 9'b" + tmp_layer.w_pattern_data2     + ";\n"
    class_text += "                        tr.w_pattern_data3    == 9'b" + tmp_layer.w_pattern_data3     + ";\n"
    class_text += "                        tr.w_pattern_data4    == 9'b" + tmp_layer.w_pattern_data4     + ";\n"
    class_text += "                        tr.w_pattern_data5    == 9'b" + tmp_layer.w_pattern_data5     + ";\n"
    class_text += "                        tr.w_pattern_data6    == 9'b" + tmp_layer.w_pattern_data6     + ";\n"
    class_text += "                        tr.w_pattern_data7    == 9'b" + tmp_layer.w_pattern_data7     + ";\n"
    class_text += "                        tr.w_pattern_data8    == 9'b" + tmp_layer.w_pattern_data8     + ";\n"
    class_text += "                        tr.w_pattern_data9    == 9'b" + tmp_layer.w_pattern_data9     + ";\n"
    class_text += "                        tr.w_pattern_data10   == 9'b" + tmp_layer.w_pattern_data10    + ";\n"
    class_text += "                        tr.w_pattern_data11   == 9'b" + tmp_layer.w_pattern_data11    + ";\n"
    class_text += "                        tr.w_pattern_data12   == 9'b" + tmp_layer.w_pattern_data12    + ";\n"
    class_text += "                        tr.w_pattern_data13   == 9'b" + tmp_layer.w_pattern_data13    + ";\n"
    class_text += "                        tr.w_pattern_data14   == 9'b" + tmp_layer.w_pattern_data14    + ";\n"
    class_text += "                        tr.w_pattern_data15   == 9'b" + tmp_layer.w_pattern_data15    + ";\n"
    class_text += "                        tr.mul_en             == " + str(tmp_layer.mul_en            )    + ";\n"
    class_text += "                        tr.sum_en             == " + str(tmp_layer.sum_en            )    + ";\n"
    class_text += "                        tr.overflow_sel       == " + str(tmp_layer.overflow_sel      )    + ";\n"
    class_text += "                        tr.relu_en            == " + str(tmp_layer.relu_en           )    + ";\n"
    class_text += "                        tr.quan_code_16fp     == " + str(tmp_layer.quan_code_16fp    )    + ";\n"
    class_text += "                        tr.output_total_num   == " + str(tmp_layer.output_total_num  )    + ";\n"
    class_text += "                        tr.ker_purning_num    == " + str(tmp_layer.ker_pruning_num-1   )    + ";\n"
    class_text += "                        tr.index_en           == " + str(tmp_layer.index_en          )    + ";\n"
    class_text += "                        tr.net_sel            == 0;\n"
    class_text += "                        tr.data_sel           == 1;\n" 
    class_text += "                        tr.layer_num          == 0; })\n"
    class_text += "                        harness.map_file      = \"../data/" + in_file    + "\" ;\n"
    class_text += "                        harness.elw_file      = \"../data/" + elw_file   + "\" ;\n"
    class_text += "                        harness.order_file    = \"../data/" + ord_file   + "\" ;\n"
    class_text += "                        harness.weight_file   = \"../data/" + wei_file   + "\" ;\n"
    class_text += "                        harness.index_file    = \"../data/" + index_file + "\" ;\n"
    class_text += "                        harness.idx_file      = \"../data/" + idx_file   + "\" ;\n"
    class_text += "                        harness.k_file        = \"../data/" + k_file     + "\" ;\n"
    class_text += "                        harness.b_file        = \"../data/" + b_file     + "\" ;\n"
    class_text += "                        harness.out_file      = \"../data/" + out_file   + "\" ;\n"
    class_text += "        `uvm_info(get_type_name(), $sformatf(\"Send %d-th in_tran done!\", layer_cnt), UVM_LOW);\n"
    class_text += "        `uvm_info(get_type_name(), \"Send one layer trans done!\", UVM_LOW);\n"
    class_text += "    endtask\n"
    class_text += "endclass\n"
    seq_file.write(class_text)
    seq_file.close()
    seq_f   = open(seq_lib+"seq.f","a+")
    str_text = seq_f.read()
    serobj = re.search(r'"' + seq_name + r'.sv"', str_text, re.M)
    if(serobj == None) :
        seq_f.write("\n`include \""+seq_name+".sv\"");
    seq_f.close()

###################################
########### gen tc file ##########
###################################
def gen_tc_file(tc_lib, name):
    seq_name   = name+"_seq"
    test_name    = name+"_test"
    test_file_name = tc_lib+test_name+".sv"
    test_file   = open(test_file_name,"w")
    class_text = "class " +test_name+ " extends base_test;\n\n"
    class_text += "    function new(string name, uvm_component parent);\n"
    class_text += "        super.new(name,parent);\n"
    class_text += "    endfunction\n\n"
    class_text += "    extern virtual function void build_phase(uvm_phase phase);\n"
    class_text += "    `uvm_component_utils("+test_name+")\n"
    class_text += "\nendclass:"+test_name+"\n\n"
    class_text += "function void "+test_name+"::build_phase(uvm_phase phase);\n"
    class_text += "    super.build_phase(phase);\n"
    class_text += "    uvm_config_db#(uvm_object_wrapper)::set(null, \"uvm_test_top.env.vsqr.main_phase\", \"default_sequence\", "+seq_name+"::type_id::get());\n"
    class_text += "endfunction:build_phase"
    test_file.write(class_text)
    test_file.close()
    tc_f    = open(tc_lib+"tc.f","a+")
    str_text = tc_f.read()
    serobj = re.search(r'"' + test_name + r'.sv"', str_text, re.M)
    if(serobj == None) :
        tc_f.write("\n`include \""+test_name+".sv\"");
    tc_f.close()

###################################
########### gen cpp file ##########
###################################
#def gen_net_file_cpp(t_name, d_name, f_list, pm_list, wwp_list, alone) :
#    test_name = t_name+"_net"
#    #net_file = open(d_name+test_name+".cpp", "w")
#    net_file = open(d_name+"vgg16_noprun_net.cpp", "w")
#    net_text = "#include \"stdafx.h\"\n"
#    #net_text += "#include \"" + test_name + ".h\"\n"
#    #net_text += test_name + "::" + test_name + "(string name)\n"
#    net_text += "#include \"vgg16_noprun_net.h\"\n"
#    net_text += "vgg16_noprun_net::vgg16_noprun_net(string name)\n"
#    net_text += "{\n"
#    net_text += "    int i = 0;\n"
#    net_text += "    int j = 0;\n\n"
#    #net_text += "    file_dir = \"" + d_name + "\"\n"
#    net_text += "    file_dir = \"D:\\\\muse_v2_temp_test_data\\\\\"\n"
#    net_text += "    this->name = name;\n"
#    net_text += "    int params_tmp[1][32] = \n"
#    net_text += "    {\n"
#    net_text += "        // conv0\n"
#    net_text += "    	 {/*layer*/"+str(pm_list[0])+", "+str(pm_list[1])+", "+str(pm_list[2])+", "
#    net_text += "/*input*/0x"+pm_list[3]+", "+str(pm_list[4])+", "+str(pm_list[5])+", "+str(pm_list[6])+", "+str(pm_list[7])+", "+str(pm_list[8])+","
#    net_text += "/*weight*/0x"+pm_list[9]+", "+str(pm_list[10])+", "+str(pm_list[11])+", "+str(pm_list[12])+", "
#    net_text += "/*order*/0x"+pm_list[13]+", "+str(pm_list[14])+", /*add*/0x"+pm_list[15]+",\n"
#    net_text += "    	  /*out*/0x"+pm_list[16]+", "+str(pm_list[17])+", "+str(pm_list[18])+", "+str(pm_list[19])+", "+str(pm_list[20])+", "+str(pm_list[21])+", "
#    net_text += "/*conv*/"+str(pm_list[22])+", "+str(pm_list[23])+", "+str(pm_list[24])+", "
#    net_text += "/*bn*/"+str(pm_list[25])+", "+str(pm_list[26])+", "+str(pm_list[27])+", "+str(pm_list[28])+", "
#    net_text += "/*pool*/"+str(pm_list[29])+", "+str(pm_list[30])+", "+str(pm_list[31])+" }\n"
#    net_text += "    };\n\n"
#    net_text += "    //copy initialized data.\n"
#    net_text += "    memcpy(params,params_tmp,sizeof(params_tmp));\n\n"
#    net_text += "    string file_names_tmp[1][4] = {\n"
#    net_text += "        { \""+f_list[0]+"\", \""+f_list[1]+"\", \""+f_list[2]+"\", \""+f_list[3]+"\" }\n"
#    net_text += "    };\n"
#    
#    net_text += "    for (i = 0; i < 1; i++)\n"
#    net_text += "    {\n"
#    net_text += "    	for (j = 0; j < 4; j++)\n"
#    net_text += "    	{\n"
#    net_text += "    		file_names[i][j] = file_names_tmp[i][j];\n"
#    net_text += "    	}\n"
#    net_text += "    }\n\n"
#    net_text += "    USHORT wwp[1][16]   = {{"
#    for wwpi in range(0,16) :
#        net_text += "0b"+wwp_list[wwpi][::-1]
#        if (wwpi!=15) :
#            net_text += ", "
#    net_text += "}};\n"
#    net_text += "}\n\n"
#    net_text += "vgg16_noprun_net::~vgg16_noprun_net()\n"
#    net_text += "{\n\n\n"
#    net_text += "}\n\n"
#    net_text += "void vgg16_noprun_net::self_config(bool alone)\n"
#    net_text += "{\n"
#    net_text += "    layer *tmp_layer = NULL;\n\n"
#    
#    net_text += "    char layer_name[64] = { 0 };\n\n"
#    net_text += "    for (int i = 0; i < sizeof(params)/ sizeof(params[0]); i++)\n"
#    net_text += "    {\n"
#    net_text += "        memset(layer_name,0,sizeof(layer_name));\n"
#    net_text += "        sprintf(layer_name, \"%s_layer_%d\", this->name.c_str(), i);\n"
#    net_text += "        tmp_layer = new layer((string)layer_name, i);\n"
#    net_text += "        tmp_layer->config_layer(file_dir, file_names[i], params[i], wwp, alone);\n"
#    
#    net_text += "        append_layer(tmp_layer);\n"
#    net_text += "    }\n"
#    net_text += "}\n"
#    net_file.write(net_text)
#    net_file.close()

def gen_net_file_cpp(t_name, d_name, nfi_list, npm_list, nwwp_list, alone) :
    test_name = t_name+"_net"
    #net_file = open(d_name+test_name+".cpp", "w")
    net_file = open(d_name+"tmp_net.cpp", "w")
    tmp_layer_num = len(npm_list)
    net_text =  "    int params_tmp["+str(tmp_layer_num)+"][32] = \n"
    net_text += "    {\n"
    for ci in range(0, tmp_layer_num) :
        pm_list  = npm_list[ci]
        net_text += "        // layer "+str(ci)+"\n"
        net_text += "    	 {/*layer*/"+str(pm_list[0])+", "+str(pm_list[1])+", "+str(pm_list[2])+", "
        net_text += "/*input*/"+hex(pm_list[3])+", "+str(pm_list[4])+", "+str(pm_list[5])+", "+str(pm_list[6])+", "+str(pm_list[7])+", "+str(pm_list[8])+","
        net_text += "/*weight*/"+hex(pm_list[9])+", "+str(pm_list[10])+", "+str(pm_list[11])+", "+str(pm_list[12])+", "
        net_text += "/*order*/"+hex(pm_list[13])+", "+str(pm_list[14])+", /*add*/"+hex(pm_list[15])+",\n"
        net_text += "    	  /*out*/"+hex(pm_list[16])+", "+str(pm_list[17])+", "+str(pm_list[18])+", "+str(pm_list[19])+", "+str(pm_list[20])+", "+str(pm_list[21])+", "
        net_text += "/*conv*/"+str(pm_list[22])+", "+str(pm_list[23])+", "+str(pm_list[24])+", "
        net_text += "/*bn*/"+str(pm_list[25])+", "+str(pm_list[26])+", "+str(pm_list[27])+", "+str(pm_list[28])+", "
        net_text += "/*pool*/"+str(pm_list[29])+", "+str(pm_list[30])+", "+str(pm_list[31])+" }"
        if ((ci+1)==tmp_layer_num) :
            net_text += "\n"
        else :
            net_text += ",\n"
    net_text += "    };\n\n"
    net_text += "    string file_names_tmp["+str(tmp_layer_num)+"][4] = {\n"
    for ci in range(0, tmp_layer_num) :
        f_list   = nfi_list[ci]
        net_text += "        { \""+f_list[0]+"\", \""+f_list[1]+"\", \""+f_list[2]+"\", \""+f_list[3]+"\" }"
        if ((ci+1)==tmp_layer_num) :
            net_text += "\n"
        else :
            net_text += ",\n"
    net_text += "    };\n"
    net_text += "    USHORT wwp_tmp["+str(tmp_layer_num)+"][16]   = {\n"
    for ci in range(0, tmp_layer_num) :
        wwp_list = nwwp_list[ci]
        net_text += "        {"
        for wwpi in range(0,16) :
            net_text += "0b"+wwp_list[wwpi][::-1]
            if (wwpi!=15) :
                net_text += ", "
        net_text += "}"
        if ((ci+1)==tmp_layer_num) :
            net_text += "\n"
        else :
            net_text += ",\n"
    net_text += "    };\n"
    net_file.write(net_text)
    net_file.close()


###################################
########### gen cpp file ##########
###################################
def gen_net_file_sv(t_name, d_name, r_dir, nfi_list, npm_list, nwwp_list, alone) :
    test_name = t_name+"_seq"
    net_name = t_name+"_net"
    net_file = open(r_dir+test_name+".sv", "w")
    tmp_layer_num = len(npm_list)
    net_text = "class " + net_name + " extends net;\n"
    net_text += "    string file_dir = \"" + d_name + "\";\n"
    net_text += "    int params["+str(tmp_layer_num)+"][32] = '{\n"
    for ci in range(0, tmp_layer_num) :
        pm_list  = npm_list[ci]
        if(ci%2==0) :
            input_addr = "'h800000"
            output_addr = "'hc00000"
        else :
            input_addr = "'hc00000"
            output_addr = "'h800000"
        net_text += "        // layer "+str(ci)+"\n"
        net_text += "    	 '{/*layer*/"+str(pm_list[0])+", "+str(pm_list[1])+", "+str(pm_list[2])+", "
        #net_text += "/*input*/"+(hex(pm_list[3])).replace('0x','\'h')+", "+str(pm_list[4])+", "+str(pm_list[5])+", "+str(pm_list[6])+", "+str(pm_list[7])+", "+str(pm_list[8])+","
        net_text += "/*input*/"+input_addr+", "+str(pm_list[4])+", "+str(pm_list[5])+", "+str(pm_list[6])+", "+str(pm_list[7])+", "+str(pm_list[8])+","
        net_text += "/*weight*/"+(hex(pm_list[9])).replace('0x','\'h')+", "+str(pm_list[10])+", "+str(pm_list[11])+", "+str(pm_list[12])+", "
        net_text += "/*order*/"+(hex(pm_list[13])).replace('0x','\'h')+", "+str(pm_list[14])+", /*add*/"+(hex(pm_list[15])).replace('0x','\'h')+",\n"
        #net_text += "    	  /*out*/"+(hex(pm_list[16])).replace('0x','\'h')+", "+str(pm_list[17])+", "+str(pm_list[18])+", "+str(pm_list[19])+", "+str(pm_list[20])+", "+str(pm_list[21])+", "
        net_text += "    	  /*out*/"+output_addr+", "+str(pm_list[17])+", "+str(pm_list[18])+", "+str(pm_list[19])+", "+str(pm_list[20])+", "+str(pm_list[21])+", "
        net_text += "/*conv*/"+str(pm_list[22])+", "+str(pm_list[23])+", "+str(pm_list[24])+", "
        net_text += "/*bn*/"+str(pm_list[25])+", "+str(pm_list[26])+", "+str(pm_list[27])+", "+str(pm_list[28])+", "
        net_text += "/*pool*/"+str(pm_list[29])+", "+str(pm_list[30])+", "+str(pm_list[31])+" }"
        if ((ci+1)==tmp_layer_num) :
            net_text += "\n"
        else :
            net_text += ",\n"
    net_text += "    };\n\n"
    net_text += "    string file_names["+str(tmp_layer_num)+"][4] = '{\n"
    for ci in range(0, tmp_layer_num) :
        f_list   = nfi_list[ci]
        net_text += "        '{ \""+f_list[0]+"\", \""+f_list[1]+"\", \""+f_list[2]+"\", \""+f_list[3]+"\" }"
        if ((ci+1)==tmp_layer_num) :
            net_text += "\n"
        else :
            net_text += ",\n"
    net_text += "    };\n"
    net_text += "    bit9 wwp["+str(tmp_layer_num)+"]   = '{\n"
    for ci in range(0, tmp_layer_num) :
        wwp_list = nwwp_list[ci]
        net_text += "        '{"
        for wwpi in range(0,16) :
            net_text += "9'b"+wwp_list[wwpi][::-1]
            if (wwpi!=15) :
                net_text += ", "
        net_text += "}"
        if ((ci+1)==tmp_layer_num) :
            net_text += "\n"
        else :
            net_text += ",\n"
    net_text += "    };\n\n"
    net_text += "    function new(string name);\n"
    net_text += "        super.new(name);\n"
    net_text += "    endfunction\n\n"
    net_text += "    function self_config(bit alone);\n"
    net_text += "        layer tmp_layer;\n\n"
    net_text += "        for (int i = 0; i < $size(params); i++) begin\n"
    net_text += "            tmp_layer = new($sformatf(\"%s_layer_%d\", this.name, i), i);\n"
    net_text += "            tmp_layer.config_layer(file_dir, file_names[i], params[i], wwp[i], alone);\n"
    net_text += "            append_layer(tmp_layer);\n"
    net_text += "        end\n"
    net_text += "    endfunction\n"
    net_text += "endclass\n\n\n\n"
    net_text += "class " + test_name + " extends base_seq;\n"
    net_text += "    `uvm_object_utils(" + test_name + ")\n"
    net_text += "    `uvm_declare_p_sequencer(virtual_sequencer)\n\n"
    net_text += "    " + net_name + "  tmp_net ;\n\n"
    net_text += "    function new(string name = \"" + test_name + "\");\n"
    net_text += "        super.new(name);\n"
    net_text += "    endfunction\n\n"
    net_text += "    virtual task body();\n"
    net_text += "        wait(sim_top.rst_n_dma == 1'b0);\n"
    net_text += "        wait(sim_top.rst_n_dma == 1'b1);\n"
    net_text += "        $display(\"***************dma rst done!****************\");\n"
    net_text += "        tmp_net = new(\"" + t_name + "\");\n"
    net_text += "        tmp_net.set_config(32'h0, 96, 1);\n"
    net_text += "        tmp_net.self_config(1);\n"
    net_text += "        start_net(tmp_net, 1, 0, 1);\n"
    net_text += "        repeat(10) @(posedge p_sequencer.u_c_if.clk);\n"
    net_text += "        `uvm_info(get_type_name(), \"Send all master trans done!\", UVM_HIGH);\n"
    net_text += "    endtask\n"
    net_text += "endclass\n"
    net_file.write(net_text)
    net_file.close()
    seq_f   = open(r_dir+"seq.f","a+")
    str_text = seq_f.read()
    serobj = re.search(r'"' + test_name + r'.sv"', str_text, re.M)
    if(serobj == None) :
        seq_f.write("\n`include \""+test_name+".sv\"");
    seq_f.close()


###################################
####        get act data       ####
###################################
def get_act_data(in_fname, quan_level, unsign):
    fin=open(in_fname,'r')
    sourceInLine=fin.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=float(line.strip('\n'))
        if unsign :
            temp2=int(temp1)
        else :
            temp2=int(temp1*(2**quan_level))
        dataset.append(temp2)
    fin.close()
    return dataset
    
def store_act_data(data_array, out_fname, wid, hei, chl, flag, p8_flag, b_flag) :
    if b_flag :
        fout=open(out_fname,'wb')
    else :
        fout=open(out_fname,'w')
    if flag :
        for i in range(0,hei) :
            for j in range(0,wid) :
                for k in range(0,(int((chl-1)/8)+1)*8) :
                    if (k<chl) :
                        if p8_flag :
                            xi = (((k/4)%4)/2)*4 + i
                            xj = (((k/4)%4)%2)*4 + j
                            xk = (k/16)*4 + k%4
                            act_data = data_array[4*hei*wid*xk+2*wid*xi+xj]
                        else :
                            #print(hei, wid, k, j, i)
                            act_data = data_array[hei*wid*k+wid*i+j]
                        if (act_data < 0) :
                            act_data_post=(128-act_data)
                        else :
                            act_data_post=act_data
                        if(act_data_post==256) :
                            act_data_post=255
                        if b_flag :
                            act_data_pp = struct.pack('B', act_data_post)
                            fout.write(act_data_pp)
                        else :
                            fout.write(str(act_data_post))
                            fout.write("\n")
                    else :
                        act_data_pp = struct.pack('B', 0x0)
                        fout.write(act_data_pp)
    else :
        for i in range(0,len(data_array)) :
            act_data = data_array[i]
            if (act_data < 0) :
                act_data_post=(128-act_data)
            else :
                act_data_post=act_data
            fout.write(str(act_data_post))
            fout.write('\n')
    fout.close()

def store_act_data_w(data_array, data_array_wise, out_fname, wid, hei, chl) :
    fout=open(out_fname,'w')
    for i in range(0,hei) :
        for j in range(0,wid) :
            for k in range(0,chl) :
                if ((i/2)%2)==0 :
                    act_data = data_array[(hei/2)*wid*k+wid*((i/2)+(i%2))+j]
                else :
                    act_data = data_array_wise[(hei/2)*wid*k+wid*((i/2)-1+(i%2))+j]
                if (act_data < 0) :
                    act_data_post=(128-act_data)
                    if (act_data_post==256) :
                        act_data_post=255
                else :
                    act_data_post=act_data
                fout.write(str(act_data_post))
                fout.write("\n")
    fout.close()


###################################
####        gen act data       ####
###################################
def gen_act_file(in_file, w_file, out_file, width, height, channel, q_num, turn_flag, pool8_flag, unsign, bin_flag) :
    if (os.path.isfile(in_file)) :
        datas = get_act_data(in_file, q_num, unsign)
        if (w_file != "") :
            datas_w = get_act_data(w_file, q_num, unsign)
            store_act_data_w(datas, datas_w, out_file, width, height, channel)
        else :
            store_act_data(datas, out_file, width, height, channel, turn_flag, pool8_flag, bin_flag)

###################################
####        gen wei data       ####
###################################
def gen_wei_data(wfile, kfile, bfile, xfile, pfile, rfile, ofile, mode, ab_value, ichl, ochl, rchl, pattern, mode_64) :
    if rfile=='' :
        i_flg=0
        i_chl=ichl
    else:
        i_flg=1
        i_chl=rchl
        odata_list=get_order(rfile,ichl)
    if mode_64 :
        group_chl=64
    else :
        group_chl=32
    if (pattern==0) :
        pattern_num=4
    elif (pattern==1) :
        pattern_num=5
    elif (pattern==2) :
        pattern_num=9
    elif (pattern==3) :
        pattern_num=1
    elif (pattern==4) :
        pattern_num=1
    else :
        pattern_num=9
        print("pattern is wrong! Please enter a num between 0 and 4 !")
        
    wdata_list=get_weight(wfile, mode, ab_value, pattern_num)
    xdata_list=get_xvalue(xfile)
    pdata_list=get_pvalue(pfile, 0)
    kdata_list=get_kb(kfile)
    bdata_list=get_kb(bfile)
    out_list=[]
    fout=open(ofile, 'w')
    for i in range(0, ochl) :
        if(len(xdata_list)!=0) :
            idx=0
            for xj in range(0,((len(xdata_list[i])-1)/64+1)*64) :
                if (xj<len(xdata_list[i])) :
                    if i_flg :
                        idx+=(int(xdata_list[i][odata_list[i/group_chl][xj]]) * (2**(xj%4)))
                    else :
                        idx+=(int(xdata_list[i][xj]) * (2**(xj%4)))
                if (xj%4==3) :
                    fout.write(int_to_shex(idx))
                    fout.write('\n')
                    idx=0
        # add value b
        if (len(bdata_list)==0) :
            fout.write('0')
            fout.write('\n')
            fout.write('0')
            fout.write('\n')
            fout.write('0')
            fout.write('\n')
            fout.write('0')
            fout.write('\n')
        else :
            for bj in range(0, 4) :
                if (bj<len(bdata_list[i])) :
                    fout.write(bdata_list[i][len(bdata_list[i])-(bj+1)])
                else :
                    fout.write('0')
                fout.write('\n')
        # add value k
        if (len(kdata_list)==0) :
            fout.write('0')
            fout.write('\n')
            fout.write('0')
            fout.write('\n')
            fout.write('C')
            fout.write('\n')
            fout.write('3')
            fout.write('\n')
        else :
            for kj in range(0, 4) :
                if (kj<len(kdata_list[i])) :
                    fout.write(kdata_list[i][len(kdata_list[i])-(kj+1)])
                else :
                    fout.write('0')
                fout.write('\n')
        # add value weight
        for wj in range(0, ichl) :
            if ((pattern_num==4) or (pattern_num==5)) :
                if i_flg:
                    if ((len(xdata_list)==0) or (int(xdata_list[i][odata_list[i/group_chl][wj]])==1)) :
                        fout.write(pdata_list[i*ichl+odata_list[i/group_chl][wj]])
                        fout.write('\n')
                else :
                    if ((len(xdata_list)==0) or (int(xdata_list[i][wj])==1)) :
                        fout.write(pdata_list[i*ichl+wj])
                        fout.write('\n')
            for k in range(0, pattern_num) :
                if i_flg :
                    if ((len(xdata_list)==0) or (int(xdata_list[i][odata_list[i/group_chl][wj]])==1)) :
                        fout.write(wdata_list[(i*ichl*pattern_num)+(odata_list[i/group_chl][wj]*pattern_num)+k])
                        fout.write('\n')
                else :
                    if ((len(xdata_list)==0) or (int(xdata_list[i][wj])==1)) :
                        fout.write(wdata_list[(i*ichl*pattern_num)+(wj*pattern_num)+k])
                        fout.write('\n')
                #fout.write('\n')
    fout.close()

def gen_wei_data_bin(wfile, kfile, bfile, xfile, pfile, rfile, ofile, mode, ab_value, ichl, ochl, rchl, pattern, mode_64) :
    if rfile=='' :
        i_flg=0
        i_chl=ichl
    else:
        i_flg=1
        i_chl=rchl
        odata_list=get_order(rfile,ichl)
    if mode_64 :
        group_chl=64
    else :
        group_chl=32
    if (pattern==0) :
        pattern_num=4
    elif (pattern==1) :
        pattern_num=5
    elif (pattern==2) :
        pattern_num=9
    elif (pattern==3) :
        pattern_num=1
    elif (pattern==4) :
        pattern_num=1
    else :
        pattern_num=9
        print("pattern is wrong! Please enter a num between 0 and 2 !")
        
    wdata_list=get_weight(wfile, mode, ab_value, pattern_num)
    xdata_list=get_xvalue(xfile)
    pdata_list=get_pvalue(pfile, 1)
    kdata_list=get_kb(kfile)
    bdata_list=get_kb(bfile)
    out_list=[]
    fout=open(ofile, 'w')
    for i in range(0, ochl) :
        wcnt = 0
        wdata = 0
        if(len(xdata_list)!=0) :
            for xj in range(0,((len(xdata_list[i])-1)/64+1)*64) :
                if (xj<len(xdata_list[i])) :
                    if i_flg :
                        #print(i, group_chl, xj)
                        wdata+=(int(xdata_list[i][odata_list[i/group_chl][xj]]) * (2**(xj%8)))
                    else :
                        wdata+=(int(xdata_list[i][xj]) * (2**(xj%8)))
                else :
                    wdata+=0
                if (xj%8==7) :
                    wcnt += 2
                    fout.write(struct.pack('B', wdata))
                    wdata=0
        # add value b
        if (len(bdata_list)==0) :
            wcnt += 2
            fout.write(struct.pack('B', 0))
            wcnt += 2
            fout.write(struct.pack('B', 0))
        else :
            for bj in range(0, 4) :
                if (bj<len(bdata_list[i])) :
                    wdata += shex_to_int(bdata_list[i][len(bdata_list[i])-(bj+1)]) * (16**(wcnt%2))
                else :
                    wdata += 0
                wcnt += 1
                if (wcnt%2==0) :
                    fout.write(struct.pack('B', wdata))
                    wdata = 0
        # add value k
        if (len(kdata_list)==0) :
            wcnt+=2
            fout.write(struct.pack('B', 0x00))
            wcnt+=2
            fout.write(struct.pack('B', 0x3c))
        else :
            for kj in range(0, 4) :
                if (kj<len(kdata_list[i])) :
                    wdata += shex_to_int(kdata_list[i][len(kdata_list[i])-(kj+1)]) * (16**(wcnt%2))
                else :
                    wdata += 0
                wcnt += 1
                if (wcnt%2==0) :
                    fout.write(struct.pack('B', wdata))
                    wdata = 0
        # add value weight
        for wj in range(0, ichl) :
            if ((pattern_num==4) or (pattern_num==5)) :
                if i_flg:
                    if ((len(xdata_list)==0) or (int(xdata_list[i][odata_list[i/group_chl][wj]])==1)) :
                        wdata += pdata_list[i*ichl+odata_list[i/group_chl][wj]] * (16**(wcnt%2))
                        wcnt += 1
                        if (wcnt%2==0) :
                            fout.write(struct.pack('B', wdata))
                            wdata = 0
                else :
                    if ((len(xdata_list)==0) or (int(xdata_list[i][wj])==1)) :
                        wdata += pdata_list[i*ichl+wj] * (16**(wcnt%2))
                        wcnt += 1
                        if (wcnt%2==0) :
                            fout.write(struct.pack('B', wdata))
                            wdata = 0
            for k in range(0, pattern_num) :
                if i_flg :
                    if ((len(xdata_list)==0) or (int(xdata_list[i][odata_list[i/group_chl][wj]])==1)) :
                        wdata += shex_to_int(wdata_list[(i*ichl*pattern_num)+(odata_list[i/group_chl][wj]*pattern_num)+k]) * (16**(wcnt%2))
                        wcnt += 1
                        if (wcnt%2==0) :
                            fout.write(struct.pack('B', wdata))
                            wdata = 0
                else :
                    if ((len(xdata_list)==0) or (int(xdata_list[i][wj])==1)) :
                        #print(i, ichl, pattern_num, wj, k)
                        wdata += shex_to_int(wdata_list[(i*ichl*pattern_num)+(wj*pattern_num)+k]) * (16**(wcnt%2))
                        wcnt += 1
                        if (wcnt%2==0) :
                            fout.write(struct.pack('B', wdata))
                            wdata = 0
        while (wcnt%32 != 0) :
            wdata += 0
            wcnt += 1
            if (wcnt%2==0) :
                fout.write(struct.pack('B', wdata))
                wdata = 0
    fout.close()

def gen_ord_file(rfile, ofile, ichl, ochl, mode_64) :
    odata_list=get_order(rfile,ichl)
    if mode_64 :
        group_chl=64
    else :
        group_chl=32
    fout = open(ofile, "w")
    for rii in range(0,((ochl-1)/group_chl+1)) :
        ord_remain = 0
        ord_cnt = 0
        for rjj in range(0,ichl) :
            ord_data = ord_remain + (1<<(rjj%7))*(odata_list[rii][rjj]%(1<<(8-rjj%7)))
            fout.write(struct.pack('B', ord_data))
            ord_cnt += 1
            ord_remain = odata_list[rii][rjj]/(1<<(8-rjj%7))
            if (rjj%7==6) :
                fout.write(struct.pack('B', ord_remain))
                ord_remain = 0
                ord_cnt += 1
        if (ord_cnt%16!=0) :
            for rkk in range(0, (16-ord_cnt%16)) :
                fout.write(struct.pack('B', ord_remain))
                ord_remain = 0
                ord_cnt += 1
    fout.close()
            


def get_order(o_file, ichl) :
    ordatas=[]
    forin=open(o_file,'r')
    orline=forin.readlines()
    orline_cnt=0
    xtemp3=[]
    for line in orline :
        xtemp1=line.strip('\n')
        xtemp2=int(xtemp1)
        xtemp3.append(xtemp2)
        orline_cnt+=1
        if ((orline_cnt%ichl)==0) :
            ordatas.append(xtemp3)
            orline_cnt=0
            xtemp3=[]
    return ordatas

def get_xvalue(x_file):
    xdatas=[]
    if (x_file=='') :
        return xdatas
    else :
        fxin=open(x_file,'r')
        xline=fxin.readlines()
        for line in xline:
            xtemp1=line.strip('\n')
            xRegex=re.compile(r'\d')
            xtemp2=xRegex.findall(xtemp1)
            #xtemp3=[]
            #for ii in range(0,((len(xtemp2)-1)/4+1)) :
            #    idx=0;
            #    for kk in range(0,4) :
            #        pst=ii*4+kk
            #        if pst<len(xtemp2) :
            #            idx+=int(xtemp2[ii*4+kk])*(2**kk)
            #    xtemp3.append(int_to_shex(idx))
            xdatas.append(xtemp2)
        fxin.close()
        return xdatas

def get_kb(kb_file):
    kbdatas=[]
    if (kb_file=='') :
        return kbdatas
    else :
        fkbin=open(kb_file,'r')
        kbline=fkbin.readlines()
        for line in kbline:
            kbtemp1=line.strip('\n')
            kbdatas.append(kbtemp1)
        fkbin.close()
        return kbdatas
        
def get_pvalue(p_file, bin_flag):
    pdatas=[]
    if (p_file=='') :
        return pdatas
    else :
        pin=open(p_file,'r')
        pline=pin.readlines()
        for line in pline:
            ptemp1=line.strip('\n')
            if (ptemp1=='10') :
                ptemp2='A'
            elif (ptemp1=='11') :
                ptemp2='B'
            elif (ptemp1=='12') :
                ptemp2='C'
            elif (ptemp1=='13') :
                ptemp2='D'
            elif (ptemp1=='14') :
                ptemp2='E'
            elif (ptemp1=='15') :
                ptemp2='F'
            else :
                ptemp2=ptemp1
            if bin_flag :
                pdatas.append(int(ptemp1))
            else :
                pdatas.append(ptemp2)
        pin.close()
        return pdatas

def get_weight(in_file, mode, ab_value, p_num):
    fwin=open(in_file,'r')
    wline=fwin.readlines()
    wdatas=[]
    line_num=0
    for line in wline:
        wtemp1=line.strip('\n')
        wtemp2=trans_wei(wtemp1, mode, ab_value)
        wdatas.append(wtemp2)
    fwin.close()
    return wdatas

def trans_wei(wdata_fp, mode, ab_value):
    if mode==0 :
        wdata_hex=trans_wei_mode0(wdata_fp)
    elif mode==1:
        wdata_hex=trans_wei_mode1(wdata_fp)
    elif mode==2:
        if ab_value==0 :
            wdata_hex=trans_wei_mode2_00(wdata_fp)
        elif ab_value==1 :
            wdata_hex=trans_wei_mode2_01(wdata_fp)
        elif ab_value==2 :
            wdata_hex=trans_wei_mode2_10(wdata_fp)
        elif ab_value==3 :
            wdata_hex=trans_wei_mode2_11(wdata_fp)
        else :
            wdata_hex='X'
            print("ab_value is wrong, please enteer a num between 0 and 3")
    else:
        wdata_hex='X'
        print("mode is wrong, please enter a num between 0 and 2")
    return wdata_hex

def trans_wei_mode0(wdata_fp):
    if ((wdata_fp=="0.0") or (wdata_fp=="-0.0")) :
        return '0'
    elif (wdata_fp=="1.0") :
        return '1'
    elif (wdata_fp=="2.0") :
        return '2'
    elif (wdata_fp=="3.0") :
        return '3'    
    elif (wdata_fp=="4.0") :
        return '4' 
    elif (wdata_fp=="5.0") :
        return '5'
    elif (wdata_fp=="6.0") :
        return '6'
    elif (wdata_fp=="7.0") :
        return '7'    
    elif (wdata_fp=="-1.0") :
        return '9'
    elif (wdata_fp=="-2.0") :
        return 'A'
    elif (wdata_fp=="-3.0") :
        return 'B'    
    elif (wdata_fp=="-4.0") :
        return 'C' 
    elif (wdata_fp=="-5.0") :
        return 'D'
    elif (wdata_fp=="-6.0") :
        return 'E'
    elif (wdata_fp=="-7.0") :
        return 'F'    
    else :
        print('weight data error!' + wdata_fp)
        return '8'
    	  
def trans_wei_mode1(wdata_fp):
    if ((wdata_fp=="0.0") or (wdata_fp=="-0.0")) :
        return '0'
    elif (wdata_fp=="1.0") :
        return '1'
    elif (wdata_fp=="2.0") :
        return '2'
    elif (wdata_fp=="4.0") :
        return '3'    
    elif (wdata_fp=="8.0") :
        return '4' 
    elif (wdata_fp=="16.0") :
        return '5'
    elif (wdata_fp=="32.0") :
        return '6'
    elif (wdata_fp=="64.0") :
        return '7'    
    elif (wdata_fp=="-1.0") :
        return '9'
    elif (wdata_fp=="-2.0") :
        return 'A'
    elif (wdata_fp=="-4.0") :
        return 'B'    
    elif (wdata_fp=="-8.0") :
        return 'C' 
    elif (wdata_fp=="-16.0") :
        return 'D'
    elif (wdata_fp=="-32.0") :
        return 'E'
    elif (wdata_fp=="-64.0") :
        return 'F'    
    else :
        print('weight data error!' + wdata_fp)
        return '8'
        
def trans_wei_mode2_00(wdata_fp):
    if ((wdata_fp=="2.0") or (wdata_fp=="2")) :
        return '0'
    elif ((wdata_fp=="0.0") or (wdata_fp=="-0.0") or (wdata_fp=="0") or (wdata_fp=="-0")) :
        return '1'
    elif ((wdata_fp=="3.0") or (wdata_fp=="3")) :
        return '2'
    elif ((wdata_fp=="4.0") or (wdata_fp=="4")):
        return '3'    
    elif ((wdata_fp=="5.0") or (wdata_fp=="5")) :
        return '4' 
    elif ((wdata_fp=="6.0") or (wdata_fp=="6")) :
        return '5'
    elif ((wdata_fp=="9.0") or (wdata_fp=="9")) :
        return '6'
    elif ((wdata_fp=="10.0") or (wdata_fp=="10")) :
        return '7'    
    elif ((wdata_fp=="-2.0") or (wdata_fp=="-2")) :
        return '8'
    elif ((wdata_fp=="-3.0") or (wdata_fp=="-3")) :
        return 'A'
    elif ((wdata_fp=="-4.0") or (wdata_fp=="-4")) :
        return 'B'    
    elif ((wdata_fp=="-5.0") or (wdata_fp=="-5")) :
        return 'C' 
    elif ((wdata_fp=="-6.0") or (wdata_fp=="-6")) :
        return 'D'
    elif ((wdata_fp=="-9.0") or (wdata_fp=="-9")) :
        return 'E'
    elif ((wdata_fp=="-10.0") or (wdata_fp=="-10")) :
        return 'F'    
    else :
        print('weight data error!' + wdata_fp)
        return 'X'
        
def trans_wei_mode2_01(wdata_fp):
    if (wdata_fp=="3.0") :
        return '0'
    elif (wdata_fp=="5.0") :
        return '1'
    elif (wdata_fp=="4.0") :
        return '2'
    elif ((wdata_fp=="0.0") or (wdata_fp=="-0.0")) :
        return '3'    
    elif (wdata_fp=="6.0") :
        return '4' 
    elif (wdata_fp=="8.0") :
        return '5'
    elif (wdata_fp=="10.0") :
        return '6'
    elif (wdata_fp=="12.0") :
        return '7'    
    elif (wdata_fp=="-3.0") :
        return '8'
    elif (wdata_fp=="-5.0") :
        return '9'
    elif (wdata_fp=="-4.0") :
        return 'A'    
    elif (wdata_fp=="-6.0") :
        return 'C' 
    elif (wdata_fp=="-8.0") :
        return 'D'
    elif (wdata_fp=="-10.0") :
        return 'E'
    elif (wdata_fp=="-12.0") :
        return 'F'    
    else :
        print('weight data error!')
        return 'X'
        
def trans_wei_mode2_10(wdata_fp):
    if (wdata_fp=="3.0") :
        return '0'
    elif (wdata_fp=="4.0") :
        return '1'
    elif (wdata_fp=="5.0") :
        return '2'
    elif (wdata_fp=="6.0") :
        return '3'    
    elif (wdata_fp=="9.0") :
        return '4' 
    elif (wdata_fp=="10.0") :
        return '5'
    elif (wdata_fp=="17.0") :
        return '6'
    elif ((wdata_fp=="0.0")  or (wdata_fp=="-0.0")):
        return '7'    
    elif (wdata_fp=="-3.0") :
        return '8'
    elif (wdata_fp=="-4.0") :
        return '9'
    elif (wdata_fp=="-5.0") :
        return 'A'    
    elif (wdata_fp=="-6.0") :
        return 'B' 
    elif (wdata_fp=="-9.0") :
        return 'C'
    elif (wdata_fp=="-10.0") :
        return 'D'
    elif (wdata_fp=="-17.0") :
        return 'E'    
    else :
        print('weight data error!' + wdata_fp)
        return 'X'

def trans_wei_mode2_11(wdata_fp):
    if (wdata_fp=="4.0") :
        return '0'
    elif ((wdata_fp=="0.0") or (wdata_fp=="-0.0")):
        return '1'
    elif (wdata_fp=="6.0") :
        return '2'
    elif (wdata_fp=="8.0") :
        return '3'    
    elif (wdata_fp=="10.0") :
        return '4' 
    elif (wdata_fp=="12.0") :
        return '5'
    elif (wdata_fp=="18.0") :
        return '6'
    elif (wdata_fp=="20.0") :
        return '7'    
    elif (wdata_fp=="-4.0") :
        return '8'
    elif (wdata_fp=="-6.0") :
        return 'A'    
    elif (wdata_fp=="-8.0") :
        return 'B' 
    elif (wdata_fp=="-10.0") :
        return 'C'
    elif (wdata_fp=="-12.0") :
        return 'D'
    elif (wdata_fp=="-18.0") :
        return 'E'    
    elif (wdata_fp=="-20.0") :
        return 'F' 
    else :
        print('weight data error!' + wdata_fp)
        return 'X'

def shex_to_int(shex):
    if (shex=='0') :
        return 0
    elif (shex=='1') :
        return 1
    elif (shex=='2') :
        return 2
    elif (shex=='3') :
        return 3    
    elif (shex=='4') :
        return 4 
    elif (shex=='5') :
        return 5
    elif (shex=='6') :
        return 6
    elif (shex=='7') :
        return 7    
    elif (shex=='8') :
        return 8
    elif (shex=='9') :
        return 9
    elif ((shex=='A') or (shex=='a')) :
        return 10    
    elif ((shex=='B') or (shex=='b')) :
        return 11 
    elif ((shex=='C') or (shex=='c')) :
        return 12
    elif ((shex=='D') or (shex=='d')) :
        return 13
    elif ((shex=='E') or (shex=='d')) :
        return 14 
    elif ((shex=='F') or (shex=='d')) :
        return 15
    else :
        print('idx data trans error!')
        return 0

def int_to_shex(num):
    if (num==0) :
        return '0'
    elif (num==1) :
        return '1'
    elif (num==2) :
        return '2'
    elif (num==3) :
        return '3'    
    elif (num==4) :
        return '4' 
    elif (num==5) :
        return '5'
    elif (num==6) :
        return '6'
    elif (num==7) :
        return '7'    
    elif (num==8) :
        return '8'
    elif (num==9) :
        return '9'
    elif (num==10) :
        return 'A'    
    elif (num==11) :
        return 'B' 
    elif (num==12) :
        return 'C'
    elif (num==13) :
        return 'D'
    elif (num==14) :
        return 'E' 
    elif (num==15) :
        return 'F'
    else :
        print('idx data trans error!')
        return 'x'


###################################
####        gen_wei_file       ####
###################################
def gen_wei_file(in_file, k_file, b_file, x_file, p_file, or_file, out_file, mode, ab_value, in_channel, out_channel, ker_channel, pattern, mode_64, bin_flag) :
    if (bin_flag==0) :
        gen_wei_data(in_file, k_file, b_file, x_file, p_file, or_file, out_file, mode, ab_value, in_channel, out_channel, ker_channel, pattern, mode_64)
    else :
        gen_wei_data_bin(in_file, k_file, b_file, x_file, p_file, or_file, out_file, mode, ab_value, in_channel, out_channel, ker_channel, pattern, mode_64)


##################################
######### cal base para ##########
##################################
def cal_parameter(p_list, lnum) :
    pool_size = 0
    pool_mode = 0
    if (p_list[16]=="Eltwise") :
        pattern_mode = 6
        pattern_num  = 0
    elif(p_list[16]=="Maxpooling") :
        pattern_mode = 5
        pattern_num  = 0
        pool_mode = 1
        if((p_list[8]=='2') and (p_list[9]=='2') ) :
            pool_size = 0
        elif((p_list[8]=='3') and (p_list[9]=='3') ) :
            pool_size = 1
        elif((p_list[8]=='4') and (p_list[9]=='4') ) :
            pool_size = 2
        elif((p_list[8]=='8') and (p_list[9]=='8') ) :
            pool_size = 3
    elif(p_list[16]=="Average pooling") :
        pattern_mode = 5
        pattern_num  = 0
        pool_mode = 0
        if((p_list[8]=='2') and (p_list[9]=='2') ) :
            pool_size = 0
        elif((p_list[8]=='3') and (p_list[9]=='3') ) :
            pool_size = 1
        elif((p_list[8]=='4') and (p_list[9]=='4') ) :
            pool_size = 2
        elif((p_list[8]=='8') and (p_list[9]=='8') ) :
            pool_size = 3
    elif ((p_list[2]=='1') and (p_list[3]=='1')) :
        pattern_mode = 4
        pattern_num  = 1
    elif ((p_list[8]=='1') and (p_list[9]=='1')) :
        pattern_mode = 3
        pattern_num  = 1
    elif ((p_list[8]=='3') and ((p_list[9]=='3'))) :
        if (p_list[12]=='4') :
            pattern_mode = 0
            pattern_num  = 5
        elif (p_list[12]=='5') :
            pattern_mode = 1
            pattern_num  = 6
        elif (p_list[12]=='9') :
            pattern_mode = 2
            pattern_num  = 9
        else :
            pattern_mode = 2
            pattern_num  = 9
    else :
        pattern_mode = 4
        pattern_num  = 1
    if (pattern_mode<4) and (p_list[3]==p_list[5]) and (p_list[3]!='1') :
        filter_str = 0
    else :
        filter_str = 1

    #order_addr = '5000'
    #act_addr0 = '300000'
    feature_hor = int(p_list[3])
    feature_ver = int(p_list[2])
    in_channel  = int(p_list[0])
    last_out_channel = ((in_channel-1)/8+1)*8
    quan_code_27fi = int(p_list[6])
    index_en = int(p_list[18])
    if ((p_list[13]!='x') and (pattern_mode != 4) and (index_en == 1)) :
        ker_pruning_num = int(p_list[13])
    else :
        ker_pruning_num = in_channel
    #weight_addr0 = '10000'
    weight_stride = (( pattern_num * ker_pruning_num + 8 + index_en * (((in_channel-1)/64+1)*16 -1) ) / 32 + 1) * 16
    if (p_list[17]=='1') :
        order_en = 1
    else :
        order_en = 0
    #act_addr_1 = '940000'
    #out_addr = '620000'
    out_hor = int(p_list[5])
    out_ver = int(p_list[4])
    out_chl = int(p_list[1])
    quan_code_16fp = int(p_list[7]) 
    pool_8x8_en = 0
    if(pattern_mode==6) :
        alu_mode = 3
        alu_ab_value = 0
    elif (p_list[11] == 'MIX') :
        alu_mode = 2
        alu_ab_value = 0
    elif (p_list[11] == 'Pow2') :
        alu_mode = 1
        alu_ab_value = 0
    elif (p_list[11] == 'FIX') :
        alu_mode = 0
        alu_ab_value = 0
    else :
        alu_mode = 2
        alu_ab_value = 0
    quan_code_8fi = int(p_list[6])
    if(pattern_mode<=4) :
        mul_en = 1
        sum_en = 1
    else:
        mul_en = 0
        sum_en = 0
    if (p_list[10] == '1') :
        relu_en = 1
    else :
        relu_en = 0
    overflow_sel = 0
    pool_up_or_down = 0
    if((pattern_mode<=3) and ((in_channel>128) or (out_chl>256))) :
        out_mode = 1
    else :
        out_mode = 0

    act0_file = "layer"+str(lnum)+"_input_act0_file.bin"
    if(pattern_mode==6) :
        act1_file = "layer"+str(lnum)+"_input_act1_file.bin"
    else :
        act1_file = ""
    if(pattern_mode<=4) :
        wei_file  = "layer"+str(lnum)+"_input_weight_file.bin"
    else :
        wei_file  = ""
    if (order_en==1) :
        ord_file = "layer"+str(lnum)+"_input_order_file.bin"
    else :
        ord_file = ""


    act_addr_0 = 0x720
    act_num = feature_hor * in_channel * feature_ver
    act_addr_1 = act_addr_0 + act_num
    if(pattern_mode==6) :
        weight_addr = act_addr_1 + act_num
    else :
        weight_addr = act_addr_0 + act_num
    order_addr = weight_addr + out_chl * weight_stride
    out_addr = 0x60000000

    w_pattern=[]
    for wm in range(0,16) :
        w_pattern.append(p_list[19+wm])

    #file_dir  = "./"
    file_list = []
    para_list = []
    wwp_list  = w_pattern
    file_list.append(act0_file)
    file_list.append(act1_file)
    file_list.append(wei_file)
    file_list.append(ord_file)
    para_list.append(pattern_mode)  #0
    para_list.append(filter_str)  #1
    para_list.append(out_mode)  #2
    para_list.append(act_addr_0)  #3
    para_list.append(feature_hor)  #4
    para_list.append(feature_ver)  #5
    para_list.append(in_channel)  #6
    para_list.append(last_out_channel)  #7
    para_list.append(quan_code_27fi)  #8
    para_list.append(weight_addr)  #9
    para_list.append(weight_stride)  #10
    para_list.append(index_en)  #11
    para_list.append(ker_pruning_num)  #12
    para_list.append(order_addr)  #13
    para_list.append(order_en)  #14
    para_list.append(act_addr_1)  #15
    para_list.append(out_addr)  #16
    para_list.append(out_hor)  #17
    para_list.append(out_ver)  #18
    para_list.append(out_chl)  #19
    para_list.append(quan_code_16fp)  #20
    para_list.append(pool_8x8_en)  #21
    para_list.append(alu_mode)  #22
    para_list.append(alu_ab_value)  #23
    para_list.append(quan_code_8fi)  #24
    para_list.append(mul_en)  #25
    para_list.append(sum_en)  #26
    para_list.append(relu_en)  #27
    para_list.append(overflow_sel)  #28
    para_list.append(pool_size)  #29
    para_list.append(pool_mode)  #30
    para_list.append(pool_up_or_down)  #31
    total_list=[]
    total_list.append(file_list)
    total_list.append(para_list)
    total_list.append(wwp_list)
    return total_list

def adjust_addr(in_list) :
    out_list = []
    act0_addr_list = []
    act1_addr_list = []
    wei_addr_list = []
    ord_addr_list = []
    out_addr_list = []
    tmp_addr = 0x60 * len(in_list)
    for ii in range(0,len(in_list)) :
        wei_addr_list.append(tmp_addr)
        pattern_mode  = in_list[ii][0]
        out_chl       = in_list[ii][19]
        weight_stride = in_list[ii][10]
        if(pattern_mode<=4) :
            tmp_addr = tmp_addr + out_chl * weight_stride
    for ii in range(0,len(in_list)) :
        order_en    = in_list[ii][14]
        in_channel  = in_list[ii][6]
        ord_addr_list.append(tmp_addr)
        if (order_en) :
            tmp_addr = tmp_addr + (in_channel/14+1) * 16
    act0_addr_list.append(tmp_addr)
    tmp_addr = 0x61000000
    for ii in range(0,len(in_list)-1) :
        out_addr_list.append(tmp_addr)
        act0_addr_list.append(tmp_addr)
        tmp_addr = tmp_addr + 0x1000000
    out_addr_list.append(0x60000000)
    for ii in range(0,len(in_list)) :
        act1_addr_list.append(0x0)

    for ii in range(0,len(in_list)) :
        tmp_out_list = []
        for jj in range(0,32) :
            if (jj == 3) :
                tmp_out_list.append(act0_addr_list[ii])
            elif (jj == 9) :
                tmp_out_list.append(wei_addr_list[ii])
            elif (jj == 13) :
                tmp_out_list.append(ord_addr_list[ii])
            elif (jj == 15) :
                tmp_out_list.append(act1_addr_list[ii])
            elif (jj == 16) :
                tmp_out_list.append(out_addr_list[ii])
            else :
                tmp_out_list.append(in_list[ii][jj])
        out_list.append(tmp_out_list)

    return out_list
        


##################################
########## decode file ###########
##################################
def dec_file(dir_name, file_name) :
    para_list=[]
    config_file = open(dir_name+file_name, "r")
    str_text  = config_file.read()
    #print str_text
    #temp_text = "input channel num  = 3\noutput channel num = 64\n"
    ### 0 - in_channel
    searchobj = re.search(r'input channel num *= *(\d*) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 1 - output_channel
    searchobj = re.search(r'output channel num *= *(\d*) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 2 - feature_ver
    searchobj = re.search(r'input feature_h *= *(\d*) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 3 - feature_hor
    searchobj = re.search(r'input feature_w *= *(\d*) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 4 - filter_str
    searchobj = re.search(r'output feature_h *= *(\d*) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 5 - filter_str
    searchobj = re.search(r'output feature_w *= *(\d*) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 6 - quan_code_27fi , quan_code_8fi
    searchobj = re.search(r'in_q *= *(\d*) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 7 - quan_code_16fp
    searchobj = re.search(r'out_q *= *(\d*) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 8 9 - pattern_mode
    searchobj = re.search(r'kernel size *= *(\d*) *.*\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
        para_list.append(searchobj.group(1))
    ### 10 - relu_en
    searchobj = re.search(r'relu *(√|1) *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('0')
    else :
        para_list.append('1')
    ### 11 - alu_mode, alu_ab_value
    searchobj = re.search(r'w_quan *\"(\w*)\" *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 12 - pattern_mode
    searchobj = re.search(r'pattern: *(\d*)-pattern *\n', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 13 - ker_pruning_num
    searchobj = re.search(r'connect_num *= *(\d*) *', str_text, re.M)
    if searchobj == None :
        para_list.append('x')
    else :
        para_list.append(searchobj.group(1))
    ### 14 - bn_k
    searchobj = re.search(r'k *= *(\S*) *(\(0x\w*\))?', str_text, re.M)
    if searchobj == None :
        para_list.append('xxxx')
    elif searchobj.group(1) == '0' :
        para_list.append('0000')
    else :
        para_list.append(searchobj.group(2)[3:len(searchobj.group(2))-1])
    ### 15 - bn_b
    searchobj = re.search(r'b *= *(\S*) *(\(0x\w*\))?', str_text, re.M)
    if searchobj == None :
        para_list.append('xxxx')
    elif searchobj.group(1) == '0' :
        para_list.append('0000')
    else :
        para_list.append(searchobj.group(2)[3:len(searchobj.group(2))-1])
    ### 16 - pattern_mode
    searchobj = re.search(r' *(Maxpooling|Average pooling|Eltwise)', str_text, re.M)
    if searchobj == None :
        para_list.append('Conv')
    else :
        para_list.append(searchobj.group(1))
    ### 35 - act0 file
    searchobj = re.search(r'act0 file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        act0_file = 'input_x.txt'
    else :
        act0_file = searchobj.group(1)
    ### 36 - act1 file
    searchobj = re.search(r'act1 file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        act1_file = 'input_y.txt'
    else :
        act1_file = searchobj.group(1)
    ### 37 - wei file
    searchobj = re.search(r'weight file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        wei_file = 'None'
    else :
        wei_file = searchobj.group(1)
    ### 38 - bn k file
    searchobj = re.search(r'bn k file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        bnk_file = 'bn_k.txt'
    else :
        bnk_file = searchobj.group(1)
    ### 39 - wei file
    searchobj = re.search(r'bn b file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        bnb_file = 'bn_b.txt'
    else :
        bnb_file = searchobj.group(1)
    ### 40 - order file
    searchobj = re.search(r'order file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        ord_file = 'Order.txt'
    else :
        ord_file = searchobj.group(1)
    ### 41 - index file
    searchobj = re.search(r'index file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        index_file = 'pattern_idx.txt'
    else :
        index_file = searchobj.group(1)
    ### 42 - kernel pruning idx file
    searchobj = re.search(r'kernel pruning idx file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        idx_file = 'kernel_pruning_idx.txt'
    else :
        idx_file = searchobj.group(1)
    ### 43 - kernel pruning idx file
    searchobj = re.search(r'output file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        out_file = 'output.txt'
    else :
        out_file = searchobj.group(1)
    ### 44 - index file
    searchobj = re.search(r'pattern dic file *: *(\S*) *\n', str_text, re.M)
    if searchobj == None :
        dic_file = 'pattern_dic.txt'
    else :
        dic_file = searchobj.group(1)

    ### 17 - order_en
    if (ord_file == "None") :
        para_list.append('0')
    elif (os.path.isfile(dir_name+ord_file)) :
        para_list.append('1')
    else :
        para_list.append('0')
    ### 18 - index_en
    if (idx_file == "None") :
        para_list.append('0')
    elif (os.path.isfile(dir_name+idx_file)) :
        para_list.append('1')
    else :
        para_list.append('0')
    
    ### 19-34 - pattern_dic
    if (idx_file == "None") :
        for dm in range(0,16) :
            para_list.append('111111111')
    elif (os.path.isfile(dir_name+dic_file)) :
        dic_file=open(dir_name+dic_file, "r")
        for dm in range(0,16) :
            dic_text=dic_file.readline()
            searchobj = re.search(r'\d* *: *(\d{9}) *\n', dic_text, re.M)
            if searchobj == None :
                print('File pattern_dic.txt is error!')
                para_list.append('111111111')
            else :
                para_list.append(searchobj.group(1))
        dic_file.close()
    else :
        for dm in range(0,16) :
            para_list.append('111111111')

    ### 35-43 - act0 file
    para_list.append(act0_file)
    para_list.append(act1_file)
    para_list.append(wei_file)
    para_list.append(bnk_file)
    para_list.append(bnb_file)
    para_list.append(ord_file)
    para_list.append(index_file)
    para_list.append(idx_file)
    para_list.append(out_file)
    config_file.close()
    return para_list

####################################
############## one task ############
####################################
def deal_one_task(dir_name, tc_name, muse_en, rtl_en, fpga_en, muse_dir, rtl_dir) :
    if (not os.path.isfile(dir_name + "config.txt")) :
        return 0
    else :
        if (not os.path.exists(dir_name + "data_for_fpga/")) :
            os.mkdir(dir_name + "data_for_fpga/")
        nf_list = []
        np_list = []
        nw_list = []
        np_list_new = []
        p_list  = []
        fact0_list =[]
        fact1_list =[]
        fwei_list =[]
        fbnk_list = []
        fbnb_list = []
        ford_list =[]
        findex_list =[]
        fidx_list =[]
        fout_list = []
        tmp_net = []
        lcc = 0
        conf_file = "config.txt"
        while os.path.isfile(dir_name + conf_file) :
            t_list = []
            ### analysis config file : get base para list from config file
            p_list_s = dec_file(dir_name, conf_file)
            p_list.append(p_list_s)
            cfg_file = open("./cfg_"+str(lcc)+".txt", "w")
            for m in range(0,len(p_list_s)):
                cfg_file.write(str(p_list_s[m]) + "\n")
            cfg_file.close()
            ### analysis parameter : cal more detail parameters list from p_list to t_list
            t_list = cal_parameter(p_list_s, lcc)
            nf_list.append(t_list[0])
            np_list.append(t_list[1])
            nw_list.append(t_list[2])
            lcc+=1
            conf_file = "config_"+str(lcc)+".txt"
            ### build network model : Build a layer model class according to the parameters
        np_list_new = adjust_addr(np_list)
        for lc in range(0,len(p_list)) :
            p_list_ns = p_list[lc]
            fi_list_ns = nf_list[lc]
            pm_list_ns = np_list_new[lc]
            ww_list_ns = nw_list[lc]
            new_layer = Layer("new_layer", 0)
            new_layer.config_layer(dir_name+"data_for_fpga/", fi_list_ns, pm_list_ns, ww_list_ns, 1)
            tmp_net.append(new_layer)
            ### supplement some file : add bnk/bnb file if it is not exist but need 
            if (new_layer.pattern_mode < 5) and (not os.path.isfile(dir_name+p_list_ns[38])) :
                k_file = open(dir_name+p_list_ns[38], 'w')
                for kk in range(0,int(p_list_ns[1])) :
                    if (p_list_ns[14]=='xxxx') :
                        k_file.write("211F\n")
                    else :
                        k_file.write(p_list_ns[14]+'\n')
                k_file.close()
            if (new_layer.pattern_mode < 5) and (not os.path.isfile(dir_name+p_list_ns[39])) :
                b_file = open(dir_name+p_list_ns[39], 'w')
                for bb in range(0,int(p_list_ns[1])) :
                    if (p_list[15]=='xxxx') :
                        b_file.write("0000\n")
                    else :
                        b_file.write(p_list_ns[15]+'\n')
                b_file.close()
            ### base files name : get files path  
            act0_file   = dir_name + p_list_ns[35]
            act1_file    = dir_name + p_list_ns[36]
            if (p_list_ns[37] != "None") :
                wei_file  = dir_name + p_list_ns[37]
            elif (new_layer.pattern_mode < 2) :
                wei_file  = dir_name + "weight_nonzero.txt"
            else :
                wei_file  = dir_name + "weight.txt"
            k_file    = dir_name + p_list_ns[38]
            b_file    = dir_name + p_list_ns[39]
            ord_file  = dir_name + p_list_ns[40]
            if (not os.path.isfile(ord_file)) :
                ord_file  = ""
            index_file= dir_name + p_list_ns[41]
            if (not os.path.isfile(index_file)) :
                index_file  = ""
            idx_file  = dir_name + p_list_ns[42]
            if (not os.path.isfile(idx_file)) :
                idx_file = ""
            out_file  = dir_name + p_list_ns[43]
            fact0_list.append(act0_file)
            fact1_list.append(act1_file)
            fwei_list.append(wei_file)
            fbnk_list.append(k_file)
            fbnb_list.append(b_file)
            ford_list.append(ord_file)
            findex_list.append(index_file)
            fidx_list.append(idx_file)
            fout_list.append(out_file)
        ##### rtl verify file
        if (muse_en) :
            ## act file
            #gen_act_file(in_file, w_file, o_file, width, height, channel, q_num, turn_flag, pool8_flag, unsign, bin_flag)
            ## wei file
            #gen_wei_file()
            ## ord file
            #gen_ord_file(ord_file, )
            ## out file
            #gen_act_file()
            #for tmp_i in range(0,len(tmp_net)) :
            #    if tmp_i == 0 :
            #        tmp_layer_name = ""
            #    else :
            #        tmp_layer_name = "_layer" + str(tmp_i)
            #    gen_seq_file(muse_dir+"/seq/", tc_name.replace('.','_').replace('-','_')+tmp_layer_name , tmp_net[i], act0_list[i], act1_list[i], ord_list[i], wei_list[i], index_list[i], idx_list[i], bnk_list[i], bnb_list[i], out_list[i])
            #    gen_tc_file(muse_dir+"/tc/", tc_name.replace('.','_').replace('-','_')+tmp_layer_name)
            gen_seq_file(muse_dir+"/seq/", tc_name.replace('.','_').replace('-','_') , tmp_net, fact0_list, fact1_list, ford_list, fwei_list, findex_list, fidx_list, fbnk_list, fbnb_list, fout_list)
            gen_tc_file(muse_dir+"/tc/", tc_name.replace('.','_').replace('-','_'))

        ###### fpga verify file
        if (fpga_en) :
            gen_net_file_cpp(tc_name.replace('.','_').replace('-','_'), dir_name+"data_for_fpga/", nf_list, np_list_new, nw_list, 1)
        if (rtl_en) :
            gen_net_file_sv(tc_name.replace('.','_').replace('-','_'), dir_name+"data_for_fpga/", rtl_dir+"/seq/", nf_list, np_list_new, nw_list, 1)
            gen_tc_file(rtl_dir+"/tc/", tc_name.replace('.','_').replace('-','_'))
        if (fpga_en or rtl_en) :
            for tmp_i in range(0,len(tmp_net)) :
                print("Start generate layer " +str(tmp_i)+ " data!")
                if(len(tmp_net)==0):
                    tmp_layer_name = ""
                else:
                    tmp_layer_name = "layer" + str(tmp_i)
                tmp_layer = tmp_net[tmp_i]
                # act0 file
                if (tmp_layer.pattern_mode == 6) :
                    act0_bfile = dir_name + "data_for_fpga/" + tmp_layer_name + "_input_act0_file.bin"
                    gen_act_file(fact0_list[tmp_i], "", act0_bfile, tmp_layer.feature_hor, tmp_layer.feature_ver/2, tmp_layer.in_channel, tmp_layer.quan_code_27fi, 1, 0, 0, 1)
                    # act1 file
                    act1_bfile = dir_name + "data_for_fpga/" + tmp_layer_name + "_input_act1_file.bin"
                    gen_act_file(fact1_list[tmp_i], "", act1_bfile, tmp_layer.feature_hor, tmp_layer.feature_ver/2, tmp_layer.in_channel, tmp_layer.quan_code_27fi, 1, 0, 0, 1)
                else :
                    act0_bfile = dir_name + "data_for_fpga/" + tmp_layer_name + "_input_act0_file.bin"
                    gen_act_file(fact0_list[tmp_i], "", act0_bfile, tmp_layer.feature_hor, tmp_layer.feature_ver, tmp_layer.in_channel, tmp_layer.quan_code_27fi, 1, 0, 0, 1)
                # wei file
                if (tmp_layer.pattern_mode <= 4) :
                    wei_bfile = dir_name + "data_for_fpga/" + tmp_layer_name + "_input_weight_file.bin"
                    gen_wei_file(fwei_list[tmp_i], fbnk_list[tmp_i], fbnb_list[tmp_i], fidx_list[tmp_i], findex_list[tmp_i], ford_list[tmp_i], wei_bfile, tmp_layer.alu_mode, tmp_layer.alu_ab_value, tmp_layer.in_channel, tmp_layer.out_channel, tmp_layer.ker_pruning_num, tmp_layer.pattern_mode, tmp_layer.rd_chl_mode, 1)
                    print("ker_pruning_num is " + str(tmp_layer.ker_pruning_num))
                # ord file
                if (ford_list[tmp_i] != "") :
                    ord_bfile = dir_name + "data_for_fpga/" + tmp_layer_name + "_input_order_file.bin"
                    gen_ord_file(ford_list[tmp_i], ord_bfile, tmp_layer.order_real_num, tmp_layer.out_channel, tmp_layer.rd_chl_mode)
                # out file
                out_bfile = dir_name + "data_for_fpga/" + tmp_layer_name + "_output_file.bin"
                gen_act_file(fout_list[tmp_i], "", out_bfile, tmp_layer.out_hor, tmp_layer.out_ver, tmp_layer.out_chl, tmp_layer.quan_code_16fp, 1, 0, 0, 1)
                ## rtl out file
                #rtl_bfile = dir_name + "data_for_fpga/" + "output_file_rtl.bin"
                #gen_act_file(rtl_file, "", rtl_bfile, tmp_layer.out_hor, tmp_layer.out_ver, tmp_layer.out_chl, tmp_layer.quan_code_16fp, 1, 0, 1, 1)
        return 1
        
def gen_regression_file(tc_dir, tc_name, tc_list, mv_flag, dir_list) :
    fr = open(tc_dir + "/sim/" + tc_name + ".sh", "w")
    for tt in range(0, len(tc_list)) :
        fr.write("make sim tc=" + tc_list[tt].replace('.','_').replace('-','_') + "_test\n")
        if (mv_flag==1) :
            fr.write("cp rtl_out_file_layer_0.bin " + dir_list[tt] + "data_for_fpga/rtl_out_file.bin\n")
    fr.close()

def min_num(a, b) :
    if (a>b) :
        return b
    else :
        return a



##################################
############## main ##############
##################################
opt = option_parse()
dname = opt.dir_name 
tname  = opt.test_name
dlist  = []
tlist  = []
print("now inside " + dname + ":" + tname)
isdone = deal_one_task(dname+"/", tname, opt.muse_flag, opt.rtl_flag, opt.fpga_flag, opt.muse_dir, opt.rtl_dir)
if (isdone) :
    dlist.append(dname+"/")
    tlist.append(tname)
for foldername, subfolders, filenames in os.walk(dname):
    for subfolder in subfolders:
        d_name = foldername + "/" + subfolder + "/"
        t_name = subfolder
        print("now inside " + foldername + ":" + subfolder)
        isdone = deal_one_task(d_name, t_name, opt.muse_flag, opt.rtl_flag, opt.fpga_flag, opt.muse_dir, opt.rtl_dir)
        if (isdone) :
            dlist.append(d_name)
            tlist.append(t_name)
if (opt.muse_flag) :
    gen_regression_file(opt.muse_dir, tname, tlist, 0, "")
if (opt.rtl_flag) :
    gen_regression_file(opt.rtl_dir, tname, tlist, 1, dlist)


