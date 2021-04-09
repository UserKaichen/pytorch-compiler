''' Very Naive API using .csv '''

import csv
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from math import ceil

from meta_model import Design, Workload, LoopParameter, ReorderCase, Block, AnalysisCase
from config import Energy_DRAM, Energy_GRS, Energy_MAC, get_Energy_RF, Energy_AL1_Rd, Energy_AL1_Wr, Energy_AL2_Rd, Energy_AL2_Wr, Energy_WL1_Rd, Energy_WL1_Wr, Energy_OL2_Rd, Energy_OL2_Wr, OL1_choices_map
from sram_access import SramAccess
from overhead_eval import OverheadEval
from generate_case import pick_possible_designs, get_analysis_cases, get_design_dict
from typing import List, Dict, DefaultDict
import size


req_mac = 4096
design_dict: DefaultDict[int, List[Design]] = get_design_dict()
designs: List[Design] = pick_possible_designs(design_dict, req_mac)

def search(workload: Workload, writer: csv.DictWriter, note: str):
    for design in tqdm(designs, desc='Try Different Hardware Parallel Designs'):
        sram_cases: List[AnalysisCase] = get_analysis_cases(design, workload)
        for sram_case in sram_cases:
            evaluator = OverheadEval(sram_case, design)
            total_memory, total_memory_access, chiplet_communication, total_runtime, area_per_chiplet, area_per_package, \
                total_energy, Energy_DRAMtoSRAM_W, Energy_DRAMtoSRAM_A, energy_breakdown = evaluator.evaluation()

            dram_access = evaluator.get_dram_communication()
            
            view_module_energy = 0
            if view_module_energy:
                print("\n")
                print('Chip to Chip: %f' % Energy_GRS)
                print('DRAM to SRAM_A: %f' % (Energy_GRS + Energy_DRAM + Energy_AL2_Wr))
                print('DRAM to SRAM_W: %f' % (Energy_GRS + Energy_DRAM + Energy_WL1_Wr))
                print('AL2 to AL1: %f' % (Energy_AL2_Rd + Energy_AL1_Wr))
                print('AL1 to MAC: %f' % (Energy_WL1_Rd))
                print('WL1 to MAC: %f' % (Energy_AL1_Rd))
                print('MAC to OL1(RF): %f' % (get_Energy_RF(size.B(384))))
                print('OL1(RF) to OL2: %f' % (Energy_OL2_Wr + get_Energy_RF(sram_case.OL1)))
                print('OL2 to DRAM: %f' % (Energy_GRS + Energy_DRAM + Energy_OL2_Rd))
                print('AL1 Size: %s' % str(sram_case.AL1))
                print('AL2 Size: %s' % str(sram_case.AL2))
                print('WL1 Size: %s' % str(sram_case.WL1))
                print('Area of 1 Chiplet: %s' % str(area_per_chiplet / 1000000))
                sys.exit()

            writer.writerow({'Chiplet': design.chiplet,
                             'Core': design.core,
                             'Lane': design.lane,
                             'Vector_Size': design.vector,
                             'cin': workload.in_channel,
                             'cout': workload.out_channel,
                             'out_size': workload.out_size.H,
                             'kernel_size': workload.kernel_size.H,
                             'stride': workload.stride.H,
                             'OL1': sram_case.OL1.to_b(),
                             'AL1': sram_case.AL1.to_b(),
                             'WL1': sram_case.WL1.to_b(),
                             'real_WL1': evaluator.real_WL1.to_b(),
                             'AL2': sram_case.AL2.to_b(),
                             'reorder_case': sram_case.reorderCase.type_n,
                             'rotation_enable': sram_case.loopParameter.rotation_enable,
                             'runtime': total_runtime,
                             'area-chiplet': area_per_chiplet,
                             'area-package': area_per_package,
                             'total_memory_footprint': total_memory,
                             'total_energy': total_energy,
                             'chiplet_communication': chiplet_communication.to_b(),
                             'dram_communication': dram_access.to_b(),
                             'DRAM_energy_A': Energy_DRAMtoSRAM_A,
                             'DRAM_energy_W': Energy_DRAMtoSRAM_W,
                             'DRAM_Energy': energy_breakdown.DRAM_energy,
                             'Die-to-Die_Energy': energy_breakdown.D2D_energy,
                             'A-L2_Energy': energy_breakdown.A_L2_energy,
                             'A-L1_Energy': energy_breakdown.A_L1_energy,
                             'W-L1_Energy': energy_breakdown.W_L1_energy,
                             'Output_Energy': energy_breakdown.output_energy,
                             'MAC_Energy': energy_breakdown.Energy_TotalMAC,
                             'MOL1': total_memory_access.OL1_Wr.to_b(),
                             'MAL1': total_memory_access.AL1_Wr.to_b(),
                             'MWL1': total_memory_access.WL1_Wr.to_b(),
                             'sMWL1': evaluator.get_sram_access().WL1_Wr,
                             'MAL2': total_memory_access.AL2_Wr.to_b(),
                             'X1': sram_case.loopParameter.W1,
                             'Y1': sram_case.loopParameter.H1,
                             'K1': sram_case.loopParameter.K1,
                             'X2': sram_case.loopParameter.W2,
                             'Y2': sram_case.loopParameter.H2,
                             'K2': sram_case.loopParameter.K2,
                             'Kp': sram_case.loopParameter.package_spatial_parameter.Kp,
                             'Yp': sram_case.loopParameter.package_spatial_parameter.Hp,
                             'Kc': sram_case.loopParameter.chiplet_spatial_parameter.Kc,
                             'Yc': sram_case.loopParameter.chiplet_spatial_parameter.Hc,
                             'Xc': sram_case.loopParameter.chiplet_spatial_parameter.Wc,
                             'C1': sram_case.loopParameter.C1,
                             'C0': sram_case.loopParameter.C0,
                             'Csa': sram_case.loopParameter.Csa,
                            #  'Ksw': sram_case.loopParameter.Ksw,
                             'X0': OL1_choices_map[sram_case.OL1].W,
                             'Y0': OL1_choices_map[sram_case.OL1].H,
                             'note': note
                             })

def main():
    dtype = {'cin': np.int64, 'in_size': np.int64, 'cout': np.int64, 'out_size': np.int64, 'kernel_size': np.int64,
             'stride': np.int64, 'note': str}
    fieldnames = ['Chiplet', 'Core', 'Lane', 'Vector_Size', 
                  'cin', 'cout', 'out_size', 'kernel_size', 'stride',
                  'OL1', 'AL1', 'WL1', 'real_WL1', 'AL2', 'reorder_case',
                  'rotation_enable',
                  'runtime',
                  'area-chiplet', 'area-package', 'total_memory_footprint', 'total_energy',
                  'chiplet_communication', 'dram_communication', 'DRAM_energy_A', 'DRAM_energy_W', 'DRAM_Energy',
                  'Die-to-Die_Energy', 'A-L2_Energy', 'A-L1_Energy', 'W-L1_Energy', 'Output_Energy', 'MAC_Energy',
                  'MOL1', 'MAL1', 'MWL1', 'sMWL1', 'MAL2', 'X1', 'Y1', 'K1',
                  'X2', 'Y2', 'K2', 'Kp', 'Yp', 'Kc', 'Yc', 'Xc', 'C1', 'C0', 'Csa', 'X0', 'Y0', 'note']

    if len(sys.argv) == 3:
        net = sys.argv[1]
        size = sys.argv[2]
    else:
        net = "VGG"
        size = "224"

    for model_name in [net]:
        for model_size in [size]:
            data = pd.read_csv(f'workload/{model_name}{model_size}.csv', dtype=dtype)
            with open(f'csv/raw4/output/{model_name}-{model_size}.csv', 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for _, row in data.iterrows():
                    cin = row['cin']
                    cout = row['cout']
                    out_size = row['out_size']
                    kernel_size = row['kernel_size']
                    stride = row['stride']
                    note = row['note']

                    workload: Workload = Workload(cin, cout, Block(out_size, out_size), Block(kernel_size, kernel_size), Block(stride, stride))
                    search(workload, writer, note)

if __name__ == '__main__':
    main()
    print(str('\n') + str('==' * 50) + str('\n') + str(' ' * 35) + str('NN-Baton finishes search\n') + str('==' * 50))
