"""Define the hardware-related configurations"""

from collections import defaultdict
import numpy as np
import size
from typing import List, Dict, DefaultDict
from dataclasses import dataclass
from meta_model import PSpatialParameter, CSpatialParameter, Block, TileSize, ForLoopSymbol


DATA_WIDTH : float = 8
WEIGHT_WIDTH : float = 8
W_BW_RATIO : float = WEIGHT_WIDTH / 8
A_BW_RATIO : float = DATA_WIDTH / 8

global ACT_MEMORY_ALIGN
if DATA_WIDTH == 16:
    ACT_MEMORY_ALIGN = 8
elif DATA_WIDTH == 8 or DATA_WIDTH == 4:
    ACT_MEMORY_ALIGN = 16
else:
    raise ValueError('The DATA-WIDTH only supports 4, 8, and 16bit')

rotation_search_list = [False]


# One tile may contains several sub-tiles.
# For the plane-dimension, we need to define the   
# specific H/W dimension under a specific H*W case.
@dataclass(frozen=True)
class CutWay:
    H: int
    W: int

# This division is for core mapping.
# We have eight cores, and we define the H:W 
# under a given H*W case.
core_choices_map: Dict[int, CutWay] = {
    1:  CutWay(1, 1),
    2:  CutWay(1, 2),
    4:  CutWay(2, 2),
    8:  CutWay(4, 2),
}

# This division is for the basic workload mapping.
# With a given core-workload, it may contain several
# basic workload across the channel and plane dimension.
W1H1_choices_map: Dict[int, CutWay] = {
    1:  CutWay(1, 1),
    2:  CutWay(1, 2),
    4:  CutWay(2, 2),
    8:  CutWay(2, 4),
    16: CutWay(4, 4),
}

# 4*MUSE-v3 System Configurations (1024-MAC/Chiplet, 4096-MAC/System):
# 8b-A, 8b-W Mode Configurations:
num_chiplets:   List[int] = [4]
num_cores:      List[int] = [8]
num_lanes:      List[int] = [16]
size_vectors:   List[int] = [8]

# 16b-A, 8b-W Mode Configurations:
# num_chiplets:   List[int] = [4]
# num_cores:      List[int] = [8]
# num_lanes:      List[int] = [8]
# size_vectors:   List[int] = [8]

# 8b-A, 4b-W Mode Configurations:
# num_chiplets:   List[int] = [4]
# num_cores:      List[int] = [8]
# num_lanes:      List[int] = [16]
# size_vectors:   List[int] = [8]

# 4b-A, 4b-W Mode Configurations:
# num_chiplets:   List[int] = [4]
# num_cores:      List[int] = [8]
# num_lanes:      List[int] = [16]
# size_vectors:   List[int] = [16]


# 8b-A, 8b-W Mode Configurations:
AL2_choices: List[size.Size] = [size.B(46080)]
AL1_choices: List[size.Size] = [size.B(8192)]
WL1_choices: List[size.Size] = [size.B(1168)]
# TODO: MUSE-V3 can support more basic sizes
OL1_choices_map: Dict[size.Size, TileSize] = {
    size.B(3): TileSize(1, 1),
    size.B(12): TileSize(2, 2),
    size.B(48): TileSize(4, 4),
    size.B(192): TileSize(8, 8)
}

# 16b-A, 8b-W Mode Configurations:
# AL2_choices: List[size.Size] = [size.B(23040)]
# AL1_choices: List[size.Size] = [size.B(4096)]
# WL1_choices: List[size.Size] = [size.B(1168)]
# TODO: MUSE-V3 can support more basic sizes
# OL1_choices_map: Dict[size.Size, TileSize] = {
#     size.B(3): TileSize(1, 1),
#     size.B(12): TileSize(2, 2),
#     size.B(48): TileSize(4, 4),
#     size.B(192): TileSize(8, 8)
# }


# 8b-A, 4b-W Mode Configurations:
# AL2_choices: List[size.Size] = [size.B(46080)]
# AL1_choices: List[size.Size] = [size.B(8192)]
# WL1_choices: List[size.Size] = [size.B(2336)]
# TODO: MUSE-V3 can support more basic sizes
# OL1_choices_map: Dict[size.Size, TileSize] = {
#     size.B(3): TileSize(1, 1),
#     size.B(12): TileSize(2, 2),
#     size.B(48): TileSize(4, 4),
#     size.B(192): TileSize(8, 8)
# }


# 4b-A, 4b-W Mode Configurations:
# AL2_choices: List[size.Size] = [size.B(92160)]
# AL1_choices: List[size.Size] = [size.B(16384)]
# WL1_choices: List[size.Size] = [size.B(2336)]
# TODO: MUSE-V3 can support more basic sizes
# OL1_choices_map: Dict[size.Size, TileSize] = {
#     size.B(3): TileSize(1, 1),
#     size.B(12): TileSize(2, 2),
#     size.B(48): TileSize(4, 4),
#     size.B(192): TileSize(8, 8)
# }


# To create the package-level spatial division
# In the present version, support Hp-only (P-Type) and Kp-only (C-Type) division
# (i.e., Hp = 1, Kp = 4 or Kp = 1, Hp = 4)
def get_package_spatial_parameters() -> DefaultDict[int, List[PSpatialParameter]]:
    parameter_dict: DefaultDict[int, List[PSpatialParameter]] = defaultdict(list)
    for chiplet in num_chiplets:
        for Kp in {1, chiplet}:
            for Hp in {1, chiplet}:
                if Kp * Hp == chiplet:
                    parameter_dict[chiplet].append(PSpatialParameter(Kp, Hp))
    return parameter_dict


# To create the chiplet-level spatial division
# It supports P-Type, C-Type, and H-Type division
# But in the plane-dimension division, the W-H ratio is under constrain 
def get_chiplet_spatial_parameters() -> DefaultDict[int, List[CSpatialParameter]]:
    parameter_dict: DefaultDict[int, List[CSpatialParameter]] = defaultdict(list)
    for core in num_cores:
        for Kc in range(1, core + 1):
            for WcHc, v in core_choices_map.items():
                if Kc * WcHc == core:
                    parameter_dict[core].append(CSpatialParameter(Kc, v.H, v.W))
    return parameter_dict



# 16nm energy comsumption in different memory hierarchies; pJ/bit
# For we need to re-evaluate the overhead under the present GF-12 HPC+
# Energy_DRAM = 8.75
Energy_DRAM = 2.9296875
# Energy_GRS = 1.285
# Please ignore the name. In our system, we use Innolink as the D2D interconnection
Energy_GRS = 2.000
Energy_MAC = 0.003

def get_Energy_RF(RF_size: size.Size):
    Energy_RF_coe = 1.1285e-05
    Energy_RF = Energy_RF_coe * RF_size.to_b()
    return Energy_RF

# def get_Energy_SRAM_Wr(SRAM_size: size.Size):
#     p1 = 2.197265625e-07
#     p2 = 0.3305
#     Energy_SRAM = p1 * (SRAM_size.to_b()) + p2
#     return Energy_SRAM


# def get_Energy_SRAM_Rd(SRAM_size: size.Size):
#     p1 = 1.953125e-07
#     p2 = 0.2984
#     Energy_SRAM = p1 * (SRAM_size.to_b()) + p2
#     return Energy_SRAM

Energy_AL1_Wr = 1.48807E-05
Energy_AL1_Rd = 1.75158E-05
Energy_WL1_Wr = 2.12696E-05
Energy_WL1_Rd = 2.24287E-05
Energy_AL2_Wr = 1.53444e-05
Energy_AL2_Rd = 1.79165e-05
Energy_OL2_Wr = 3.18749E-05
Energy_OL2_Rd = 2.42544E-05

# For MUSE-v3 scheduling, we do not care the area overhead
# The following configurations are from UCM-28nm, please ignore

def get_area_SRAM(SRAM_size: size.Size): #um2
    p1 = 0.062928466796875
    p2 = 5289
    area_sram = p1 * (SRAM_size.to_b()) + p2
    return area_sram

def get_area_RF(RF_size: size.Size):
    p1 = 1.0760498046875e-04
    p2 = 281.91
    area_RF = p1 * (RF_size.to_b()) + p2
    return area_RF

def get_area_PE(vector_size: int, lane: int):
    p1 = 135.1
    area_PE = lane * vector_size * p1
    return area_PE