"""Define the functions for generate design cases (pre-design flow) and analysis cases (used in both flows)"""

import csv
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from math import ceil
from typing import List, Dict, DefaultDict
from collections import defaultdict

import size
from meta_model import Design, Workload, LoopParameter, ReorderCase, Block, TileSize, AnalysisCase
from config import num_chiplets, num_cores, num_lanes, size_vectors
from config import get_chiplet_spatial_parameters, get_package_spatial_parameters
from config import AL1_choices, WL1_choices, OL1_choices_map, AL2_choices, W1H1_choices_map
from config import ACT_MEMORY_ALIGN, rotation_search_list
from sram_access import SramAccess
from overhead_eval import OverheadEval


# Create the design space
# For MUSE-V3, there is only one case in the design space
def get_design_dict() -> DefaultDict[int, List[Design]]:
    """
    Returns: design_dict: DefaultDict[int, List[Design]] = { 64: [Design(chiplet=1, core=1, lane=8, vector_size=8)],
    128: [Design(chiplet=1, core=1, lane=8, vector_size=16), Design(chiplet=1, core=1, lane=16, vector_size=8),
    Design(chiplet=1, core=2, lane=8, vector_size=8), Design(chiplet=2, core=1, lane=8, vector_size=8)],
    256: [Design(chiplet=1, core=1, lane=16, vector_size=16), Design(chiplet=1, core=2, lane=8, vector_size=16),
    Design(chiplet=1, core=2, lane=16, vector_size=8), Design(chiplet=1, core=4, lane=8, vector_size=8),
    Design(chiplet=2, core=1, lane=8, vector_size=16), Design(chiplet=2, core=1, lane=16, vector_size=8),
    Design(chiplet=2, core=2, lane=8, vector_size=8), Design(chiplet=4, core=1, lane=8, vector_size=8)], 512: ...,
    1024: ..., 2048: ..., 4096: ..., 8192: ..., 16384: ... }
    """

    design_dict: DefaultDict[int, List[Design]] = defaultdict(list)

    for chiplet in num_chiplets:
        for core in num_cores:
            for lane in num_lanes:
                for vector_size in size_vectors:
                    mac = chiplet * core * lane * vector_size
                    design = Design(chiplet, core, lane, vector_size)
                    design_dict[mac].append(design)

    return design_dict


# To select the designs that meet the performance requirement
# For MUSE-V3, this function can be ignored
def pick_possible_designs(design_dict: DefaultDict[int, List[Design]], req_mac: int) -> List[Design]:
    result: List[Design] = []

    for mac_, designs in design_dict.items():
        if mac_ >= req_mac and mac_ < 2 * req_mac:
            result.extend(designs)

    return result


# To generate all analysis cases
def get_analysis_cases(design: Design, workload: Workload) -> List[AnalysisCase]:
    result: List[AnalysisCase] = []

    # To pick the corresponding package-spatial division cases
    package_spatial_parameter_map = get_package_spatial_parameters()
    package_spatial_parameters = package_spatial_parameter_map[design.chiplet]
    K0 = design.lane

    for package_spatial_parameter in package_spatial_parameters:
        Kp = package_spatial_parameter.Kp
        Hp = package_spatial_parameter.Hp
        Fw = workload.kernel_size.W
        Fh = workload.kernel_size.H
        C0 = design.vector

        chiplet_workload_W: int = ceil(workload.out_size.W  / 1 )
        chiplet_workload_H: int = ceil(workload.out_size.H  / Hp)
        chiplet_workload_K: int = ceil(workload.out_channel / Kp)

        for WL1 in WL1_choices:                             # This loop is no meaning is the post-design flow (i.e., for MUSE-V3)
            # Chiplet-level spatial division: Kc, Hc, Wc
            chiplet_spatial_parameter_map = get_chiplet_spatial_parameters()
            chiplet_spatial_parameters = chiplet_spatial_parameter_map[design.core]

            # Temp WL1 to avoid overwrite the original value in the following iteration
            WL1_temp = WL1

            for chiplet_spatial_parameter in chiplet_spatial_parameters:    # To generate different packae-level spatial division

                Kc = chiplet_spatial_parameter.Kc
                Wc = chiplet_spatial_parameter.Wc
                Hc = chiplet_spatial_parameter.Hc

                # In MUSE-V3, if weight data can be shared by multi-cores, their local WL1 can be fused to 
                # form a larger buffer. Wc * Hc refers to the number of shared cores
                WL1 = WL1_temp * Wc * Hc

                for OL1, BasicSize in OL1_choices_map.items():              # To generate different basic output-tile size
                    OL1: size.Size
                    for AL2 in AL2_choices:                                 # This loop is no meaning is the post-design flow (i.e., for MUSE-V3)
                        AL2: size.Size

                        chiplet_workload_in = TileSize(chiplet_workload_W, chiplet_workload_H).in_tile(workload.kernel_size, workload.stride)
                        # Adapt multiple basic-tile (mini-tile) for a core (plane dimension)
                        n = 1
                        # TODO: To support more cases for the total number of mini-tiles (HW can do it)
                        for i in [2, 4, 8, 16]:                             # To find the different Level-1 temporal cases (for H & W)
                            W1H1_choice = W1H1_choices_map[i]
                            sub_tile_in = TileSize(BasicSize.H * W1H1_choice.H, BasicSize.W * W1H1_choice.W).in_tile(workload.kernel_size, workload.stride)
                            tile_in = TileSize(Hc * sub_tile_in.H, Wc * sub_tile_in.W).in_tile(workload.kernel_size, workload.stride)

                            # To check whether the tile size is larger than the chiplet workload
                            if tile_in.W > chiplet_workload_in.W or tile_in.H > chiplet_workload_in.H:
                                break
                            # The tile_in needs to be fit in the AL2
                            # Align by the vector_size
                            if AL2 >= size.B(tile_in.W * tile_in.H * ceil(workload.in_channel / C0) * C0):
                                n = i
                            else:
                                break
                        W1H1_choice = W1H1_choices_map[n]
                        W1 = W1H1_choice.W
                        H1 = W1H1_choice.H

                        # Adapt multiple basic-tile (mini-tile) for a core (channel dimension)
                        K1 = 1 # Initialization
                        # It can be larger
                        # TODO: j can be any integer, e.g., j=3. But it needs to handle the margin case.
                        for j in [1, 2, 4, 8, 16]:                          # To find the different Level-1 temporal cases (for K)
                            # To check whether the tile channel is larger than the chiplet workload
                            if Kp * Kc * K0 * j >= chiplet_workload_K:
                                break
                            
                            # The tile_in needs to be fit in the WL1
                            # Align by the vector_size
                            if WL1 >= size.B(Fw * Fh * ceil(workload.in_channel / C0) * C0 * j):
                                K1 = j
                            else:
                                break
                        
                        # Generate Loop Parameters and Analysis Cases
                        tile_W = BasicSize.W * W1 * Wc
                        tile_H = BasicSize.H * H1 * Hc
                        tile_K = K0 * K1 * Kc

                        W2: int = ceil(chiplet_workload_W / tile_W)
                        H2: int = ceil(chiplet_workload_H / tile_H)
                        K2: int = ceil(chiplet_workload_K / tile_K)

                        for rotation_enable in rotation_search_list:                # To generate rotation or non-rotation cases
                            
                            if rotation_enable:
                                if workload.in_channel % (ACT_MEMORY_ALIGN * design.chiplet) != 0:
                                    continue
                                else:
                                    aligned_CI = ceil(workload.in_channel / (C0 * Kp)) * C0 * Kp
                            else:
                                aligned_CI = ceil(workload.in_channel / C0) * C0 

                            aligned_workload = Workload(aligned_CI, tile_K * K2 * Kp,
                                                    Block(tile_H * H2, tile_W * W2),
                                                    workload.kernel_size, workload.stride)

                            loopParameter = LoopParameter(W1, H1, K1, W2, H2, K2, package_spatial_parameter, \
                                chiplet_spatial_parameter, aligned_workload, design, rotation_enable)

                            for AL1 in AL1_choices:                         # This loop is no meaning is the post-design flow (i.e., for MUSE-V3)
                                if AL1 > AL2:
                                    continue
                                for reorderCase in [ReorderCase(type_n) for type_n in [1, 2]]:  # To generate different reordering cases
                                    result.append(AnalysisCase(OL1, AL1, WL1, AL2, reorderCase, loopParameter, aligned_workload))

    return result