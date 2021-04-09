"""Define the various data and scheduling package"""

from dataclasses import dataclass
from enum import Enum, auto
from math import ceil
from typing import List

import size

# We have 7-dimensional Nested Loop
# TODO: Batch-dimension
class ForLoopSymbol(Enum):
    W1 = auto()
    H1 = auto()
    K1 = auto()
    W2 = auto()
    H2 = auto()
    K2 = auto()
    Csa = auto()


# Define the related loop for Weight and Activation
WImpactFactors  = {ForLoopSymbol.K1, ForLoopSymbol.K2}
IAImpactFactors = {ForLoopSymbol.W1, ForLoopSymbol.H1, ForLoopSymbol.W2, ForLoopSymbol.H2, ForLoopSymbol.Csa}


# For all size elements and stride are in Block unit
@dataclass(frozen=True)
class Block:
    H: int
    W: int

    def size(self) -> int:
        return self.W * self.H
    
    def get_params(self):
        return self.W, self.H


# Redefine the Block class as TileSize, to be specifically used in the "Tile" description (for output). 
# in_tile is used to infer the input size with the given out_tile size.
class TileSize(Block):
    def in_tile(self, kernel_size: Block, stride: Block):
        return TileSize((self.H - 1) * stride.H + kernel_size.H, (self.W - 1) * stride.W + kernel_size.W)


# Package-level Spatial Parameters:
@dataclass(frozen=True)
class PSpatialParameter:
    Kp: int
    Hp: int


# Chiplet-level Spatial Parameters:
@dataclass(frozen=True)
class CSpatialParameter:
    Kc: int
    Hc: int
    Wc: int


# To describe the workload
@dataclass(frozen=True)
class Workload:
    in_channel:     int
    out_channel:    int
    out_size:       Block
    kernel_size:    Block
    stride:         Block


# To describe the AI system design
# For our compilation flow, there is only one design case
@dataclass(frozen=True)
class Design:
    chiplet:    int
    core:       int
    lane:       int
    vector:     int


# To package all loop parameters
class LoopParameter:
    def __init__(self, W1: int, H1: int, K1: int, W2: int, H2: int, K2: int, \
        package_spatial_parameter: PSpatialParameter, \
        chiplet_spatial_parameter: CSpatialParameter, \
        workload: Workload, design: Design, rotation_enable=True):
        # Spatial Division Parameters:
        self.package_spatial_parameter = package_spatial_parameter
        self.chiplet_spatial_parameter = chiplet_spatial_parameter
        # Rotation Parameters:
        if rotation_enable:
            self.Csa = self.package_spatial_parameter.Kp
        else:
            self.Csa = 1
        # Temporal Parameters:
        self.C0 = design.vector
        self.C1 = ceil(workload.in_channel / (self.C0 * self.Csa))
        self.W1 = W1
        self.H1 = H1
        self.K1 = K1
        self.W2 = W2
        self.H2 = H2
        self.K2 = K2
        # Rotation mode
        self.rotation_enable = rotation_enable
    
    def symbol_to_count(self, forLoopSymbol: ForLoopSymbol) -> int:
        if forLoopSymbol == ForLoopSymbol.W1:
            return self.W1
        elif forLoopSymbol == ForLoopSymbol.H1:
            return self.H1
        elif forLoopSymbol == ForLoopSymbol.K1:
            return self.K1
        elif forLoopSymbol == ForLoopSymbol.W2:
            return self.W2
        elif forLoopSymbol == ForLoopSymbol.H2:
            return self.H2
        elif forLoopSymbol == ForLoopSymbol.K2:
            return self.K2
        else:
            raise ValueError('Only support getting X1/Y1/K1/X2/Y2/K2 from LoopParameter')

    def get_temporal_count(self):
        return self.C0, self.C1, self.W1, self.W2, self.H1, self.H2, self.K1, self.K2
    
    def get_rotation_count(self):
        return self.Csa
    
    def get_spatial_count(self):
        return self.package_spatial_parameter.Kp, self.package_spatial_parameter.Hp, \
            self.chiplet_spatial_parameter.Kc, self.chiplet_spatial_parameter.Wc, self.chiplet_spatial_parameter.Hc


# To define two loop reorder cases
@dataclass(frozen=True)
class ReorderCase:
    type_n: int

    def __post_init__(self):
        assert self.type_n == 1 or self.type_n == 2, 'type_n should be 1 or 2'

    def getreorder(self) -> List[ForLoopSymbol]:
        if self.type_n == 1:
            # Channel (K2) - First:
            return [ForLoopSymbol.H2, ForLoopSymbol.W2, ForLoopSymbol.K2, ForLoopSymbol.K1, ForLoopSymbol.H1, ForLoopSymbol.W1]
        elif self.type_n == 2:
            # Plane (H2, W2) - First:
            return [ForLoopSymbol.K2, ForLoopSymbol.H2, ForLoopSymbol.W2, ForLoopSymbol.K1, ForLoopSymbol.H1, ForLoopSymbol.W1]
        else:
            raise ValueError('type_n should be 1 or 2')


# To package the analysis case, including all SRAM, loop-reorder, loop-parameter (tiling), and workload 
@dataclass(frozen=True)
class AnalysisCase:
    OL1: size.Size
    AL1: size.Size
    WL1: size.Size
    AL2: size.Size
    reorderCase: ReorderCase
    loopParameter: LoopParameter
    workload: Workload


# To define all the memory access costs
@dataclass(frozen=True)
class MemoryAccess:
    WL1_Wr: size.Size
    WL1_Rd: size.Size
    
    OL1_Wr: size.Size
    OL1_Rd: size.Size
    
    AL1_Wr: size.Size
    AL1_Rd: size.Size
    
    AL2_Wr: size.Size
    AL2_Rd: size.Size
    
    OL2_Wr: size.Size
    OL2_Rd: size.Size