''' To analyse the access penalty '''

from typing import List, Optional
from dataclasses import dataclass

import config as cfg
import size
from config import OL1_choices_map, TileSize
from meta_model import IAImpactFactors, WImpactFactors, ForLoopSymbol, AnalysisCase

class C3PMethodology:
    def __init__(self, case: AnalysisCase):
        self.case = case

    def get_c3p_info(self):
        workloadForLoopDescription = self.case.reorderCase.getreorder()

        basic_workload = OL1_choices_map[self.case.OL1]
        W0, H0 = basic_workload.get_params()
        W0_in, H0_in = basic_workload.in_tile(self.case.workload.kernel_size, self.case.workload.stride).get_params()

        C0, C1, W1, W2, H1, H2, K1, K2 = self.case.loopParameter.get_temporal_count()
        _, _, _, Wc, Hc = self.case.loopParameter.get_spatial_count()
        Csa = self.case.loopParameter.get_rotation_count() 

        self.c3p_info = {}
        self.Cc_fake = {}

        ######################################################## WL1 ########################################################
        Fx, Fy = self.case.workload.kernel_size.get_params()
        # The critical point is None if it doesn't make sense
        '''
        WL1 corresponds to one lane.
        Cc0: Once basic workload (for most workload, our MUSE-v3 can satisfy it)
        Cc1: Buffer all input channels (some extreme cases maybe fail to satisfy it and then should consider ping-pong in compiler)
        Cc2: Buffer all mini-tiles with all input channels. 
        In fact, in the generate_case.py, if WL1 >= Cc1, and then can satisfy the Cc2
        Cc3: Buffer all weights
        '''
        self.WL1_Cc0: size.Size = size.B(Fx * Fy * C0)  # At least to support once mapping of C0 kernels for a lane
        self.WL1_Cc1: Optional[size.Size] = None
        self.WL1_Cc2: Optional[size.Size] = None
        self.WL1_Cc3: Optional[size.Size] = None
        
        self.WL1_Cc0_Penalty: int = 1
        self.WL1_Cc1_Penalty: int = 1
        self.WL1_Cc2_Penalty: int = 1
        self.WL1_Cc3_Penalty: int = 1

        WL1_Cc2_Penalty_start = 0
        WL1_Cc3_Penalty_start = 0

        WL1_info = {}

        # WL1: Cc1
        self.WL1_Cc1 = self.WL1_Cc0 * C1 * Csa

        # Calculate WL1: Cc1_Penalty
        for forLoopSymbol in reversed(workloadForLoopDescription):
            if forLoopSymbol in WImpactFactors:
                break
            else:
                self.WL1_Cc1_Penalty = self.WL1_Cc1_Penalty * self.case.loopParameter.symbol_to_count(forLoopSymbol)
        
        # Check WL1: Cc2
        K1Index = workloadForLoopDescription.index(ForLoopSymbol.K1)
        if K1Index != 0:
            for forLoopSymbol in reversed(workloadForLoopDescription[:K1Index]):
                if self.case.loopParameter.symbol_to_count(forLoopSymbol) != 1:
                    # Find the first real loop
                    WL1_Cc2_Penalty_start = workloadForLoopDescription.index(forLoopSymbol) + 1
                    if forLoopSymbol in WImpactFactors:
                        # Cc2 doesn't make sense
                        pass
                    else:
                        self.WL1_Cc2 = self.WL1_Cc0 * C1 * Csa * K1
                    break
        
        # Calculate WL1: Cc2_Penalty
        WL1_Cc2_Penalty_start = workloadForLoopDescription.index(forLoopSymbol.K1)
        if WL1_Cc2_Penalty_start > 0:
            for forLoopSymbol in reversed(workloadForLoopDescription[:WL1_Cc2_Penalty_start]):
                if forLoopSymbol in WImpactFactors:
                    break
                else:
                    self.WL1_Cc2_Penalty = self.WL1_Cc2_Penalty * self.case.loopParameter.symbol_to_count(forLoopSymbol)
                        
        # Check WL1: Cc3
        K2Index = workloadForLoopDescription.index(ForLoopSymbol.K2)
        if K2Index != 0:
            for forLoopSymbol in reversed(workloadForLoopDescription[:K2Index]):
                if self.case.loopParameter.symbol_to_count(forLoopSymbol) != 1:
                    # Find the first real loop
                    WL1_Cc3_Penalty_start = workloadForLoopDescription.index(forLoopSymbol) + 1
                    if forLoopSymbol in WImpactFactors:
                        # Cc3 doesn't make sense
                        pass
                    else:
                        self.WL1_Cc3 = self.WL1_Cc0 * C1 * Csa * K1 * K2
                    break
        
        # Calculate WL1: Cc3_Penalty
        WL1_Cc3_Penalty_start = workloadForLoopDescription.index(forLoopSymbol.K2)
        if WL1_Cc3_Penalty_start > 0:
            for forLoopSymbol in reversed(workloadForLoopDescription[:WL1_Cc3_Penalty_start]):
                if forLoopSymbol in WImpactFactors:
                    break
                else:
                    self.WL1_Cc3_Penalty = self.WL1_Cc3_Penalty * self.case.loopParameter.symbol_to_count(forLoopSymbol)

        WL1_info['Critical_Capacity'] = [self.WL1_Cc0, self.WL1_Cc1, self.WL1_Cc2, self.WL1_Cc3]
        WL1_info['Penalty'] = [self.WL1_Cc0_Penalty, self.WL1_Cc1_Penalty, self.WL1_Cc2_Penalty, self.WL1_Cc3_Penalty]
        self.c3p_info['WL1'] = WL1_info
        #####################################################################################################################
        
        ######################################################## AL1 ########################################################
        # if (self.case.loopParameter.K1 == 1 and self.case.loopParameter.H1 == 1 and self.case.loopParameter.W1 == 1 \
        #     and self.case.loopParameter.K2 == 2 and self.case.loopParameter.H2 == 8 and self.case.loopParameter.W2 == 8 \
        #     and self.case.workload.stride.H == 2 and self.case.loopParameter.C1 == 16 and self.case.workload.kernel_size.H == 1):
        #     print(1)  # debug
        '''
        In our design, the AL1 is just like a vector-register (very small) in the pipeline.
        Therefore, the Penalty is the number of Basic-Tiles (HO * WO * CO) 
        '''
        self.AL1_Cc0: size.Size = size.B(W0_in * H0_in * C0)
        self.AL1_Cc0_Penalty: int = 1
        self.AL1_Penalty: int = 1

        AL1_info = {}   

        for forLoopSymbol in reversed(workloadForLoopDescription):      # All the outer loops are penalty
            self.AL1_Penalty = self.AL1_Penalty * self.case.loopParameter.symbol_to_count(forLoopSymbol)
        
        AL1_info['Critical_Capacity'] = [self.AL1_Cc0]
        AL1_info['Penalty'] = [self.AL1_Cc0_Penalty, self.AL1_Penalty]
        self.c3p_info['AL1'] = AL1_info

        #####################################################################################################################

        ######################################################## AL2 ########################################################
        tileSize = TileSize(H0 * Hc, W0 * Wc)
        tileSize_in = tileSize.in_tile(self.case.workload.kernel_size, self.case.workload.stride)
        W_Cc2 = W0 * W1 * Wc
        H_Cc2 = H0 * H1 * Hc
        W_Cc3 = W0 * W1 * Wc * W2
        H_Cc3 = H0 * H1 * Hc * H2
        
        '''
        Cc0: At least once mapping for all cores
        Cc1: Can buffer all input channels
        Cc2: Can buffer all mini-tile
        Cc3: Can buffer all tiles  (e.g., some layers have very the small feature map size)

        But in fact, generate_case.py has guaranteed AL2 >= Cc2.
        The following functions prevent some exception cases that I have ignored.
        '''

        self.AL2_Cc0: size.Size = size.B(tileSize_in.size() * C0)
        self.AL2_Cc1_fake: size.Size = self.AL2_Cc0 * C1
        self.AL2_Cc2_fake = size.B(TileSize(H_Cc2, W_Cc2).in_tile(self.case.workload.kernel_size, self.case.workload.stride).size() * C0 * C1)
        self.AL2_Cc3_fake = size.B(TileSize(H_Cc3, W_Cc3).in_tile(self.case.workload.kernel_size, self.case.workload.stride).size() * C0 * C1)
        
        


        # The critical point is None if it doesn't make sense
        self.AL2_Cc1: Optional[size.Size] = None
        self.AL2_Cc2: Optional[size.Size] = None
        self.AL2_Cc3: Optional[size.Size] = None

        self.AL2_Cc0_Penalty: int = 1
        self.AL2_Cc1_Penalty: int = 1
        self.AL2_Cc2_Penalty: int = 1
        self.AL2_Cc3_Penalty: int = 1

        AL2_Cc2_Penalty_start = 0
        AL2_Cc3_Penalty_start = 0

        AL2_info = {}
        
        # Check AL2: Cc1 (Check whether critical points are meaningful)
        for forLoopSymbol in reversed(workloadForLoopDescription):
            if self.case.loopParameter.symbol_to_count(forLoopSymbol) != 1:
                # Find the first real loop
                if forLoopSymbol in IAImpactFactors:
                    # Cc1 doesn't make sense
                    pass
                else:
                    self.AL2_Cc1 = self.AL2_Cc1_fake
                break
        
        for forLoopSymbol in reversed(workloadForLoopDescription):
            if forLoopSymbol in IAImpactFactors:
                break
            else:
                self.AL2_Cc1_Penalty = self.AL2_Cc1_Penalty * self.case.loopParameter.symbol_to_count(forLoopSymbol)

        # Check AL2: Cc2
        H1Index = workloadForLoopDescription.index(ForLoopSymbol.H1)
        if H1Index != 0:
            for forLoopSymbol in reversed(workloadForLoopDescription[:H1Index]):
                if self.case.loopParameter.symbol_to_count(forLoopSymbol) != 1:
                    # Find the first real loop
                    if forLoopSymbol in IAImpactFactors:
                        # Cc2 doesn't make sense
                        pass
                    else:
                        self.AL2_Cc2 = self.AL2_Cc2_fake
                    break
        
        AL2_Cc2_Penalty_start = workloadForLoopDescription.index(forLoopSymbol.H1)
        if AL2_Cc2_Penalty_start > 0:
            for forLoopSymbol in reversed(workloadForLoopDescription[:AL2_Cc2_Penalty_start]):
                if forLoopSymbol in IAImpactFactors:
                    break
                else:
                    self.AL2_Cc2_Penalty = self.AL2_Cc2_Penalty * self.case.loopParameter.symbol_to_count(forLoopSymbol)

        # Check AL2: T3
        H2Index = workloadForLoopDescription.index(ForLoopSymbol.H2)
        if H2Index != 0:
            for forLoopSymbol in reversed(workloadForLoopDescription[:H2Index]):
                if self.case.loopParameter.symbol_to_count(forLoopSymbol) != 1:
                    # Find the first real loop
                    if forLoopSymbol in IAImpactFactors:
                        # T3_n doesn't make sense
                        pass
                    else:
                        self.AL2_Cc3 = self.AL2_Cc3_fake
                    break

        AL2_Cc3_Penalty_start = workloadForLoopDescription.index(forLoopSymbol.H2)
        if AL2_Cc3_Penalty_start > 0:
            for forLoopSymbol in reversed(workloadForLoopDescription[:AL2_Cc3_Penalty_start]):
                if forLoopSymbol in IAImpactFactors:
                    break
                else:
                    self.AL2_Cc3_Penalty = self.AL2_Cc3_Penalty * self.case.loopParameter.symbol_to_count(forLoopSymbol)

        AL2_info['Critical_Capacity'] = [self.AL2_Cc0, self.AL2_Cc1, self.AL2_Cc2, self.AL2_Cc3]
        AL2_info['Penalty'] = [self.AL2_Cc0_Penalty, self.AL2_Cc1_Penalty, self.AL2_Cc2_Penalty, self.AL2_Cc3_Penalty]
        self.c3p_info['AL2'] = AL2_info
        self.Cc_fake['AL2'] = [self.AL2_Cc1_fake, self.AL2_Cc2_fake, self.AL2_Cc3_fake]
        #####################################################################################################################
        
        return self.c3p_info, self.Cc_fake
