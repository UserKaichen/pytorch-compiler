''' To Calculate memory access (read & write) based on the penalty obtained from C3P '''

from C3Pmethodology import C3PMethodology
from meta_model import AnalysisCase, Design, MemoryAccess, TileSize
import config as cfg
import size

class SramAccess:
    def __init__(self, memcase: AnalysisCase, computecase: Design):
        self.memcase = memcase
        self.computecase = computecase
        self.cp = C3PMethodology(self.memcase)
        
    # c3p execution
    def c3p_analysis(self):
        return self.cp.get_c3p_info()
    
    # To decode the c3p_info
    def get_penalty(self, capacity, c3p_info, check):
        penalty = 1
        if (capacity < c3p_info['Critical_Capacity'][0]) & check == 1:
            raise ValueError('The capacity should larger than Critical-Capacity-0')
        if c3p_info['Critical_Capacity'][3] and capacity < c3p_info['Critical_Capacity'][3]:
            penalty = penalty * c3p_info['Penalty'][3]
        if c3p_info['Critical_Capacity'][2] and capacity < c3p_info['Critical_Capacity'][2]:
            penalty = penalty * c3p_info['Penalty'][2]
        if c3p_info['Critical_Capacity'][1] and capacity < c3p_info['Critical_Capacity'][1]:
            penalty = penalty * c3p_info['Penalty'][1]
        
        return penalty

    # Get the sigle SRAM access
    def get_sram_access(self) -> MemoryAccess:
        Fw, Fh = self.memcase.workload.kernel_size.get_params()
        Csa = self.memcase.loopParameter.get_rotation_count() 
        W0, H0 = cfg.OL1_choices_map[self.memcase.OL1].get_params()
        C0, C1, W1, W2, H1, H2, K1, K2 = self.memcase.loopParameter.get_temporal_count()
        _, _, _, Wc, Hc = self.memcase.loopParameter.get_spatial_count()
        

        c3p_info, Cc_fake = self.c3p_analysis()

        # Memory Write for one WL1
        WL1_Wr = size.B(Fw * Fh * C0 * C1 * Csa * K1 * K2)                          # Min access count
        WL1_penalty = self.get_penalty(self.memcase.WL1, c3p_info['WL1'], 1)        # Get the penalty item
        WL1_Wr = WL1_Wr * WL1_penalty                                               # The real access count
        WL1_Rd = size.B(Fw * Fh * K1 * K2 * C0 * C1 * Csa * W1 * H1 * W2 * H2)      # Weight-stationary
        

        # OL1 & OL2;  the '4' in OL1 means that the bit-width of partial sums is 32bit 
        OL1_Rd = size.B(4 * W0 * H0 * Fw * Fh * C1 * Csa * W1 * H1 * W2 * H2 * K2 * K1 - 1) \
            + size.B(4 * W0 * H0 * W1 * H1 * W2 * H2 * K2 * K1)                         # MAC Read + Read to OL2.
        OL1_Wr = size.B(4 * W0 * H0 * Fw * Fh * C1 * Csa * W1 * H1 * W2 * H2 * K2 * K1) # MAC update
        OL2_Wr = size.B(W0 * H0 * W1 * H1 * W2 * H2 * K1 * K2 * Wc * Hc)            # Read from OL1 and write to OL2
        OL2_Rd = size.B(W0 * H0 * W1 * H1 * W2 * H2 * K1 * K2 * Wc * Hc)            # Read from OL2 and write to DDR


        # AL1 (just like a pipeline register, reuse for a basic tile of H0 * W0)
        if self.memcase.AL1 < c3p_info['AL1']['Critical_Capacity'][0]:
            raise ValueError('The capacity should larger than Critical-Capacity-0')             # See the Error description
        basic_tile_in = TileSize(W0, H0).in_tile(self.memcase.workload.kernel_size, self.memcase.workload.stride)
        H0_in = basic_tile_in.H
        W0_in = basic_tile_in.W

        AL1_Wr = size.B(W0_in * H0_in * c3p_info['AL1']['Penalty'][1] * C0 * C1 * Csa)          # all the temporal loop counts are penalty
        AL1_Rd = size.B(C0 * Csa * C1 * K1 * K2 * W0_in * H0_in * Fw * Fh * W1 * W2 * H1 * H2)


        # AL2
        W0W1Wc_H0H1Hc_in_tile_size = cfg.TileSize(H0 * H1 * Hc, W0 * W1 * Wc).in_tile(self.memcase.workload.kernel_size, self.memcase.workload.stride).size()
        W0W1W2Wc_H0H1H2Hc_in_tile_size = cfg.TileSize(H0 * H1 * H2 * Hc, W0 * W1 * W2 * Wc).in_tile(self.memcase.workload.kernel_size, self.memcase.workload.stride).size()

        AL2_Wr = W0W1W2Wc_H0H1H2Hc_in_tile_size

        if self.memcase.AL2 < Cc_fake['AL2'][2]:                # There are some cases that AL2 can buffer the whole chiplet workload
            AL2_Wr = W0W1Wc_H0H1Hc_in_tile_size * W2 * H2       # If cannot buffer, each tile size is "W0W1Wc_H0H1Hc_in_tile_size"

        AL2_penalty = self.get_penalty(self.memcase.AL2, c3p_info['AL2'], 0)
        AL2_Wr = size.B(AL2_Wr * AL2_penalty * C0 * C1)

        AL2_Rd = AL1_Wr * Hc * Wc                               # Broadcast to Kc cores (input reuse)

        return MemoryAccess(WL1_Wr, WL1_Rd, OL1_Wr, OL1_Rd, AL1_Wr, AL1_Rd, AL2_Wr, AL2_Rd, OL2_Wr, OL2_Rd)

    def get_total_memory_access(self) -> MemoryAccess:
        singleMemAccess = self.get_sram_access()
        lane = self.computecase.lane
        core = self.computecase.core
        chiplet = self.computecase.chiplet
        Kc = self.memcase.loopParameter.chiplet_spatial_parameter.Kc

        T_WL1_Wr = singleMemAccess.WL1_Wr * lane * chiplet * Kc         # In the architectual view, only Kc cores receive data, shared by other Hc*Wc cores
        T_WL1_Rd = singleMemAccess.WL1_Rd * lane * chiplet * core       # MUSE-V3 does not support the multi-core boardcasting, so multiply by 'core'

        T_OL1_Wr = singleMemAccess.OL1_Wr * lane * core * chiplet       # 'lane * core * chiplet' OL1 in total
        T_OL1_Rd = singleMemAccess.OL1_Rd * lane * core * chiplet       # 'lane * core * chiplet' OL1 in total

        T_OL2_Wr = singleMemAccess.OL2_Wr * chiplet                     # 'chiplet' OL2
        T_OL2_Rd = singleMemAccess.OL2_Rd * chiplet                     # 'chiplet' OL2
        
        T_AL1_Wr = singleMemAccess.AL1_Wr * core * chiplet              # 'core * chiplet' AL1
        T_AL1_Rd = singleMemAccess.AL1_Rd * core * chiplet              # 'core * chiplet' AL1

        T_AL2_Wr = singleMemAccess.AL2_Wr * chiplet                     # 'chiplet' AL2
        T_AL2_Rd = singleMemAccess.AL2_Rd * chiplet                     # 'chiplet' AL2


        return MemoryAccess(T_WL1_Wr, T_WL1_Rd, T_OL1_Wr, T_OL1_Rd, T_AL1_Wr, T_AL1_Rd, T_AL2_Wr, T_AL2_Rd, T_OL2_Wr, T_OL2_Rd)