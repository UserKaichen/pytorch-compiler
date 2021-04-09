''' To calculate the overhead, including area and energy '''

from dataclasses import dataclass

from C3Pmethodology import C3PMethodology
from meta_model import AnalysisCase, Design, MemoryAccess
from sram_access import SramAccess
import config as cfg
import size

class OverheadEval(SramAccess):
    def __init__(self, analysis_case: AnalysisCase, arch_case: Design):
        super(OverheadEval, self).__init__(analysis_case, arch_case)
        self.total_memory_access = self.get_total_memory_access()
        
        self.OL2_size: size.Size = size.B((cfg.OL1_choices_map[self.memcase.OL1].W \
            * cfg.OL1_choices_map[self.memcase.OL1].H) * self.computecase.lane * self.computecase.core * self.computecase.chiplet) 
        
        self.real_WL1 = size.B(self.memcase.WL1.to_b() / 8 / (self.memcase.loopParameter.chiplet_spatial_parameter.Wc \
            * self.memcase.loopParameter.chiplet_spatial_parameter.Hc))

    def get_chiplet_communication(self):
        C0, C1, W1, W2, H1, H2, K1, K2 = self.memcase.loopParameter.get_temporal_count()
        Kp, Hp, _, Wc, Hc = self.memcase.loopParameter.get_spatial_count()
        stride = self.memcase.workload.stride
        kernel_size = self.memcase.workload.kernel_size
        OL1 = cfg.OL1_choices_map[self.memcase.OL1]

        # MUSE-V3 does not support weight roration reuse 
        W: size.Size = size.B(0)

        if self.memcase.loopParameter.rotation_enable:
            A: size.Size = size.B(2 * Kp * (Kp - 1) * OL1.in_tile(kernel_size, stride).size() \
                * C0 * C1 * K2 * K1 * Hp * W1 * H1 * W2 * H2 * Wc * Hc)
        else:
            A: size.Size = size.B(0)
        
        return W + A
    
    def get_dram_communication(self):
        dram_access = self.total_memory_access.OL2_Rd.to_b() + self.total_memory_access.WL1_Wr.to_b() + self.total_memory_access.AL2_Wr.to_b()

        return size.B(dram_access)
    
    def get_runtime(self):
        Fw, Fh = self.memcase.workload.kernel_size.get_params()
        C0, C1, W1, W2, H1, H2, K1, K2 = self.memcase.loopParameter.get_temporal_count()
        Kp = self.memcase.loopParameter.package_spatial_parameter.Kp
        Hp = self.memcase.loopParameter.package_spatial_parameter.Hp
        Kp, Hp, Kc, Wc, Hc = self.memcase.loopParameter.get_spatial_count()
        Csa = self.memcase.loopParameter.get_rotation_count()

        OL1_W0, OL1_H0 = cfg.OL1_choices_map[self.memcase.OL1].get_params()

        chiplet_sub_workload_W = OL1_W0 * W1 * Wc
        chiplet_sub_workload_H = OL1_H0 * H1 * Hc
        chiplet_sub_workload_C = self.computecase.lane * K1 * Kc

        mac: int = (chiplet_sub_workload_W * chiplet_sub_workload_H * chiplet_sub_workload_C) * (W2 * H2 * K2) * \
            (Fw * Fh * C0 * C1 * Csa) * (Kp * Hp)
        system: int = self.computecase.chiplet * self.computecase.core * self.computecase.lane * self.computecase.vector
        
        return mac / system

    def get_energy(self):
        total_runtime = self.get_runtime()
        chiplet_communication = self.get_chiplet_communication()

        Energy_DRAMtoSRAM_A: float = cfg.A_BW_RATIO * self.total_memory_access.AL2_Wr.to_b() * \
            (cfg.Energy_GRS + cfg.Energy_DRAM + cfg.Energy_AL2_Wr)
        Energy_DRAMtoSRAM_W: float = cfg.W_BW_RATIO * self.total_memory_access.WL1_Wr.to_b() * \
            (cfg.Energy_GRS + cfg.Energy_DRAM + cfg.Energy_WL1_Wr)
        Energy_TotalMAC: float = total_runtime * cfg.Energy_MAC * cfg.DATA_WIDTH * (
                self.computecase.lane * self.computecase.vector * self.computecase.chiplet * self.computecase.core)

        # Energy breakdown
        DRAM_energy: float = cfg.A_BW_RATIO * self.total_memory_access.OL2_Rd.to_b() * cfg.Energy_DRAM \
                                + cfg.W_BW_RATIO * self.total_memory_access.WL1_Wr.to_b() * cfg.Energy_DRAM \
                                + cfg.A_BW_RATIO * self.total_memory_access.AL2_Wr.to_b() * cfg.Energy_DRAM \
                                + cfg.A_BW_RATIO * self.total_memory_access.AL2_Wr.to_b() * cfg.Energy_GRS \
                                + cfg.W_BW_RATIO * self.total_memory_access.WL1_Wr.to_b() * cfg.Energy_GRS \
                                + cfg.A_BW_RATIO * self.total_memory_access.OL2_Rd.to_b() * cfg.Energy_GRS

        D2D_energy: float = cfg.A_BW_RATIO * chiplet_communication.to_b() * cfg.Energy_GRS
        A_L2_energy: float = cfg.A_BW_RATIO * self.total_memory_access.AL2_Wr.to_b() * cfg.Energy_AL2_Wr \
                                + cfg.A_BW_RATIO * self.total_memory_access.AL1_Rd.to_b() * cfg.Energy_AL2_Rd

        A_L1_energy: float = cfg.A_BW_RATIO * self.total_memory_access.AL1_Wr.to_b() * cfg.Energy_AL1_Wr \
                                + cfg.A_BW_RATIO * self.total_memory_access.AL1_Rd.to_b() * cfg.Energy_AL1_Rd

        W_L1_energy: float = cfg.W_BW_RATIO * self.total_memory_access.WL1_Wr.to_b() * cfg.Energy_WL1_Wr \
                                + cfg.A_BW_RATIO * self.total_memory_access.WL1_Rd.to_b() * cfg.Energy_WL1_Rd

        output_energy: float = cfg.A_BW_RATIO * self.total_memory_access.OL2_Rd.to_b() * cfg.Energy_OL2_Rd \
                                + cfg.A_BW_RATIO * self.total_memory_access.OL2_Wr.to_b() * cfg.Energy_OL2_Wr \
                                + 4 * self.total_memory_access.OL1_Wr.to_b() * (cfg.get_Energy_RF(size.B(384))) \
                                + 4 * self.total_memory_access.OL1_Rd.to_b() * (cfg.get_Energy_RF(size.B(384)))

        MAC_energy: float = Energy_TotalMAC

        total_energy = DRAM_energy + D2D_energy + A_L2_energy + A_L1_energy + W_L1_energy + output_energy + MAC_energy
        energy_breakdown = EnergyBreakdown(DRAM_energy, D2D_energy, A_L2_energy, A_L1_energy, W_L1_energy, output_energy, Energy_TotalMAC)

        return total_energy, Energy_DRAMtoSRAM_W, Energy_DRAMtoSRAM_A, energy_breakdown
    
    # It can be ignored in MUSE-V3 scheduling
    def get_area(self):
        area_PE:  float = cfg.get_area_PE(self.computecase.vector, self.computecase.lane)
        area_AL1: float = cfg.get_area_SRAM(self.memcase.AL1) * self.computecase.core
        area_AL2: float = cfg.get_area_SRAM(self.memcase.AL2)
        area_WL1: float = cfg.get_area_SRAM(self.real_WL1) * self.computecase.lane * self.computecase.core
        area_OL1: float = cfg.get_area_RF(self.memcase.OL1) * self.computecase.lane * self.computecase.core
        area_OL2: float = cfg.get_area_SRAM(self.OL2_size)
        area_PHY: float = 3 * 387590
        
        area_per_chiplet: float =  area_PE + area_AL1 + area_AL2 + area_WL1 + area_OL1 + area_OL2 + area_PHY
        area_per_package: float = area_per_chiplet * self.computecase.chiplet

        return area_per_chiplet, area_per_package
    
    # It can be ignored in MUSE-V3 scheduling
    def get_mem_footprint(self):
        A_l1_memory = (self.memcase.AL1.to_b() * self.computecase.core)
        W_l1_memory = self.memcase.WL1.to_b() / ( self.memcase.loopParameter.chiplet_spatial_parameter.Wc \
            * self.memcase.loopParameter.chiplet_spatial_parameter.Hc) * self.computecase.lane * self.computecase.core
        A_l2_memory = self.memcase.AL2.to_b()
        o_l1_memory = size.B(192).to_b() * self.computecase.lane * self.computecase.core
        o_l2_memory = (size.B(8 * 8).to_b() * self.computecase.lane * self.computecase.core)

        total_memory = self.computecase.chiplet * (A_l1_memory + W_l1_memory + A_l2_memory + o_l1_memory + o_l2_memory) / (8192)

        return total_memory

    def evaluation(self):
        memory_access: MemoryAccess = self.get_total_memory_access()
        chiplet_communication: size.Size = self.get_chiplet_communication()
        total_runtime = self.get_runtime()
        area_per_chiplet, area_per_package = self.get_area()
        total_energy, Energy_DRAMtoSRAM_W, Energy_DRAMtoSRAM_A, energy_breakdown = self.get_energy()
        total_memory = self.get_mem_footprint()
        return total_memory, memory_access, chiplet_communication, total_runtime, area_per_chiplet, area_per_package, \
            total_energy, Energy_DRAMtoSRAM_W, Energy_DRAMtoSRAM_A, energy_breakdown


@dataclass(frozen=True)
class EnergyBreakdown:
    DRAM_energy: float
    D2D_energy: float
    A_L2_energy: float
    A_L1_energy: float
    W_L1_energy: float
    output_energy: float
    Energy_TotalMAC: float