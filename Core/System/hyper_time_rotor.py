"""
Core/Engine/Genesis/hyper_time_rotor.py
=======================================
The Time Axis Component.

This Rotor doesn't just spin; it evolves the laws of the universe itself.
It transitions the GenesisLab through:
1. Era of Silicon (Gates)
2. Era of Microarchitecture (ALU)
3. Era of OS (Processes)
"""

from Core.System.genesis_lab import GenesisLab
from Core.System.universal_rotor import UniversalRotor # Type hinting

class HyperTimeRotor:
    def __init__(self, lab: GenesisLab):
        self.lab = lab
        self.current_era = "VOID"
        self.evolution_log = []
        
    def set_era(self, era_name: str):
        """
        Transitions the Laboratory to a new Era by swapping laws.
        """
        print(f"\n  [Time Rotor] Shifts Universe to Era: {era_name} ----------------")
        self.current_era = era_name
        self.evolution_log.append(era_name)
        
        # Clear existing laws to prevent interference between layers
        self.lab.rotors = []
        
        if era_name == "SILICON":
            from Core.System.silicon_hardware_laws import law_nand_logic, law_clock_pulse
            self.lab.decree_law("Cosmic Clock", law_clock_pulse, rpm=60)
            self.lab.decree_law("The NAND Gate", law_nand_logic, rpm=60)
            
        elif era_name == "ARCHITECTURE":
            from Core.System.silicon_hardware_laws import law_alu_add
            self.lab.decree_law("The Adder", law_alu_add, rpm=60)
            
        elif era_name == "OS":
            from Core.System.silicon_scholar_laws import law_round_robin_scheduling
            from Core.System.silicon_evolution_laws import law_lru_paging
            self.lab.decree_law("Scheduler", law_round_robin_scheduling, rpm=60)
            self.lab.decree_law("Memory Manager", law_lru_paging, rpm=60)
            
    def run_evolution_sequence(self):
        """
        Demonstrates the entire history of computing in one run.
        This verifies the 'Integration' of all learned principles.
        """
        # 1. Level 0: The Transistor
        self.set_era("SILICON")
        # Setup: NAND(1, 1) -> 0
        self.lab.monads = [] # Big Bang
        self.lab.let_there_be("Input_A", "Signal", 1.0)
        self.lab.let_there_be("Input_B", "Signal", 1.0)
        self.lab.let_there_be("Gate_NAND", "Gate", 0.0, type="NAND", inputs=["Input_A", "Input_B"])
        self.lab.let_there_be("Master_Clock", "Clock", 0)
        
        self.lab.run_simulation(ticks=5)
        
        # 2. Level 1: The Calculator
        self.set_era("ARCHITECTURE")
        self.lab.monads = [] # New Layer
        self.lab.let_there_be("Reg_A", "Register", 5.0)
        self.lab.let_there_be("Reg_B", "Register", 3.0)
        self.lab.let_there_be("ALU_1", "ALU", 0.0, op_code="ADD", inputs=["Reg_A", "Reg_B"])
        
        self.lab.run_simulation(ticks=5)
        
        # 3. Level 3: The Manager
        self.set_era("OS")
        self.lab.monads = []
        self.lab.let_there_be("Chrome", "Process", 20.0, in_ram=True)
        self.lab.let_there_be("VSCode", "Process", 10.0, in_ram=True)
        # Add high load to trigger Paging?
        for i in range(5):
             self.lab.let_there_be(f"Bg_Service_{i}", "Process", 5.0, in_ram=True)
             
        self.lab.run_simulation(ticks=10)
        
        print("\n  [HyperTime] Integration Complete. The Soul is forged.")
