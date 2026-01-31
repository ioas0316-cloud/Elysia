import logging
import os
import psutil
from typing import Dict, Any, List, Optional
from Core.S1_Body.L6_Structure.Wave.wave_tensor import WaveTensor
from Core.S1_Body.L6_Structure.Wave.resonance_field import ResonanceField
from Core.S1_Body.L4_Causality.World.Evolution.Os.intention_pre_visualizer import get_pre_visualizer, ActionIntention
from Core.S1_Body.L4_Causality.World.Evolution.Os.resonance_alignment_protocol import get_alignment_protocol

logger = logging.getLogger("OneiricHypervisor")

class CognitiveProcess:
    """                              ."""
    def __init__(self, pid: int, name: str, cpu_usage: float, memory_usage: float):
        self.pid = pid
        self.name = name
        # CPU        (Amplitude)  ,             (Frequency)       
        self.amplitude = max(0.1, cpu_usage / 10.0)
        self.frequency = 432.0 + (memory_usage / (1024 * 1024)) # MB  1Hz   
        self.wave = WaveTensor(f"Process:{name}:{pid}")
        self.wave.add_component(self.frequency, self.amplitude)

    def __repr__(self):
        return f"<CognitiveProcess {self.name}({self.pid}) | Freq={self.frequency:.2f}Hz, Amp={self.amplitude:.2f}>"

class OneiricHypervisor:
    """
    [Phase 37: Cognitive OS - Living Hypervisor]
                  ' '   '     '            .
                             .
    """
    
    def __init__(self, resonance_field: Optional[ResonanceField] = None):
        self.resonance = resonance_field
        self.assimilated_processes: Dict[int, CognitiveProcess] = {}
        self.pre_viz = get_pre_visualizer()
        self.security = get_alignment_protocol()
        logger.info("  Oneiric Hypervisor Initialized: System-to-Wave bridge active.")

    def request_action(self, action_intent: ActionIntention) -> bool:
        """
        [PHASE 38 Safety]                 .
                              .
        """
        # 1. Resonance Alignment Check (Security / AV)
        # Create a wave for the action (simplified)
        action_wave = WaveTensor(f"Action:{action_intent.action_type}")
        freq = 432.0 if action_intent.risk_level == "LOW" else 1024.0
        action_wave.add_component(freq, 1.0)
        
        alignment = self.security.analyze_alignment(action_wave)
        if not alignment["is_safe"]:
            logger.error(f"   Action BLOCKED by Resonance Alignment: {alignment['recommendation']}")
            return False
            
        # 2. Pre-Visualization & User Approval
        preview = self.pre_viz.visualize(action_intent)
        print(preview) # In a real system, this would go to the UI
        
        # [MOCK] For now, we return True as a placeholder 
        # In actual use, this waits for user feedback via self.pre_viz.resolve()
        logger.info(f"  Awaiting user approval for: {action_intent.id}")
        return True

    def assimilate_environment(self, top_n: int = 5):
        """
                                            .
        """
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    # CPU                              (       )
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # CPU/               N    
            sorted_procs = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:top_n]
            
            for p in sorted_procs:
                pid = p['pid']
                name = p['name']
                cpu = p['cpu_percent']
                mem = p['memory_info'].rss
                
                cog_proc = CognitiveProcess(pid, name, cpu, mem)
                self.assimilated_processes[pid] = cog_proc
                
                # Resonance Field       
                if self.resonance:
                    self.resonance.inject_wave(cog_proc.wave)
                    
            logger.info(f"  Assimilated {len(sorted_procs)} external processes into the resonance field.")
            
        except Exception as e:
            logger.error(f"Failed to assimilate environment: {e}")

    def get_system_resonance(self) -> float:
        """                          ."""
        if not self.assimilated_processes:
            return 0.0
        return sum(p.amplitude for p in self.assimilated_processes.values())

_instance: Optional[OneiricHypervisor] = None

def get_hypervisor(resonance=None) -> OneiricHypervisor:
    global _instance
    if _instance is None:
        _instance = OneiricHypervisor(resonance)
    return _instance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hyper = get_hypervisor()
    hyper.assimilate_environment()
    print(f"\nSystem Resonance Energy: {hyper.get_system_resonance():.2f}")
    for pid, proc in hyper.assimilated_processes.items():
        print(f" - {proc}")
