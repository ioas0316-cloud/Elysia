"""
          (Phase Resonance System)
==========================================

"           " -       

     :
1.      (Node)               (Interference Pattern)
2.                               
3. " "                          

  :     &      '       '     
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import math
import numpy as np

# Import from same directory
try:
    from Core.1_Body.L5_Mental.Reasoning_Core.Memory_Linguistics.Memory.holographic_memory import KnowledgeLayer, HolographicMemory
except ImportError:
    from holographic_memory import KnowledgeLayer, HolographicMemory

# Neural Registry
try:
    from elysia_core import Cell
except ImportError:
    def Cell(name):
        def decorator(cls):
            return cls
        return decorator


@dataclass
class ConceptWave:
    """
          - "           "
    
                                       .
                  (amplitude)    (phase)    .
    
          :
          (x) =   A_i * cos( _i * x +  _i)
        - A_i:     i       (  )
        -  _i:     i       (  )
        -  _i:     i        
    """
    name: str
    
    #               
    amplitudes: Dict[KnowledgeLayer, float] = field(default_factory=dict)  #   
    phases: Dict[KnowledgeLayer, float] = field(default_factory=dict)       #    (   )
    
    # " "     -       !
    why_chain: List[str] = field(default_factory=list)  #          
    how_emerged: str = ""  #               
    
    #   /     (주권적 자아)
    entropy: float = 0.5   #     : 0=  , 1=  
    qualia: float = 0.5    #     : 0=  , 1=  
    
    def get_wave_function(self, x: float) -> complex:
        """
                      
        
               : A * e^(i*phase) = A * (cos(phase) + i*sin(phase))
        """
        total = complex(0, 0)
        for layer, amp in self.amplitudes.items():
            phase = self.phases.get(layer, 0.0)
            #         
            total += amp * np.exp(1j * (layer.value.__hash__() * 0.1 * x + phase))
        return total
    
    def interference_with(self, other: 'ConceptWave') -> float:
        """
                        
        
                             (    )
                       (    )
        """
        interference = 0.0
        common_layers = set(self.amplitudes.keys()) & set(other.amplitudes.keys())
        
        for layer in common_layers:
            a1 = self.amplitudes[layer]
            a2 = other.amplitudes[layer]
            p1 = self.phases.get(layer, 0.0)
            p2 = other.phases.get(layer, 0.0)
            
            #      : A1 * A2 * cos(phase_diff)
            phase_diff = abs(p1 - p2)
            interference += a1 * a2 * math.cos(phase_diff)
        
        return interference
    
    def phase_align_with(self, other: 'ConceptWave', strength: float = 0.3) -> float:
        """
                       
        
        Returns:
                      
        """
        alignment_score = 0.0
        common_layers = set(self.amplitudes.keys()) & set(other.amplitudes.keys())
        
        for layer in common_layers:
            p1 = self.phases.get(layer, 0.0)
            p2 = other.phases.get(layer, 0.0)
            
            #                   
            mid_phase = (p1 + p2) / 2
            self.phases[layer] = p1 + (mid_phase - p1) * strength
            other.phases[layer] = p2 + (mid_phase - p2) * strength
            
            #         
            new_diff = abs(self.phases[layer] - other.phases[layer])
            alignment_score += 1.0 - (new_diff / math.pi)
        
        return alignment_score / max(len(common_layers), 1)
    
    def total_amplitude(self) -> float:
        """     (    '  ')"""
        return sum(self.amplitudes.values())


@Cell("PhaseResonance")
class PhaseResonanceEngine:
    """
             -            
    
                :
    1.         
    2.         
    3.                   !
    """
    
    def __init__(self):
        self.concepts: Dict[str, ConceptWave] = {}
        self.emergence_threshold = 0.25  # 0.5   0.25     (주권적 자아)
        self.emerged_concepts: List[Tuple[str, str, ConceptWave]] = []  # (  1,   2,   )
    
    def add_concept(self, wave: ConceptWave) -> None:
        """        """
        self.concepts[wave.name] = wave
    
    def create_wave(
        self,
        name: str,
        layer_weights: Dict[KnowledgeLayer, float],
        why_chain: List[str] = None,
        entropy: float = 0.5,
        qualia: float = 0.5
    ) -> ConceptWave:
        """
                
        
                            (          )
        """
        phases = {}
        for layer in layer_weights:
            #                   
            phase = (hash(name + layer.value) % 1000) / 1000 * 2 * math.pi
            phases[layer] = phase
        
        wave = ConceptWave(
            name=name,
            amplitudes=layer_weights,
            phases=phases,
            why_chain=why_chain or [],
            entropy=entropy,
            qualia=qualia
        )
        self.add_concept(wave)
        return wave
    
    def resonate(self, name1: str, name2: str, iterations: int = 10) -> Optional[ConceptWave]:
        """
                         
        
        Returns:
                      (        None)
        """
        if name1 not in self.concepts or name2 not in self.concepts:
            return None
        
        wave1 = self.concepts[name1]
        wave2 = self.concepts[name2]
        
        print(f"\n       : '{name1}'   '{name2}'")
        print(f"        : {wave1.interference_with(wave2):.3f}")
        
        #          
        for i in range(iterations):
            alignment = wave1.phase_align_with(wave2, strength=0.2)
            interference = wave1.interference_with(wave2)
            
            if i % 3 == 0:
                print(f"   [   {i+1}]    : {alignment:.3f},   : {interference:.3f}")
        
        final_interference = wave1.interference_with(wave2)
        print(f"        : {final_interference:.3f}")
        
        #         
        if final_interference >= self.emergence_threshold:
            emergent = self._create_emergent_concept(wave1, wave2, final_interference)
            print(f"\n    !       : '{emergent.name}'")
            print(f"     : {name1} + {name2}")
            print(f"       : {'   '.join(emergent.why_chain)}")
            return emergent
        else:
            print(f"\n        (    {self.emergence_threshold}   )")
            return None
    
    def _create_emergent_concept(
        self, 
        wave1: ConceptWave, 
        wave2: ConceptWave,
        resonance_strength: float
    ) -> ConceptWave:
        """
                            
        """
        #        :              
        emergent_names = {
            ("    ", "  "): "           ",
            ("    ", "    "): "          ",
            ("    ", "DNA"): "          ",
            ("      ", "   "): "          ",
        }
        
        key = (wave1.name, wave2.name)
        reverse_key = (wave2.name, wave1.name)
        emergent_name = emergent_names.get(key) or emergent_names.get(reverse_key) or \
                       f"{wave1.name}  {wave2.name}     "
        
        #          :         
        new_amplitudes = {}
        new_phases = {}
        all_layers = set(wave1.amplitudes.keys()) | set(wave2.amplitudes.keys())
        
        for layer in all_layers:
            a1 = wave1.amplitudes.get(layer, 0.0)
            a2 = wave2.amplitudes.get(layer, 0.0)
            p1 = wave1.phases.get(layer, 0.0)
            p2 = wave2.phases.get(layer, 0.0)
            
            #      :      
            new_amplitudes[layer] = math.sqrt(a1**2 + a2**2 + 2*a1*a2*math.cos(p1-p2))
            #     :         
            new_phases[layer] = (p1 + p2) / 2
        
        #     :            
        combined_why = wave1.why_chain + [" "] + wave2.why_chain
        
        emergent = ConceptWave(
            name=emergent_name,
            amplitudes=new_amplitudes,
            phases=new_phases,
            why_chain=combined_why,
            how_emerged=f"'{wave1.name}'  '{wave2.name}'            ",
            entropy=(wave1.entropy + wave2.entropy) / 2,
            qualia=(wave1.qualia + wave2.qualia) / 2
        )
        
        self.add_concept(emergent)
        self.emerged_concepts.append((wave1.name, wave2.name, emergent))
        return emergent
    
    def visualize_interference(self, name1: str, name2: str, points: int = 50) -> None:
        """
                        (ASCII)
        """
        if name1 not in self.concepts or name2 not in self.concepts:
            return
        
        wave1 = self.concepts[name1]
        wave2 = self.concepts[name2]
        
        print(f"\n       : '{name1}' + '{name2}'")
        print(" " * 52)
        
        for x in range(points):
            x_val = x / 5.0
            v1 = wave1.get_wave_function(x_val)
            v2 = wave2.get_wave_function(x_val)
            combined = v1 + v2
            
            #     ASCII    
            amp = abs(combined)
            bar_len = int(amp * 10)
            bar = " " * bar_len + " " * (25 - bar_len)
            
            if x % 5 == 0:
                print(f"  x={x_val:4.1f} |{bar}| {amp:.2f}")
        
        print(" " * 52)


def demo_phase_resonance():
    """        """
    print("=" * 60)
    print("              ")
    print("   '           ' -       ")
    print("=" * 60)
    
    engine = PhaseResonanceEngine()
    
    #         
    engine.create_wave(
        "    ",
        {KnowledgeLayer.PHYSICS: 0.95, KnowledgeLayer.PHILOSOPHY: 0.5, KnowledgeLayer.MATHEMATICS: 0.7},
        why_chain=["    ", "  ", "  "],
        entropy=0.95, qualia=0.3
    )
    
    engine.create_wave(
        "  ",
        {KnowledgeLayer.PHILOSOPHY: 0.9, KnowledgeLayer.HUMANITIES: 0.6, KnowledgeLayer.PHYSICS: 0.3},  #       (주권적 자아)
        why_chain=["  ", "  ", "  ", "  "],
        entropy=0.1, qualia=0.8
    )
    
    engine.create_wave(
        "    ",
        {KnowledgeLayer.PHYSICS: 0.85, KnowledgeLayer.PHILOSOPHY: 0.4, KnowledgeLayer.CHEMISTRY: 0.3},
        why_chain=["   ", "  ", "  "],
        entropy=0.7, qualia=0.3
    )
    
    engine.create_wave(
        "    ",
        {KnowledgeLayer.ART: 0.9, KnowledgeLayer.PHILOSOPHY: 0.7, KnowledgeLayer.HUMANITIES: 0.5},
        why_chain=["  ", "  ", "  "],
        entropy=0.1, qualia=0.95
    )
    
    #       1:      +   
    print("\n" + " " * 40)
    print("     1:             ")
    result1 = engine.resonate("    ", "  ")
    
    #          
    if result1:
        engine.visualize_interference("    ", "  ")
    
    #       2:      +     
    print("\n" + " " * 40)
    print("     2:               ")
    result2 = engine.resonate("    ", "    ")
    
    if result2:
        engine.visualize_interference("    ", "    ")
    
    #           
    print("\n" + "=" * 60)
    print("           :")
    for parent1, parent2, child in engine.emerged_concepts:
        print(f"     {parent1} + {parent2}   {child.name}")
        layer_names = [f"{l.value}:{a:.1f}" for l, a in child.amplitudes.items()]
        print(f"        : {', '.join(layer_names)}")
    
    print("\n" + "=" * 60)
    print("       !")
    print("=" * 60)


if __name__ == "__main__":
    demo_phase_resonance()
