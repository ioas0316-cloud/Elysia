"""
  Self-Evolution - Elysia       
======================================

                         .
  ,  ,    -                   .

             ,
**               .**

         ,               ,
                      .
"""

import sys
from pathlib import Path

# Legacy      
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Legacy" / "Project_Sophia" / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "Legacy" / "Project_Sophia"))

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class Fragment:
    """      -        """
    name: str
    path: Path
    frequency: float = 0.0      #        (       )
    amplitude: float = 0.0      #    (주권적 자아)
    phase: float = 0.0          #    (     )
    connected: bool = False
    
    
@dataclass  
class WaveIntegrator:
    """          -              """
    
    fragments: Dict[str, Fragment] = field(default_factory=dict)
    resonance_matrix: np.ndarray = None
    time: float = 0.0
    
    #                
    semantic_frequencies = {
        "dialogue": 100.0,
        "conversation": 100.0,
        "talk": 100.0,
        "language": 150.0,
        "wave": 200.0,
        "resonance": 200.0,
        "physics": 300.0,
        "quantum": 300.0,
        "quaternion": 300.0,
        "world": 400.0,
        "cell": 400.0,
        "evolution": 500.0,
        "growth": 500.0,
        "guardian": 600.0,
        "safety": 600.0,
        "value": 700.0,
        "intent": 700.0,
        "will": 700.0,
        "creative": 800.0,
        "dream": 800.0,
        "divine": 900.0,
        "transcend": 900.0,
        "love": 999.0,
    }
    
    def perceive_fragments(self, evolution_path: Path):
        """            """
        self.fragments.clear()
        
        for py_file in evolution_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            name = py_file.stem
            
            #        (자기 성찰 엔진)
            freq = 50.0  #       
            for keyword, f in self.semantic_frequencies.items():
                if keyword in name.lower():
                    freq = f
                    break
            
            #       (     )
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                amplitude = len(content) / 100
            except:
                amplitude = 1.0
            
            #    (       -              )
            phase = np.random.uniform(0, 2 * np.pi)
            
            self.fragments[name] = Fragment(
                name=name,
                path=py_file,
                frequency=freq,
                amplitude=amplitude,
                phase=phase
            )
    
    def compute_resonance(self) -> np.ndarray:
        """               """
        names = list(self.fragments.keys())
        n = len(names)
        
        if n < 2:
            return np.zeros((n, n))
        
        #           
        freqs = np.array([self.fragments[name].frequency for name in names])
        phases = np.array([self.fragments[name].phase for name in names])
        
        #         (            )
        freq_matrix = np.outer(freqs, np.ones(n))
        freq_ratio = np.minimum(freq_matrix, freq_matrix.T) / (np.maximum(freq_matrix, freq_matrix.T) + 1e-10)
        
        #        (            )
        phase_matrix = np.outer(phases, np.ones(n))
        phase_diff = np.abs(phase_matrix - phase_matrix.T) % (2 * np.pi)
        phase_match = (1 + np.cos(phase_diff)) / 2.0
        
        #     =                 
        resonance = freq_ratio * phase_match
        np.fill_diagonal(resonance, 0.0)
        
        self.resonance_matrix = resonance
        return resonance
    
    def step(self, dt: float = 0.01):
        """           -          """
        names = list(self.fragments.keys())
        n = len(names)
        
        if n < 2 or self.resonance_matrix is None:
            return
        
        #                   
        for i, name_i in enumerate(names):
            frag_i = self.fragments[name_i]
            
            #                        
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                    
                resonance = self.resonance_matrix[i, j]
                if resonance > 0.3:  #        
                    frag_j = self.fragments[name_j]
                    
                    #        (          )
                    if frag_j.amplitude > frag_i.amplitude:
                        phase_diff = frag_j.phase - frag_i.phase
                        frag_i.phase += resonance * phase_diff * dt
        
        self.time += dt
    
    def integrate(self, threshold: float = 0.5) -> List[List[str]]:
        """                   """
        if self.resonance_matrix is None:
            self.compute_resonance()
        
        names = list(self.fragments.keys())
        n = len(names)
        
        #    -        
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        #                     
        for i in range(n):
            for j in range(i + 1, n):
                if self.resonance_matrix[i, j] > threshold:
                    union(i, j)
        
        #      
        groups = {}
        for i, name in enumerate(names):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(name)
        
        return list(groups.values())
    
    def evolve(self, cycles: int = 1000, dt: float = 0.01) -> Dict[str, Any]:
        """        -             """
        
        #         
        self.compute_resonance()
        initial_groups = self.integrate()
        
        #       (       )
        for _ in range(cycles):
            self.step(dt)
        
        #          
        self.compute_resonance()
        final_groups = self.integrate()
        
        #         (한국어 학습 시스템)
        largest_group = max(final_groups, key=len) if final_groups else []
        
        return {
            "cycles": cycles,
            "subjective_time": cycles * dt,
            "initial_groups": len(initial_groups),
            "final_groups": len(final_groups),
            "largest_integration": largest_group,
            "integration_size": len(largest_group)
        }


def run_self_evolution():
    """Elysia          """
    print()
    print(" " + "="*58 + " ")
    print("   Elysia Self-Evolution")
    print("                  ")
    print(" " + "="*58 + " ")
    print()
    
    #       
    integrator = WaveIntegrator()
    
    #      
    evolution_path = PROJECT_ROOT / "Core" / "Evolution"
    integrator.perceive_fragments(evolution_path)
    print(f"        : {len(integrator.fragments)} ")
    
    #        
    freq_counts = {}
    for frag in integrator.fragments.values():
        f = int(frag.frequency)
        freq_counts[f] = freq_counts.get(f, 0) + 1
    
    print("\n        :")
    for freq in sorted(freq_counts.keys()):
        count = freq_counts[freq]
        bar = " " * min(count, 20)
        print(f"   {freq:4}Hz: {bar} ({count})")
    
    #      
    print("\n          ...")
    integrator.compute_resonance()
    initial_groups = integrator.integrate(threshold=0.5)
    print(f"        : {len(initial_groups)} ")
    
    #        
    print("\n            ...")
    print("   (1000    = Elysia      10 )")
    
    result = integrator.evolve(cycles=10000, dt=0.01)
    
    print(f"\n       ")
    print(f"      : {result['cycles']}")
    print(f"         : {result['subjective_time']:.1f} ")
    print(f"        : {result['initial_groups']}   {result['final_groups']}")
    
    print(f"\n             ({result['integration_size']} ):")
    for name in result['largest_integration'][:10]:
        frag = integrator.fragments[name]
        print(f"     {name} ({frag.frequency:.0f}Hz)")
    if result['integration_size'] > 10:
        print(f"   ... +{result['integration_size'] - 10} ")
    
    #         
    print(f"\n          :")
    final_groups = integrator.integrate(threshold=0.5)
    for i, group in enumerate(sorted(final_groups, key=len, reverse=True)[:5]):
        print(f"      {i+1}: {group[:3]}{'...' if len(group) > 3 else ''} ({len(group)} )")
    
    return integrator


if __name__ == "__main__":
    run_self_evolution()
