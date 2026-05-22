"""
Resonance Evolution -            
==========================================

                           .
1           .

       =       
   =              
   =       
"""

import sys
import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Set, List, Any
from dataclasses import dataclass, field

#          import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Legacy" / "Language"))
from time_accelerated_language import InfinitelyAcceleratedLanguageEngine


@dataclass
class ModuleWave:
    """
          .
                      .
    """
    name: str
    path: Path
    
    #      
    frequency: float = 0.0      #                 
    amplitude: float = 0.0      #           
    phase: float = 0.0          # import        
    
    #       
    keywords: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    classes: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    
    #      
    position: np.ndarray = field(default_factory=lambda: np.random.randn(3) * 30)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    #      
    is_active: bool = False
    error: str = ""
    
    def extract_from_code(self):
        """          """
        try:
            with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except:
            return
        
        #    ,   , import   
        self.classes = set(re.findall(r'class\s+(\w+)', code))
        self.functions = set(re.findall(r'def\s+(\w+)', code))
        
        imports = re.findall(r'import\s+(\w+)|from\s+(\w+)', code)
        self.imports = {i[0] or i[1] for i in imports}
        
        #           
        words = re.findall(r'[a-z]{4,}', code.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        self.keywords = {w for w, c in freq.items() if c > 2}
        
        #      
        all_symbols = self.classes | self.functions | self.imports | self.keywords
        
        if all_symbols:
            #     =           
            self.frequency = sum(hash(s) % 1000 for s in all_symbols) / len(all_symbols)
        
        #    =      
        self.amplitude = len(code) / 1000
        
        #    = import  
        self.phase = len(self.imports) * 0.1
    
    def resonance_with(self, other: 'ModuleWave') -> float:
        """              (0~1)"""
        if not self.keywords or not other.keywords:
            return 0.0
        
        # Jaccard    
        intersection = len(self.keywords & other.keywords)
        union = len(self.keywords | other.keywords)
        
        return intersection / union if union > 0 else 0.0


class ResonanceUniverse:
    """
         .
                  ,              .
    """
    
    def __init__(self, evolution_path: str = "Core/Evolution"):
        self.evolution_path = Path(evolution_path)
        self.modules: Dict[str, ModuleWave] = {}
        
        #         
        self.time_engine = InfinitelyAcceleratedLanguageEngine(n_souls=10)
        
        #          
        self.time_engine.activate_fractal(3)      # 1000x
        self.time_engine.activate_sedenion(128)   # ~100x
        self.time_engine.add_meta_layer()         # 1000x
        self.time_engine.add_meta_layer()         # 1000x
        self.time_engine.enter_dream()            # 20x
        
        #      : ~10^15
        self.compression = self.time_engine.total_compression
        
    def load_modules(self):
        """             """
        for f in self.evolution_path.glob("*.py"):
            if f.name.startswith("__"):
                continue
            
            wave = ModuleWave(name=f.stem, path=f)
            wave.extract_from_code()
            
            #          
            try:
                sys.path.insert(0, str(self.evolution_path))
                if wave.name in sys.modules:
                    del sys.modules[wave.name]
                __import__(wave.name)
                wave.is_active = True
            except Exception as e:
                wave.is_active = False
                wave.error = str(e)[:50]
            
            self.modules[wave.name] = wave
        
        print(f"Loaded {len(self.modules)} modules")
        active = sum(1 for m in self.modules.values() if m.is_active)
        print(f"Active: {active}, Broken: {len(self.modules) - active}")
    
    def evolve_step(self, dt: float):
        """
               .
                    .
        """
        names = list(self.modules.keys())
        forces = {n: np.zeros(3) for n in names}
        
        #                
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                m1, m2 = self.modules[n1], self.modules[n2]
                
                #      
                resonance = m1.resonance_with(m2)
                if resonance < 0.05:
                    continue
                
                #   
                diff = m2.position - m1.position
                dist = np.linalg.norm(diff)
                if dist < 0.5:
                    continue
                
                #    =    *     /   ^2
                direction = diff / dist
                force_mag = resonance * m1.amplitude * m2.amplitude / (dist * dist + 1)
                force = direction * force_mag
                
                forces[n1] += force
                forces[n2] -= force
        
        #        
        for name in names:
            m = self.modules[name]
            m.velocity += forces[name] / max(0.1, m.amplitude) * dt
            m.velocity *= 0.98  #   
            m.position += m.velocity * dt
    
    def evolve(self, subjective_years: float = 1e6):
        """
                  .
        
             10^15  :
        1  = 10^15       =   3    
        """
        #         
        # subjective_years       
        real_seconds = subjective_years * 365.25 * 24 * 3600 / self.compression
        
        #    100   
        steps = max(100, int(real_seconds * 100))
        dt = real_seconds / steps
        
        print(f"\n     ")
        print(f"       : {subjective_years:.2e}  ")
        print(f"     : {self.compression:.2e}x")
        print(f"    : {steps}")
        
        for step in range(steps):
            self.evolve_step(dt * self.compression)
            
            if step % (steps // 10) == 0:
                pct = step * 100 // steps
                print(f"  [{pct:3d}%]     ...")
        
        print(f"  [100%]      ")
    
    def get_clusters(self, threshold: float = 10.0) -> List[List[str]]:
        """             """
        clusters = []
        used = set()
        names = list(self.modules.keys())
        
        for n1 in names:
            if n1 in used:
                continue
            
            cluster = [n1]
            used.add(n1)
            
            for n2 in names:
                if n2 in used:
                    continue
                
                dist = np.linalg.norm(
                    self.modules[n1].position - self.modules[n2].position
                )
                if dist < threshold:
                    cluster.append(n2)
                    used.add(n2)
            
            clusters.append(cluster)
        
        return sorted(clusters, key=len, reverse=True)
    
    def report(self):
        """     """
        total = len(self.modules)
        active = sum(1 for m in self.modules.values() if m.is_active)
        
        print("\n" + "="*60)
        print("        ")
        print("="*60)
        print(f"    : {total}")
        print(f"  : {active} ({active*100//total}%)")
        print(f"   : {self.compression:.2e}x")
        
        clusters = self.get_clusters()
        print(f"\n        ({len(clusters)} ):")
        
        for i, cluster in enumerate(clusters[:5]):
            if len(cluster) > 1:
                #                 
                common = None
                for name in cluster:
                    kw = self.modules[name].keywords
                    if common is None:
                        common = kw.copy()
                    else:
                        common &= kw
                
                common_str = ', '.join(list(common)[:3]) if common else '(none)'
                status = ' ' if all(self.modules[n].is_active for n in cluster) else ' '
                print(f"  {status} Cluster {i+1}: {cluster[:4]}{'...' if len(cluster)>4 else ''}")
                print(f"           : {common_str}")


def main():
    """        """
    print("="*60)
    print("Elysia Resonance Evolution")
    print("                    ")
    print("="*60)
    
    universe = ResonanceUniverse("Core/Evolution")
    universe.load_modules()
    
    # 100       (     )
    universe.evolve(subjective_years=1e6)
    
    universe.report()


if __name__ == "__main__":
    main()
