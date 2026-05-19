"""
        (Knowledge Spectrum)
==================================

"              " -       

     :
-     (   )    (  )   
-                      
-                     (  )

         :
-           "   "     
-         

    :
-      (Diffusion Dynamics)   
-                  
-                 
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from enum import Enum

# Neural Registry
try:
    from elysia_core import Cell
except ImportError:
    def Cell(name):
        def decorator(cls):
            return cls
        return decorator


class SpectrumDomain(Enum):
    """          (     )"""
    PHYSICS = "  "
    CHEMISTRY = "  "
    BIOLOGY = "  "
    ART = "  "
    HUMANITIES = "  "
    PHILOSOPHY = "  "
    MATHEMATICS = "  "


@Cell("KnowledgeSpectrum")
class KnowledgeSpectrum:
    """
            -               
    
          :
    -      =    (  )
    -    =   (주권적 자아)
    -     =                  
    """
    
    def __init__(self, resolution: int = 100):
        """
        Args:
            resolution:          (       )
        """
        self.resolution = resolution
        self.domains = list(SpectrumDomain)
        
        #        1D      (    2D/3D       )
        # field[domain] = numpy array of concentrations
        self.field: Dict[SpectrumDomain, np.ndarray] = {
            domain: np.zeros(resolution) for domain in self.domains
        }
        
        #         (              )
        self.poured_concepts: Dict[str, Dict] = {}
        
        #         
        self.crystals: List[Dict] = []
        
        #       (코드 베이스 구조 로터)
        self.diffusion_rate = 0.15
        
        #           (한국어 학습 시스템)
        self.crystal_threshold = 0.8
    
    def pour(
        self, 
        concept: str,
        domains: Dict[SpectrumDomain, float],
        position: float = 0.5,
        intensity: float = 1.0
    ) -> None:
        """
                    
        
        Args:
            concept:      
            domains:               
            position:              (0.0~1.0)
            intensity:    (    )
        """
        pos_idx = int(position * (self.resolution - 1))
        
        #                         
        for domain, weight in domains.items():
            #           (                )
            x = np.arange(self.resolution)
            gaussian = np.exp(-0.5 * ((x - pos_idx) / 5) ** 2)
            self.field[domain] += gaussian * weight * intensity
        
        self.poured_concepts[concept] = {
            "domains": domains,
            "position": position,
            "intensity": intensity
        }
        
        print(f"  '{concept}'     (  : {position:.1f},   : {intensity})")
    
    def diffuse(self, steps: int = 1) -> None:
        """
              (                 )
        
          :  C/ t = D   C (주권적 자아)
        """
        for _ in range(steps):
            for domain in self.domains:
                c = self.field[domain]
                
                #         :   C   C[i-1] - 2*C[i] + C[i+1]
                laplacian = np.zeros_like(c)
                laplacian[1:-1] = c[:-2] - 2*c[1:-1] + c[2:]
                
                #       (   )
                laplacian[0] = c[1] - c[0]
                laplacian[-1] = c[-2] - c[-1]
                
                #   
                self.field[domain] += self.diffusion_rate * laplacian
                
                #      
                self.field[domain] = np.maximum(self.field[domain], 0)
    
    def find_meetings(self) -> List[Tuple[int, float, Set[SpectrumDomain]]]:
        """
                        
        
        Returns:
            [(  ,    ,        ), ...]
        """
        meetings = []
        threshold = 0.3  #           "   "
        
        for i in range(self.resolution):
            present_domains = set()
            total_concentration = 0
            
            for domain in self.domains:
                c = self.field[domain][i]
                if c >= threshold:
                    present_domains.add(domain)
                    total_concentration += c
            
            # 2                
            if len(present_domains) >= 2:
                meetings.append((i, total_concentration, present_domains))
        
        return meetings
    
    def crystallize(self) -> List[Dict]:
        """
                       (    /     )
        
                            !
        """
        meetings = self.find_meetings()
        new_crystals = []
        
        for pos, concentration, domains in meetings:
            if concentration >= self.crystal_threshold:
                #    !
                crystal = {
                    "position": pos / self.resolution,
                    "concentration": concentration,
                    "domains": domains,
                    "name": self._generate_crystal_name(domains),
                    "parents": self._find_parents_at(pos)
                }
                
                #                     
                existing = [c for c in self.crystals if abs(c["position"] - crystal["position"]) < 0.05]
                if not existing:
                    self.crystals.append(crystal)
                    new_crystals.append(crystal)
                    
                    #                    (         )
                    for domain in domains:
                        self.field[domain][pos] *= 0.5
        
        return new_crystals
    
    def _generate_crystal_name(self, domains: Set[SpectrumDomain]) -> str:
        """        """
        domain_names = sorted([d.value for d in domains])
        
        #             
        name_map = {
            frozenset(["  ", "  "]): "       ",
            frozenset(["  ", "  ", "  "]): "      ",
            frozenset(["  ", "  "]): "      ",
            frozenset(["  ", "  "]): "     ",
            frozenset(["  ", "  "]): "      ",
            frozenset(["  ", "  "]): "      ",
        }
        
        key = frozenset(domain_names)
        return name_map.get(key, f"{' '.join(domain_names)}    ")
    
    def _find_parents_at(self, pos: int) -> List[str]:
        """                    """
        parents = []
        for name, info in self.poured_concepts.items():
            concept_pos = int(info["position"] * (self.resolution - 1))
            #                
            if abs(concept_pos - pos) < self.resolution // 4:
                parents.append(name)
        return parents
    
    def simulate(self, diffusion_steps: int = 50, verbose: bool = True) -> None:
        """
                   
        
        1.    (자기 성찰 엔진)
        2.     (            )
        """
        if verbose:
            print(f"\n        ({diffusion_steps}   )...")
        
        crystals_formed = []
        
        for step in range(diffusion_steps):
            self.diffuse(steps=1)
            
            #             
            if step % 10 == 0:
                new_crystals = self.crystallize()
                if new_crystals and verbose:
                    for c in new_crystals:
                        print(f"     [{step}  ]    : '{c['name']}'")
                        print(f"        : {c['position']:.2f},   : {c['concentration']:.2f}")
                        print(f"        : {', '.join(c['parents'])}")
                crystals_formed.extend(new_crystals)
        
        if verbose:
            print(f"\n    {len(crystals_formed)}         ")
    
    def visualize_spectrum(self, show_domains: List[SpectrumDomain] = None) -> None:
        """        ASCII    """
        domains = show_domains or self.domains[:3]  #    3     
        
        print("\n         :")
        print(" " * 60)
        
        for domain in domains:
            c = self.field[domain]
            line = f"{domain.value:>4}: "
            
            # 10         
            for i in range(10):
                start = i * (self.resolution // 10)
                end = start + (self.resolution // 10)
                avg = np.mean(c[start:end])
                
                if avg > 0.8:
                    line += " "
                elif avg > 0.5:
                    line += " "
                elif avg > 0.2:
                    line += " "
                elif avg > 0.05:
                    line += " "
                else:
                    line += " "
            
            max_val = np.max(c)
            line += f" | max: {max_val:.2f}"
            print(line)
        
        #         
        if self.crystals:
            crystal_line = "  : "
            for c in self.crystals:
                pos = int(c["position"] * 10)
                crystal_line += " " * (pos - len(crystal_line) + 6) + " "
            print(crystal_line)
        
        print(" " * 60)


def demo_knowledge_spectrum():
    """           -          """
    print("=" * 60)
    print("            ")
    print("   '              ' -       ")
    print("=" * 60)
    
    spectrum = KnowledgeSpectrum(resolution=100)
    
    #      "  "
    print("\n       :")
    
    spectrum.pour("    ", {
        SpectrumDomain.PHYSICS: 0.9,
        SpectrumDomain.MATHEMATICS: 0.7,
        SpectrumDomain.PHILOSOPHY: 0.4
    }, position=0.2, intensity=1.5)
    
    spectrum.pour("  ", {
        SpectrumDomain.PHILOSOPHY: 0.9,
        SpectrumDomain.HUMANITIES: 0.6
    }, position=0.35, intensity=1.2)
    
    spectrum.pour("    ", {
        SpectrumDomain.PHYSICS: 0.85,
        SpectrumDomain.CHEMISTRY: 0.5,
        SpectrumDomain.PHILOSOPHY: 0.3
    }, position=0.6, intensity=1.0)
    
    spectrum.pour("    ", {
        SpectrumDomain.ART: 0.9,
        SpectrumDomain.PHILOSOPHY: 0.7
    }, position=0.75, intensity=1.3)
    
    #      
    print("\n        (       ):")
    spectrum.visualize_spectrum([
        SpectrumDomain.PHYSICS,
        SpectrumDomain.PHILOSOPHY,
        SpectrumDomain.ART
    ])
    
    #         
    spectrum.simulate(diffusion_steps=80, verbose=True)
    
    #      
    print("\n        (    ):")
    spectrum.visualize_spectrum([
        SpectrumDomain.PHYSICS,
        SpectrumDomain.PHILOSOPHY,
        SpectrumDomain.ART
    ])
    
    #      
    print("\n" + "=" * 60)
    print("         (            ):")
    for crystal in spectrum.crystals:
        domains = [d.value for d in crystal["domains"]]
        print(f"     {crystal['name']}")
        print(f"         : {', '.join(domains)}")
        print(f"        : {' + '.join(crystal['parents'])}")
    
    print("\n" + "=" * 60)
    print("       !")
    print("=" * 60)


if __name__ == "__main__":
    demo_knowledge_spectrum()
