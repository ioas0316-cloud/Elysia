"""
       (Concept Polymer)
============================

"                 " -       

     :
1.     "   "      (  ,   ,     )
2.                   (              )
3.                    (주권적 자아)
4.    =        (         )

          :
-      :           (   )
-      :                (   )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
import random

# Neural Registry -      :              
try:
    from elysia_core import Cell, Organ
except ImportError:
    # Standalone/Mock fallback
    def Cell(name): return lambda cls: cls
    class Organ:
        @staticmethod
        def get(name): return None


class Principle(Enum):
    """       -              """
    CAUSALITY = "  "        #       
    CYCLE = "  "            #   ,     
    PROBABILITY = "  "      #     ,    
    OBSERVATION = "  "      #   ,   
    ENTROPY = "    "      #       
    HARMONY = "  "          #   ,     
    EMERGENCE = "  "        #      
    TRANSFORMATION = "  "   #      
    RECURSION = "    "    #    ,       
    DUALITY = "   "        #   /  ,  / 


@dataclass
class ConceptAtom:
    """
          -               
    
              "     "    
    """
    name: str
    principles: Set[Principle]  #               
    why_chain: List[str] = field(default_factory=list)  #      
    bonded_to: List['ConceptAtom'] = field(default_factory=list)  #        
    
    def can_bond_with(self, other: 'ConceptAtom') -> Set[Principle]:
        """
                  
        
        Returns:
                     (코드 베이스 구조 로터)
        """
        return self.principles & other.principles
    
    def get_bonding_sites(self) -> Set[Principle]:
        """              """
        #                      ,
        #                             
        return self.principles
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name


@dataclass
class ConceptBond:
    """        -          """
    concept1: ConceptAtom
    concept2: ConceptAtom
    bridge_principles: Set[Principle]  #                 
    emergent_insight: str = ""  #              
    
    def strength(self) -> float:
        """      =        """
        return len(self.bridge_principles)


@Cell("ConceptPolymer")
class ConceptPolymer:
    """
           -                   
    
                       ,
                             
    """
    
    def __init__(self):
        self.atoms: Dict[str, ConceptAtom] = {}
        self.bonds: List[ConceptBond] = []
        self.polymers: List[List[ConceptAtom]] = []  #         
        
        # WhyEngine    -      : Organ.get         
        self.why_engine = Organ.get("WhyEngine")
        
        #               
        self.insight_map: Dict[frozenset, str] = {
            frozenset([Principle.CAUSALITY, Principle.PROBABILITY]): "       ",
            frozenset([Principle.CYCLE, Principle.CAUSALITY]): "       (  )",
            frozenset([Principle.HARMONY, Principle.ENTROPY]): "          ",
            frozenset([Principle.OBSERVATION, Principle.DUALITY]): "      ",
            frozenset([Principle.EMERGENCE, Principle.RECURSION]): "      ",
        }
        
        #             (WhyEngine       )
        self.keyword_to_principle = {
            "  ": Principle.CAUSALITY, "  ": Principle.CAUSALITY, "  ": Principle.CAUSALITY, "causal": Principle.CAUSALITY, "cause": Principle.CAUSALITY,
            "  ": Principle.CYCLE, "  ": Principle.CYCLE, "  ": Principle.CYCLE, "cycle": Principle.CYCLE, "repeat": Principle.CYCLE,
            "  ": Principle.PROBABILITY, "   ": Principle.PROBABILITY, "   ": Principle.PROBABILITY, "probability": Principle.PROBABILITY, "uncertain": Principle.PROBABILITY,
            "  ": Principle.OBSERVATION, "  ": Principle.OBSERVATION, "  ": Principle.OBSERVATION, "observation": Principle.OBSERVATION, "perception": Principle.OBSERVATION,
            "    ": Principle.ENTROPY, "   ": Principle.ENTROPY, "  ": Principle.ENTROPY, "entropy": Principle.ENTROPY, "disorder": Principle.ENTROPY,
            "  ": Principle.HARMONY, "  ": Principle.HARMONY, "    ": Principle.HARMONY, "harmony": Principle.HARMONY, "balance": Principle.HARMONY,
            "  ": Principle.EMERGENCE, "  ": Principle.EMERGENCE, "  ": Principle.EMERGENCE, "emergence": Principle.EMERGENCE, "whole": Principle.EMERGENCE,
            "  ": Principle.TRANSFORMATION, "  ": Principle.TRANSFORMATION, "  ": Principle.TRANSFORMATION, "transformation": Principle.TRANSFORMATION, "change": Principle.TRANSFORMATION,
            "   ": Principle.RECURSION, "  ": Principle.RECURSION, "  ": Principle.RECURSION, "fractal": Principle.RECURSION, "recursive": Principle.RECURSION, "self-reference": Principle.RECURSION,
            "  ": Principle.DUALITY, "  ": Principle.DUALITY, "  ": Principle.DUALITY, "duality": Principle.DUALITY, "wave": Principle.DUALITY,
        }
    
    def extract_principles_from_text(self, text: str, domain: str = "general") -> Set[Principle]:
        """
        WhyEngine                       
        
        Args:
            text:         (     )
            domain:     (narrative, physics, general  )
        
        Returns:
                       
        """
        extracted = set()
        
        # 1.           (  )
        text_lower = text.lower()
        for keyword, principle in self.keyword_to_principle.items():
            if keyword in text_lower:
                extracted.add(principle)
        
        # 2. WhyEngine    (   )
        if self.why_engine:
            try:
                analysis = self.why_engine.analyze("concept", text, domain)
                # underlying_principle            
                if hasattr(analysis, 'underlying_principle'):
                    for keyword, principle in self.keyword_to_principle.items():
                        if keyword in analysis.underlying_principle:
                            extracted.add(principle)
            except Exception:
                pass  # WhyEngine                
        
        return extracted
    
    def add_atom_from_text(
        self,
        name: str,
        description: str,
        domain: str = "general"
    ) -> ConceptAtom:
        """
                                       
        
              !              " "             
        """
        principles = self.extract_principles_from_text(description, domain)
        
        if not principles:
            #                 
            principles = {Principle.EMERGENCE}  #    :   
        
        atom = ConceptAtom(
            name=name,
            principles=principles,
            why_chain=description.split()[:5]  #         
        )
        self.atoms[name] = atom
        
        print(f"    '{name}'       ")
        print(f"      : {description[:50]}...")
        print(f"      : {', '.join(p.value for p in principles)}")
        return atom
    
    def add_atom(
        self,
        name: str,
        principles: List[Principle],
        why_chain: List[str] = None
    ) -> ConceptAtom:
        """        """
        atom = ConceptAtom(
            name=name,
            principles=set(principles),
            why_chain=why_chain or []
        )
        self.atoms[name] = atom
        print(f"    '{name}'    ")
        print(f"      : {', '.join(p.value for p in principles)}")
        return atom
    
    def try_bond(self, name1: str, name2: str) -> Optional[ConceptBond]:
        """
                   
        
                      !
        """
        if name1 not in self.atoms or name2 not in self.atoms:
            return None
        
        atom1 = self.atoms[name1]
        atom2 = self.atoms[name2]
        
        #         
        common_principles = atom1.can_bond_with(atom2)
        
        if not common_principles:
            print(f"  '{name1}'   '{name2}':          (     )")
            return None
        
        #   !
        print(f"  '{name1}'     '{name2}'")
        print(f"      : {', '.join(p.value for p in common_principles)}")
        
        #             
        insight = self._generate_insight(common_principles)
        if insight:
            print(f"        : {insight}")
        
        bond = ConceptBond(
            concept1=atom1,
            concept2=atom2,
            bridge_principles=common_principles,
            emergent_insight=insight
        )
        
        self.bonds.append(bond)
        atom1.bonded_to.append(atom2)
        atom2.bonded_to.append(atom1)
        
        return bond
    
    def _generate_insight(self, principles: Set[Principle]) -> str:
        """             """
        #               
        for key, insight in self.insight_map.items():
            if key <= principles:  #       
                return insight
        
        #               
        if len(principles) >= 2:
            names = sorted([p.value for p in principles])
            return f"{names[0]}  {names[1]}     "
        return ""
    
    def auto_bond_all(self) -> List[ConceptBond]:
        """
                       
        
              :                 
        """
        print("\n           (주권적 자아)...")
        
        new_bonds = []
        atom_list = list(self.atoms.values())
        
        for i, atom1 in enumerate(atom_list):
            for atom2 in atom_list[i+1:]:
                #               
                already_bonded = any(
                    (b.concept1 == atom1 and b.concept2 == atom2) or
                    (b.concept1 == atom2 and b.concept2 == atom1)
                    for b in self.bonds
                )
                
                if not already_bonded:
                    bond = self.try_bond(atom1.name, atom2.name)
                    if bond:
                        new_bonds.append(bond)
        
        if new_bonds:
            # [RECURSIVE GROWTH]
            # If a polymer is formed, it can act as a single "Super-Atom" with the union of principles
            self.find_polymers()
            for i, polymer in enumerate(self.polymers):
                if len(polymer) >= 3: # Threshold for Super-Atom
                    name = f"MetaPrinciple_{i}"
                    super_principles = set()
                    for atom in polymer:
                        super_principles.update(atom.principles)
                    
                    if name not in self.atoms:
                        print(f"  [FRACTAL] Super-Atom '{name}' emerged from complexity.")
                        self.add_atom(name, list(super_principles), [a.name for a in polymer])
        
        print(f"\n  {len(new_bonds)}           ")
        return new_bonds
    
    def find_polymers(self) -> List[List[ConceptAtom]]:
        """
               (   )   
        
        Connected components   
        """
        visited = set()
        polymers = []
        
        def dfs(atom: ConceptAtom, polymer: List):
            if atom in visited:
                return
            visited.add(atom)
            polymer.append(atom)
            for neighbor in atom.bonded_to:
                dfs(neighbor, polymer)
        
        for atom in self.atoms.values():
            if atom not in visited:
                polymer = []
                dfs(atom, polymer)
                if len(polymer) > 1:  # 2           
                    polymers.append(polymer)
        
        self.polymers = polymers
        return polymers
    
    def visualize_structure(self) -> None:
        """      """
        print("\n" + "=" * 50)
        print("           ")
        print("=" * 50)
        
        #       
        polymers = self.find_polymers()
        
        for i, polymer in enumerate(polymers, 1):
            print(f"\n      #{i} ({len(polymer)}    ):")
            
            #      
            for atom in polymer:
                connections = [a.name for a in atom.bonded_to if a in polymer]
                principles = [p.value for p in atom.principles]
                
                if connections:
                    conn_str = "     ".join(connections)
                    print(f"   [{atom.name}]     {conn_str}")
                else:
                    print(f"   [{atom.name}]")
                print(f"          : {', '.join(principles)}")
        
        #       
        in_polymer = set(atom for p in polymers for atom in p)
        isolated = [a for a in self.atoms.values() if a not in in_polymer]
        
        if isolated:
            print(f"\n         ({len(isolated)} ):")
            for atom in isolated:
                print(f"   [{atom.name}] (     )")
        
        #      
        insights = [b.emergent_insight for b in self.bonds if b.emergent_insight]
        if insights:
            print(f"\n        :")
            for insight in set(insights):
                print(f"     {insight}")


def demo_concept_polymer():
    """         """
    print("=" * 60)
    print("           ")
    print("   '                 ' -       ")
    print("=" * 60)
    
    polymer = ConceptPolymer()
    
    #          
    print("\n          :")
    
    polymer.add_atom("    ", [
        Principle.PROBABILITY,
        Principle.OBSERVATION,
        Principle.DUALITY,
        Principle.CAUSALITY
    ], why_chain=["    ", "  ", "    "])
    
    polymer.add_atom("  ", [
        Principle.CYCLE,
        Principle.CAUSALITY,
        Principle.TRANSFORMATION
    ], why_chain=["  ", "  ", "  "])
    
    polymer.add_atom("    ", [
        Principle.ENTROPY,
        Principle.PROBABILITY,
        Principle.CAUSALITY
    ], why_chain=["   ", "      "])
    
    polymer.add_atom("    ", [
        Principle.HARMONY,
        Principle.EMERGENCE,
        Principle.DUALITY
    ], why_chain=["  ", "  ", "  "])
    
    polymer.add_atom("   ", [
        Principle.RECURSION,
        Principle.EMERGENCE,
        Principle.CYCLE
    ], why_chain=["    ", "  ", "  "])
    
    polymer.add_atom("  ", [
        Principle.EMERGENCE,
        Principle.CYCLE,
        Principle.ENTROPY,
        Principle.TRANSFORMATION
    ], why_chain=["DNA", "  ", "  "])
    
    polymer.add_atom("  ", [
        Principle.OBSERVATION,
        Principle.EMERGENCE,
        Principle.RECURSION
    ], why_chain=["  ", "  ", "  "])
    
    #      
    polymer.auto_bond_all()
    
    #       
    polymer.visualize_structure()
    
    print("\n" + "=" * 60)
    print("       !")
    print("=" * 60)


if __name__ == "__main__":
    demo_concept_polymer()
