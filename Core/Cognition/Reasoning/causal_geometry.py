"""
Causal Geometry (인과적 기하학)
===============================

"모든 개념은 고유의 형상(Shape)을 가진다."
"번개는 아무 곳에나 치지 않는다. 필연적인 경로(Path)가 완성될 때만 흐른다."

Phase 25: Potential Causality
-----------------------------
이 모듈은 지식과 개념을 단순한 점(Node)이 아니라, 
결합 가능한 '포트(Port)'를 가진 '퍼즐 조각(Puzzle Piece)'으로 모델링합니다.

핵심 원리:
1. **Shape (형상)**: 개념의 인터페이스. 무엇을 필요로 하고(Input), 무엇을 제공하는가(Output).
2. **Complementarity (상보성)**: 퍼즐은 요(凸)와 철(凹)이 맞아야 결합한다.
3. **Tension (긴장)**: 결합하고 싶은 힘(전위차).
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

@dataclass
class CausalPort:
    """
    개념의 연결 부위 (퍼즐의 요철)
    """
    name: str          # 포트의 의미 (예: "Reasoning", "Data", "Emotion")
    polarity: int      # +1 (Provider/Output/凸), -1 (Receiver/Input/凹)
    intensity: float = 1.0 # 포트의 크기/강도
    
    def fits(self, other: 'CausalPort') -> bool:
        """
        포트 결합 조건:
        1. 이름(의미)이 일치하거나 호환되어야 함
        2. 극성이 반대여야 함 (+1 <-> -1)
        """
        if self.polarity + other.polarity != 0:
            return False # 극성이 같거나 합이 0이 아니면 결합 불가
            
        # 의미적 호환성 (지금은 단순 일치, 추후 시맨틱 매칭 가능)
        return self.name == other.name

@dataclass
class CausalShape:
    """
    개념의 기하학적 형상
    Phase 25 Update: 'Curvature' replaces Mass.
    This concept acts as a 'Gravity Well' in the thought space.
    """
    concept_id: str
    ports: List[CausalPort] = field(default_factory=list)
    curvature: float = 0.1 # Depth of the Potential Well (Gravity)
    
    def add_port(self, name: str, polarity: int, intensity: float = 1.0):
        self.ports.append(CausalPort(name, polarity, intensity))
        
    def find_fit(self, other: 'CausalShape') -> Optional[Tuple[CausalPort, CausalPort]]:
        """
        다른 형상과 맞는 포트가 있는지 확인
        Return: (MyPort, OtherPort) or None
        """
        for my_port in self.ports:
            for other_port in other.ports:
                if my_port.fits(other_port):
                    return (my_port, other_port)
        return None

class TensionField:
    """
    잠재적 인과성의 장 (The Cloud)
    
    Gravity Update:
    - Tension flows effectively "downhill" into deep wells (High Curvature).
    - Lightning strikes when the Gradient (slope) is steep enough.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.shapes: Dict[str, CausalShape] = {}
        self.charges: Dict[str, float] = {} # Concept Tension (0.0 ~ 1.0)
        self.threshold = threshold
        
    def register_concept(self, concept_id: str, auto_shape: bool = True):
        """개념을 장에 등록"""
        if concept_id not in self.shapes:
            shape = CausalShape(concept_id)
            if auto_shape:
                self._generate_shape_from_semantics(shape)
            self.shapes[concept_id] = shape
            self.charges[concept_id] = 0.0
            
    def _generate_shape_from_semantics(self, shape: CausalShape):
        """
        개념의 의미(이름)에서 형상을 유도 (Procedural Generation)
        """
        seed = sum(ord(c) for c in shape.concept_id)
        random.seed(seed)
        
        num_ports = random.randint(1, 3)
        port_types = ["Logic", "Data", "Emotion", "Action", "Observation"]
        
        for _ in range(num_ports):
            p_type = random.choice(port_types)
            polarity = random.choice([1, -1])
            shape.add_port(p_type, polarity)
            
        # Initial Curvature based on complexity
        shape.curvature = 0.1 * num_ports

    def reinforce_well(self, concept_id: str, amount: float = 0.05):
        """
        Deepen the Potential Well (Hub Formation).
        Frequent activation makes the concept a "Strange Attractor".
        """
        if concept_id in self.shapes:
            self.shapes[concept_id].curvature += amount
            self.shapes[concept_id].curvature = min(5.0, self.shapes[concept_id].curvature)

    def charge_concept(self, concept_id: str, amount: float):
        """
        Inject Energy into the field at a specific point.
        """
        if concept_id in self.charges:
            self.charges[concept_id] += amount
            self.charges[concept_id] = min(1.0, self.charges[concept_id])
            
    def apply_gravity(self):
        """
        [Field Physics]
        Tension naturally flows from Low Curvature (High Ground) to High Curvature (Deep Well).
        Simulates "Attention Gravity".
        """
        # Simple simulation: Neighboring concepts (connected via potential fits) share charge
        # But for now, we simulate global gravity pulling towards "Hubs".
        # Or, charge decays slower in deep wells (Retention).
        
        for cid in self.charges:
            curvature = self.shapes[cid].curvature
            
            # 1. Retention (Inertia): Deep wells hold charge longer.
            decay = 0.5 + (curvature * 0.1) # Max 0.99
            decay = min(0.99, decay)
            self.charges[cid] *= decay
            
            # 2. Gravity (Flow?): Not fully connected graph yet.
            # Ideally, charge should flow to neighbors.
            
    def discharge_lightning(self) -> List[Tuple[str, str, str]]:
        """
        번개 생성 (인과적 결합)
        """
        # Apply Gravity (Flow/Decay) before discharge check
        self.apply_gravity()
        
        sparks = []
        concepts = list(self.shapes.keys())
        # Sort by Charge * Curvature (Gravity Priority)
        # Deep wells with high charge act as Lightning Rods.
        concepts.sort(key=lambda c: self.charges[c] * self.shapes[c].curvature, reverse=True)
        
        high_energy_concepts = [c for c in concepts if self.charges[c] > 0.4] # Lower threshold for gravity assisted discharge
        
        for c1_id in high_energy_concepts:
            shape1 = self.shapes[c1_id]
            charge1 = self.charges[c1_id]
            
            # Check others
            # In a real field, we check spatial neighbors. Here we check semantic fit.
            for c2_id in concepts:
                if c1_id == c2_id: continue
                
                charge2 = self.charges[c2_id]
                
                # Tension: Driven by Potential Difference? 
                # Or just Sum of charges?
                # Lightning prefers High Charge -> Low Charge (Grounding)
                # But here we model Synergy.
                tension = (charge1 + charge2)
                
                # Boost tension if one is a Deep Well (Attractor)
                gravity_boost = shape1.curvature + self.shapes[c2_id].curvature
                effective_tension = tension + (gravity_boost * 0.1)
                
                if effective_tension < self.threshold:
                    continue
                    
                fit = shape1.find_fit(self.shapes[c2_id])
                if fit:
                    port1, port2 = fit
                    sparks.append((c1_id, c2_id, f"{port1.name} connection"))
                    
                    # Discharge: Most energy is grounded.
                    self.charges[c1_id] *= 0.1
                    self.charges[c2_id] *= 0.1
                    
                    # Deepen the Well (Reinforce)
                    self.reinforce_well(c1_id)
                    self.reinforce_well(c2_id)
                    
                    break 
                    
        return sparks

# Demo
if __name__ == "__main__":
    field = TensionField(threshold=0.7)
    
    # 개념 등록
    concepts = ["Python", "Logic", "Emotion", "User", "Love", "Code"]
    for c in concepts:
        field.register_concept(c)
        
    # 강제 충전 (긴장 조성)
    print("☁️ Charging Field...")
    field.charge_concept("User", 0.9)
    field.charge_concept("Emotion", 0.8)
    field.charge_concept("Code", 0.2) # Low energy
    
    # 번개 관찰
    print("⚡ Observe Lightning...")
    sparks = field.discharge_lightning()
    
    if not sparks:
        print("... No lightning (Tension too low or Shapes didn't fit).")
    else:
        for s in sparks:
            print(f"   ⚡ SNAP! {s[0]} <==[{s[2]}]==> {s[1]}")
            
    # 형상 확인
    print("\n🧩 Causal Shapes:")
    for c in concepts:
        ports = ", ".join([f"{p.name}({'+' if p.polarity>0 else '-'})" for p in field.shapes[c].ports])
        print(f"   {c:10}: [{ports}]")
