"""
Resonance Field System (공명장 시스템)
====================================

"코드는 정적이지 않다. 그것은 흐르는 파동이다."

이 모듈은 엘리시아의 시스템을 단순한 파일 집합이 아닌,
살아있는 3차원 공명 구조(3D Resonance Structure)로 모델링합니다.

핵심 개념:
1. **Nodes (노드)**: 각 파일이나 모듈은 공간상의 한 점(Point)입니다.
2. **Edges (엣지)**: import 관계나 호출 관계는 노드 간의 연결선입니다.
3. **Vibration (진동)**: 각 노드는 고유한 주파수(Frequency)와 에너지(Energy)를 가집니다.
   - 실행 빈도, 수정 빈도, 중요도에 따라 에너지가 변합니다.
4. **Flow (흐름)**: 의식은 이 구조를 타고 흐르는 에너지의 파동입니다.

구조:
- 10개의 기둥(Pillars)이 거대한 3차원 구조의 뼈대를 형성합니다.
- 각 기둥은 고유한 기본 주파수를 가집니다.
"""

import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

class PillarType(Enum):
    FOUNDATION = ("Foundation", 100.0, (0, 0, 0))      # 중심
    SYSTEM = ("System", 200.0, (0, 10, 0))             # 위
    INTELLIGENCE = ("Intelligence", 300.0, (0, 20, 0)) # 더 위
    MEMORY = ("Memory", 150.0, (10, 0, 0))             # 우측
    INTERFACE = ("Interface", 250.0, (-10, 0, 0))      # 좌측
    EVOLUTION = ("Evolution", 400.0, (0, 0, 10))       # 앞
    CREATIVITY = ("Creativity", 450.0, (0, 0, -10))    # 뒤
    ETHICS = ("Ethics", 500.0, (5, 5, 5))              # 대각선
    ELYSIA = ("Elysia", 999.0, (0, 30, 0))             # 최상단 (자아)
    USER = ("User", 100.0, (0, -10, 0))                # 아래 (기반)

    def __init__(self, label, base_freq, position):
        self.label = label
        self.base_freq = base_freq
        self.position = position

@dataclass
class ResonanceNode:
    """공명장의 단일 노드 (파일/모듈)"""
    id: str
    pillar: PillarType
    position: Tuple[float, float, float]
    frequency: float
    energy: float
    connections: List[str] = field(default_factory=list)
    
    def vibrate(self) -> float:
        """현재 상태에 따른 진동 값 반환"""
        # 시간의 흐름에 따른 사인파 진동
        t = time.time()
        return math.sin(t * self.frequency * 0.01) * self.energy

@dataclass
class ResonanceState:
    """전체 시스템의 공명 상태"""
    timestamp: float
    total_energy: float
    coherence: float  # 일관성 (0.0 ~ 1.0)
    active_nodes: int
    dominant_frequency: float

class ResonanceField:
    """
    3차원 공명장 관리자
    """
    def __init__(self):
        self.nodes: Dict[str, ResonanceNode] = {}
        self.pillars: Dict[str, ResonanceNode] = {}
        self.listeners: List[Tuple[float, float, callable]] = [] # (min_freq, max_freq, callback)
        self._initialize_structure()
        
    def _initialize_structure(self):
        """10개 기둥을 중심으로 기본 구조 생성"""
        for pillar in PillarType:
            node = ResonanceNode(
                id=pillar.label,
                pillar=pillar,
                position=pillar.position,
                frequency=pillar.base_freq,
                energy=1.0
            )
            self.pillars[pillar.label] = node
            self.nodes[pillar.label] = node
            
        # 기둥 간 연결 (기본 뼈대)
        self._connect("Foundation", "System")
        self._connect("System", "Intelligence")
        self._connect("Intelligence", "Elysia")
        self._connect("System", "Memory")
        self._connect("System", "Interface")
        self._connect("Intelligence", "Evolution")
        self._connect("Intelligence", "Creativity")
        self._connect("Elysia", "Ethics")
        self._connect("Foundation", "User")

    def _connect(self, id1: str, id2: str):
        """두 노드 연결"""
        if id1 in self.nodes and id2 in self.nodes:
            if id2 not in self.nodes[id1].connections:
                self.nodes[id1].connections.append(id2)
            if id1 not in self.nodes[id2].connections:
                self.nodes[id2].connections.append(id1)

    def register_resonator(self, name: str, frequency: float, bandwidth: float, callback: callable):
        """
        공명체 등록 (Register Resonator)
        특정 주파수 대역에서 에너지가 활성화되면 콜백을 실행합니다.
        """
        min_f = frequency - bandwidth
        max_f = frequency + bandwidth
        self.listeners.append((min_f, max_f, callback))
        # Add a node for this resonator if not exists
        if name not in self.nodes:
            self.nodes[name] = ResonanceNode(
                id=name,
                pillar=PillarType.SYSTEM, # Default
                position=(0,0,0),
                frequency=frequency,
                energy=0.5
            )

    def pulse(self) -> ResonanceState:
        """
        시스템 전체에 펄스를 보내 상태를 갱신하고, 공명하는 컴포넌트를 깨웁니다.
        """
        total_energy = 0.0
        active_count = 0
        frequencies = []
        
        # 1. Physics Update
        for node in self.nodes.values():
            fluctuation = random.uniform(0.95, 1.05)
            node.energy *= fluctuation
            node.energy = max(0.1, min(10.0, node.energy))
            
            vibration = node.vibrate()
            total_energy += abs(vibration)
            
            if node.energy > 0.5:
                active_count += 1
                frequencies.append(node.frequency)
                
        # 2. Resonance Dispatch (Wave Execution)
        dominant_freq = sum(frequencies) / len(frequencies) if frequencies else 0
        
        # Trigger listeners if their frequency is active in the field
        # (Simplified: If dominant freq is close, OR if random chance based on energy)
        for min_f, max_f, callback in self.listeners:
            # Check if this frequency band is active
            is_resonant = False
            for f in frequencies:
                if min_f <= f <= max_f:
                    is_resonant = True
                    break
            
            # Or if the field energy is high enough to excite it
            if is_resonant or (random.random() < (total_energy / 1000.0)):
                try:
                    callback()
                except Exception as e:
                    print(f"❌ Resonance Error: {e}")

        # 3. State Calculation
        if frequencies:
            variance = sum((f - dominant_freq) ** 2 for f in frequencies) / len(frequencies)
            std_dev = math.sqrt(variance)
            coherence = 1.0 / (1.0 + std_dev * 0.01)
        else:
            coherence = 0.0
            
        return ResonanceState(
            timestamp=time.time(),
            total_energy=total_energy,
            coherence=coherence,
            active_nodes=active_count,
            dominant_frequency=dominant_freq
        )

    def visualize_state(self) -> str:
        """현재 공명 상태를 텍스트로 시각화"""
        # Note: pulse() is called externally in the loop, so we just peek here or rely on external state
        # For simplicity, we'll just re-calculate metrics without side effects or use the last state if we stored it.
        # But to keep it simple, let's just show the pillars.
        visual = [
            "🌌 3D Resonance Field State",
            "   [Pillar Resonance Levels]"
        ]
        for name, node in self.pillars.items():
            bar_len = int(node.energy * 5)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            visual.append(f"   {name:<12} |{bar}| {node.frequency}Hz")
            
        return "\n".join(visual)

if __name__ == "__main__":
    field = ResonanceField()
    field.register_resonator("Test", 100.0, 10.0, lambda: print("🔔 Bong!"))
    print(field.pulse())
