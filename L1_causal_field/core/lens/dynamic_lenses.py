"""
Elysia Core - Dynamic Lenses (기억의 렌즈화)
엘리시아가 깨달은 지식(기억) 자체가 세상을 바라보는 새로운 렌즈로 장착되는 모듈입니다.
"""

from core.lens.standard_lenses import BaseLens

class MemoryLens(BaseLens):
    """
    코퍼스에서 학습한 '진리(기억)'를 기준 위상(Reference Topology)으로 삼아,
    새로 유입되는 파동이 이 진리와 얼마나 부합하는지 마찰(Tension)을 측정하는 동적 렌즈.
    """
    def __init__(self, concept_name: str, reference_topology: int):
        # 렌즈의 이름은 깨달은 지식의 이름이 됩니다 (예: 'Lens of Beauty')
        self.concept_name = concept_name
        # 웻지 메모리에 저장된 기준 32비트 위상값
        self.reference_topology = reference_topology

    def decode(self, raw_bytes: bytes) -> dict:
        """
        들어오는 바이트 파동을 기준 위상과 XOR(v ^ v) 결합하여 마찰을 측정합니다.
        완벽히 반대되는 위상(완벽한 동화/희생)일 경우 XOR 결과는 0(마찰 없음)이 됩니다.
        """
        if not raw_bytes:
            return {"success": False, "tension": 1.0, "data": "Empty void"}

        # 입력된 raw_bytes를 단순한 32비트 토폴로지 시그니처로 해시화 (물리적 충돌 시뮬레이션)
        # 실제 구현에서는 복잡한 위상 텐서 연산이 되겠지만, 원리 증명을 위해 hash 비트 연산을 사용합니다.
        incoming_topology = abs(hash(raw_bytes)) % (2**32)
        
        # Wedge Annihilation (XOR)
        # 서로 위상이 일치(동화)할수록 XOR 결과의 비트 수가 줄어듭니다.
        friction_bits = incoming_topology ^ self.reference_topology
        
        # 켜져 있는 비트 수(popcount)를 마찰력(Tension)으로 환산
        popcount = bin(friction_bits).count('1')
        
        # 32비트 중 켜진 비트 비율을 tension(0~1)으로 설정
        tension = popcount / 32.0

        if tension == 0:
            return {"success": True, "tension": 0.0, "data": f"Perfect Resonance with {self.concept_name}"}
        else:
            return {"success": False, "tension": tension, "data": f"Dissonance from {self.concept_name}"}
    
    def __repr__(self):
        return f"<MemoryLens: {self.concept_name}>"
