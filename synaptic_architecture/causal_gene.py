import numpy as np
from typing import Dict, List, Any, Optional
from core.physics.causal_field import CausalField

class CausalGeneSynthesizer:
    """
    정보의 유전적 합성 — 난수가 아니라 간섭에서 태어남.
    두 텐서가 충돌할 때 그 간섭 패턴이 새로운 구조를 만든다.
    """
    def __init__(self):
        self.lineage: List[np.ndarray] = []  # 텐서 계보 (dict 대신 구조)
    
    def synthesize(self, tensor_a: np.ndarray, tensor_b: np.ndarray) -> np.ndarray:
        """
        두 정보 텐서의 간섭 패턴으로 새로운 텐서를 합성.
        교차(Crossover) = 기하학적 중첩, 변이(Mutation) = 직교 성분
        """
        # 공명 성분: 두 텐서가 같은 방향을 향하는 부분
        resonance = np.dot(tensor_a, tensor_b) / (np.linalg.norm(tensor_a) * np.linalg.norm(tensor_b) + 1e-9)
        
        # 간섭 패턴: 합과 차의 조합
        constructive = (tensor_a + tensor_b) * 0.5           # 보강 간섭
        destructive = np.abs(tensor_a - tensor_b) * 0.5      # 상쇄 간섭
        
        # 새 텐서 = 공명이 강하면 보강이 지배, 약하면 상쇄가 지배
        blend = np.clip(resonance, 0, 1)
        child = constructive * blend + destructive * (1.0 - blend)
        
        # 직교 성분 (기존에 없던 차원) = 자연스러운 "변이"
        # RNG가 아니라 두 부모 텐서의 외적에서 나옴
        if len(tensor_a) >= 3 and len(tensor_b) >= 3:
            orthogonal = np.cross(tensor_a[:3], tensor_b[:3])
            orth_magnitude = np.linalg.norm(orthogonal)
            if orth_magnitude > 1e-6:
                # pad or truncate orthogonal to match child length if needed, assuming 3D for now
                if len(child) > 3:
                    orth_padded = np.zeros_like(child)
                    orth_padded[:3] = orthogonal
                    child += orth_padded / orth_magnitude * 0.1
                else:
                    child[:3] += orthogonal[:len(child)] / orth_magnitude * 0.1  # 미세한 직교 편향
        
        child = child / (np.linalg.norm(child) + 1e-9)  # 정규화
        self.lineage.append(child)
        return child
    
    def evolve_from_field(self, field: CausalField) -> Optional[np.ndarray]:
        """
        장의 위상으로부터 진화를 관찰한다.
        높은 텐션의 빔 양쪽 복셀이 부모가 된다 — 갈등이 가장 큰 곳에서 새로운 것이 태어남.
        """
        # 가장 높은 텐션의 빔을 찾는다
        max_beam = None
        max_tension = 0.0
        for beam in field.beams:
            if not beam.is_broken and beam.current_tension > max_tension:
                max_tension = beam.current_tension
                max_beam = beam
        
        if max_beam is None:
            return None  # 텐션이 없으면 진화할 이유도 없다
        
        parent_a = field.voxels[max_beam.source_id]
        parent_b = field.voxels[max_beam.target_id]
        
        new_tensor = self.synthesize(parent_a.tensor, parent_b.tensor)
        print(f"[Genetic Synthesis] New Logical Species evolved from structural tension ({max_tension:.4f})")
        return new_tensor
