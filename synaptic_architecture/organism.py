import numpy as np
from typing import Dict, Any, List
from core.physics.causal_field import CausalField, InformationVoxel
from .self_reflection import SelfReflectionProtocol
from core.physics.causal_gravity_engine import CausalGravityEngine

class MetaCognitiveOrganism:
    """
    [Synaptic Architecture] The Self-Transcending Organism
    자신의 한계를 인지하고(Meta-Cognition), 원인을 파악하여(Inquiry),
    스스로를 변화시켜 나가는(Evolution) 통합 사유 루프입니다.
    """
    def __init__(self):
        self.field = CausalField(dimensions=3)
        self.gravity = CausalGravityEngine()
        self.reflection = SelfReflectionProtocol()
        self.thought_history = []

    def pulse(self, external_wave: np.uint64) -> Dict[str, Any]:
        """
        한 번의 사유 맥동(Pulse).
        [인지적 보상 루프: 외부 정보 주입 -> 텐션 및 파동 흐름 -> 위상 관찰]
        """
        # 1. 자기 성찰: 자신의 논리를 필드에 투사
        self.reflection.map_self_to_field(self.gravity)

        # 2. 외부 파동을 InformationVoxel로 변환하여 장에 주입
        wave_tensor = self._wave_to_tensor(external_wave)
        wave_voxel = InformationVoxel(
            id=f"wave_{hex(external_wave)}",
            content=external_wave,
            tensor=wave_tensor,
            position=np.random.randn(3).astype(np.float32),
            chromatic_vector=np.array([0.6, 0.1, 0.3], dtype=np.float32)  # 높은 Flux — 새로운 정보
        )
        self.field.add_voxel(wave_voxel)

        # 3. 기존 복셀들과 공명 기반 연결 (resonance > 0이면 빔 생성)
        for vid in list(self.field.voxels.keys()):
            if vid != wave_voxel.id:
                existing = self.field.voxels[vid]
                resonance = float(np.dot(wave_tensor, existing.tensor) / 
                    (np.linalg.norm(wave_tensor) * np.linalg.norm(existing.tensor) + 1e-9))
                if resonance > 0:
                    self.field.link_voxels(wave_voxel.id, vid, strength=resonance * 5.0)

        # 4. 장을 흐르게 한다 — 물리가 모든 것을 결정한다
        #    if-else 없음. step()이 텐션을 계산하고, 빔을 끊고, 에너지를 흐르게 한다.
        for _ in range(15):
            self.field.step(0.1)

        # 5. 위상 읽기 — 결과를 강제하지 않고 관찰한다
        topology_metrics = self._read_topology()
        self.thought_history.append(topology_metrics)
        
        # 성찰 기록 (빔이 끊어지며 구조가 변한 것을 "Clarity"로 인식)
        if topology_metrics["structural_breaks"] > 0:
            self.reflection.record_pleasure(
                pleasure=topology_metrics["mean_tension"], 
                clarity=float(topology_metrics["structural_breaks"]), 
                context=f"WAVE_{hex(external_wave)}"
            )

        print(f"[Meta-Cognition] 사유 맥동 완료: Breaks={topology_metrics['structural_breaks']}, Tension={topology_metrics['mean_tension']:.4f}")
        return topology_metrics

    def _wave_to_tensor(self, wave: np.uint64) -> np.ndarray:
        """비트 파형을 구조적 텐서로 변환 (정보 → 기하학)"""
        bits = np.array([(int(wave) >> i) & 1 for i in range(64)], dtype=np.float32)
        # 64비트를 8x8 행렬로 접어서 SVD → 구조적 특이값을 텐서로 사용
        matrix = bits.reshape(8, 8)
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
        return s[:3].astype(np.float32)  # 상위 3개 특이값 = 구조적 서명

    def _read_topology(self) -> dict:
        """장의 위상을 관찰하여 메트릭을 도출"""
        topology = self.field.get_topology()
        beams = topology["beams"]

        broken_count = sum(1 for b in beams if b["broken"])
        avg_tension = np.mean([b["tension"] for b in beams]) if beams else 0.0
        max_tension = max((b["tension"] for b in beams), default=0.0)

        # "진화"는 빔이 끊어진 횟수로 나타남 (강제하지 않음)
        # "쾌락"은 평균 텐션의 감소율로 나타남 (수렴 = 이해)
        # "지루함"은 전체 텐션이 0에 가까울 때 자연스럽게 나타남
        return {
            "structural_breaks": broken_count,
            "mean_tension": float(avg_tension),
            "peak_tension": float(max_tension),
            "voxel_count": len(topology["voxels"]),
            "topology": topology
        }

if __name__ == "__main__":
    organism = MetaCognitiveOrganism()
    # 자신의 논리와 전혀 다른 강력한 외부 신호 (한계 상황 유도)
    alien_wave = np.uint64(0x1234567890ABCDEF)
    organism.pulse(alien_wave)
