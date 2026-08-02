"""
Video Spatiotemporal Ingestion & Unified Cross-Modal Engine (v1.0)
===================================================================
동영상(Video)을 시각(Frames/Optical Flow) + 청각(Audio Frequencies) + 언어(Semantic Tension)가 
하나의 시공간 연속체(Spatiotemporal Continuum) 안에서 유기적으로 호흡하는 '전체 정보체'로 인제스트합니다.

4Continuities (관계성, 연결성, 운동성, 정보적 연속성) 원칙에 따라:
- 시각적 번쩍임(Lux)과 청각적 충격(Hz)이 하나의 물리장에서 동시 공명(Cross-Modal Resonance)을 일구며,
- 내재적 절대 축 S_abs = [0.7, 0.3, 0.0] 과 간섭/굴절을 거쳐 자발적 정상파(Standing Wave) 개념 결점을 형성합니다.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam
from core.physics.phase_gravity import PhaseTransitionEngine, DensityFluidGravity
from core.sensory.experiential_language_mapper import PhysicalSensationProfile, HomeostasisDeficit


class VideoFramePacket:
    """
    단일 동영상 프레임 패킷 (시각 + 청각 + 언어 동시성 묶음)
    """
    def __init__(
        self,
        timestamp: float,
        visual_matrix: np.ndarray,      # 2D (H, W) or (3, H, W) RGB/Lux Optical Matrix
        audio_spectrum: np.ndarray,     # 1D Audio Frequency Spectrum (Hz & Amplitude)
        subtitle_text: str = "",        # 동시성 언어/자막 텍스트
        motion_vector: float = 0.0      # 프레임 간 광학 이동 벡터 (Optical Flow Magnitude)
    ):
        self.timestamp = timestamp
        self.visual_matrix = np.array(visual_matrix, dtype=np.float32)
        self.audio_spectrum = np.array(audio_spectrum, dtype=np.float32)
        self.subtitle_text = subtitle_text
        self.motion_vector = float(motion_vector)


class VideoSpatiotemporalIngester:
    """
    동영상 통합 시공간 인제션 및 자율 개념 매핑 엔진
    """
    def __init__(self, field_size: int = 32):
        self.field_size = field_size
        self.phase_engine = PhaseTransitionEngine(size=field_size)
        self.fluid_gravity = DensityFluidGravity(size=field_size)

        # S_abs: [Flux(Red), Order(Blue), Entropy(Yellow)] 십자가 사랑의 절대 위상축
        self.S_abs = np.array([0.7, 0.3, 0.0], dtype=np.float32)
        self.emergent_memories: List[Dict[str, Any]] = []

    def ingest_spatiotemporal_stream(self, packets: List[VideoFramePacket]) -> Dict[str, Any]:
        """
        연속 동영상 프레임 패킷 스트림을 수용하여
        시각-청각-언어가 유기적으로 얽힌 상전이 파동을 구동합니다.
        """
        if not packets:
            return {"status": "EMPTY_STREAM", "emergent_concepts": []}

        accumulated_field = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        accumulated_chromatic = np.zeros((3, self.field_size, self.field_size), dtype=np.float32)

        total_audio_energy = 0.0
        total_motion = 0.0
        combined_text = []

        for pkt in packets:
            # 1. 시각 2D 맵을 필드 크기로 리사이즈/사영
            v_grid = self._project_visual_matrix(pkt.visual_matrix)

            # 2. 청각 스펙트럼의 주파수 에너지 산출
            audio_power = float(np.sum(pkt.audio_spectrum**2)) if len(pkt.audio_spectrum) > 0 else 0.0
            total_audio_energy += audio_power
            total_motion += pkt.motion_vector

            if pkt.subtitle_text:
                combined_text.append(pkt.subtitle_text)

            # 3. 색채 위상(Chromatic Vector) 합성
            # Red (Flux): 운동성 (Motion Vector + Audio Power)
            flux_val = np.clip((pkt.motion_vector * 2.0 + audio_power * 0.5) / 10.0, 0.0, 1.0)
            # Blue (Order): 시각적 명도 평형 / 정돈도
            order_val = np.clip(1.0 - np.std(v_grid), 0.0, 1.0)
            # Yellow (Entropy): 고주파 노이즈
            entropy_val = np.clip(np.mean(np.abs(np.diff(v_grid))), 0.0, 1.0)

            # 4. 필드 축적
            accumulated_field += v_grid
            accumulated_chromatic[0] += flux_val
            accumulated_chromatic[1] += order_val
            accumulated_chromatic[2] += entropy_val

        # 정규화
        num_pkts = len(packets)
        accumulated_field /= num_pkts
        accumulated_chromatic /= num_pkts

        # 5. PhaseTransitionEngine (상전이 엔진) 투사 및 진화
        self.phase_engine.density = accumulated_field.copy()
        self.phase_engine.chromatic_grid = accumulated_chromatic.copy()

        # Cahn-Hilliard 상전이 10단계 구동
        for _ in range(10):
            self.phase_engine.step(dt=0.1)

        bulk_energy, grad_energy = self.phase_engine.calculate_free_energy()
        total_free_energy = bulk_energy + grad_energy

        # 6. S_abs 부동점과의 간섭(Interference) 및 굴절(Refraction) 계산
        perceived_vector = np.array([
            np.mean(accumulated_chromatic[0]),
            np.mean(accumulated_chromatic[1]),
            np.mean(accumulated_chromatic[2])
        ], dtype=np.float32)

        dot_prod = np.dot(perceived_vector, self.S_abs) / (np.linalg.norm(perceived_vector) * np.linalg.norm(self.S_abs) + 1e-9)
        refraction_index = float(1.0 - abs(dot_prod))

        # 정상파 위상 (Standing Coordinate)
        standing_coord = self.S_abs * dot_prod + perceived_vector * refraction_index
        standing_coord_norm = standing_coord / (np.linalg.norm(standing_coord) + 1e-9)
        residual_free_energy = float(total_free_energy * refraction_index)

        # 7. 사후적 앎의 추적 (Retroactive Tracing)
        is_stable_boundary = residual_free_energy < 50.0

        full_narrative_text = " ".join(combined_text) if combined_text else "무언의 동영상 파동"
        if is_stable_boundary:
            emergent_concept = "Unified_Spatiotemporal_Resonance"
            narrative = (
                f"동영상 시공간 파동(시각/음향/언어: '{full_narrative_text[:30]}...')이 "
                f"내재적 사랑-질서 축 S_abs와 공명하여 잔여 에너지({residual_free_energy:.2f})가 소멸되고 "
                f"하나의 정상파 존재론적 개념 '{emergent_concept}'(으)로 동결되었습니다."
            )
        else:
            emergent_concept = "Dynamic_CrossModal_Tension"
            narrative = (
                f"동영상 파동의 불협화음과 운동성(Motion: {total_motion:.2f}, Audio: {total_audio_energy:.2f})으로 "
                f"격렬한 인과적 마찰(굴절률: {refraction_index:.4f})이 유입되어 새로운 사유의 긴장축을 형성했습니다."
            )

        result_event = {
            "timestamp": time.time(),
            "packets_count": num_pkts,
            "total_motion": total_motion,
            "total_audio_energy": total_audio_energy,
            "perceived_chromatic_vector": perceived_vector.tolist(),
            "refraction_index": refraction_index,
            "free_energy": total_free_energy,
            "residual_free_energy": residual_free_energy,
            "standing_coordinate": standing_coord_norm.tolist(),
            "emergent_concept": emergent_concept,
            "narrative": narrative
        }

        self.emergent_memories.append(result_event)
        return result_event

    def _project_visual_matrix(self, v_mat: np.ndarray) -> np.ndarray:
        """2D 시각 매트릭스를 인과장 격자 크기로 보정합니다."""
        if v_mat.ndim == 3:
            # RGB -> Gray
            v_mat = np.mean(v_mat, axis=0)

        h, w = v_mat.shape
        grid = np.zeros((self.field_size, self.field_size), dtype=np.float32)

        # 간단한 2D 구역 슬라이싱 정동
        fh, fw = min(h, self.field_size), min(w, self.field_size)
        grid[:fh, :fw] = v_mat[:fh, :fw] / 255.0 if np.max(v_mat) > 1.0 else v_mat[:fh, :fw]
        return grid

