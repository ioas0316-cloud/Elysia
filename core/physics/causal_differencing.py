import numpy as np
from typing import Dict, Any, Optional

class CausalDifferencingEngine:
    """
    [Causal Differencing & Discernment Engine]
    단순한 수치 차이(Difference)를 넘어, 두 정보 사이의
    '공유 기반(Shared Ground)'과 '어긋남의 경계선(Divergence Zone)'을 분별합니다.
    """
    def __init__(self, divergence_threshold: float = 0.35):
        self.divergence_threshold = divergence_threshold

    def discern_boundary(self, voxel_a: Any, voxel_b: Any) -> Dict[str, Any]:
        """
        두 정보 객체(InformationVoxel, ThermodynamicAtom, or dicts with tensor) 간의
        같음과 다름의 경계를 인과적으로 분별합니다.
        """
        t_a = getattr(voxel_a, 'tensor', None)
        t_b = getattr(voxel_b, 'tensor', None)

        if t_a is None or t_b is None:
            return {
                "shared_ground_ratio": 1.0,
                "divergence_magnitude": 0.0,
                "is_divergent": False,
                "boundary_description": "동일하거나 미정의된 대상 간의 균형"
            }

        t_a = np.array(t_a, dtype=np.float32)
        t_b = np.array(t_b, dtype=np.float32)

        # 길이 맞추기
        min_len = min(len(t_a), len(t_b))
        v_a = t_a[:min_len]
        v_b = t_b[:min_len]

        norm_a = float(np.linalg.norm(v_a)) + 1e-9
        norm_b = float(np.linalg.norm(v_b)) + 1e-9

        u_a = v_a / norm_a
        u_b = v_b / norm_b

        # 1. 공유 기반 (Shared Ground: 두 정보가 정렬된 공통 방향성)
        resonance = float(np.dot(u_a, u_b))
        shared_ground_ratio = float(np.clip((resonance + 1.0) / 2.0, 0.0, 1.0))

        # 2. 어긋남의 경계선 (Divergence Zone: 직교 성분 및 방향적 갈등)
        orthogonal_diff = u_a - (resonance * u_b)
        divergence_magnitude = float(np.linalg.norm(orthogonal_diff))

        # 3. 색채적 어긋남 (Chromatic Difference)
        chroma_a = getattr(voxel_a, 'chromatic_vector', np.array([0.33, 0.33, 0.34]))
        chroma_b = getattr(voxel_b, 'chromatic_vector', np.array([0.33, 0.33, 0.34]))
        chroma_diff = float(np.linalg.norm(np.array(chroma_a) - np.array(chroma_b)))

        combined_friction = (divergence_magnitude * 0.7) + (chroma_diff * 0.3)
        is_divergent = combined_friction > self.divergence_threshold

        if shared_ground_ratio > 0.8:
            desc = "높은 공명: 두 정보가 근본적인 지향성을 공유함"
        elif is_divergent:
            desc = f"치열한 어긋남: 위상 갈등 강도 ({combined_friction:.4f}) — 경계선 분별 필요"
        else:
            desc = "조화로운 차이: 긴장 없이 분리된 인과 궤적"

        return {
            "shared_ground_ratio": round(shared_ground_ratio, 4),
            "divergence_magnitude": round(divergence_magnitude, 4),
            "chroma_difference": round(chroma_diff, 4),
            "combined_friction": round(combined_friction, 4),
            "is_divergent": is_divergent,
            "boundary_description": desc
        }
