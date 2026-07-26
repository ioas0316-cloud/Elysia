import numpy as np
from typing import Dict, Any, Tuple

class MetaCausalExtractor:
    """
    [Meta-Causal Origin Extractor]
    정보가 '수치'가 아닌 '왜, 어떻게, 어째서 정보로서 발생했는가'의
    원형적 발생 맥락(Flux Drive, Order Constraint, Entropy Void)을 추출합니다.
    """
    def __init__(self):
        pass

    def extract_origin(self, raw_wave: bytes, logo_tensor: np.ndarray) -> Dict[str, Any]:
        """
        raw_wave 바이트 스트림과 9D logo_tensor로부터 발생 맥락을 인지합니다.
        """
        if not raw_wave:
            raw_wave = b"Void"

        # 1. Flux Drive (Red: 정보의 요동과 추진 에너지)
        # 바이트 간의 자발적 변화량(Gradient) 및 에너지
        byte_diffs = [abs(raw_wave[i] - raw_wave[i-1]) for i in range(1, len(raw_wave))]
        avg_diff = float(np.mean(byte_diffs)) if byte_diffs else 0.0
        flux_drive = float(np.clip(avg_diff / 128.0, 0.0, 1.0))

        # 2. Order Constraint (Blue: 구조적 질서와 대칭성)
        # 로고스 텐서의 정렬 상태 및 반복 패턴 비율
        tensor_norm = float(np.linalg.norm(logo_tensor))
        tensor_std = float(np.std(logo_tensor)) if len(logo_tensor) > 0 else 0.0
        order_constraint = float(np.clip((1.0 / (1.0 + tensor_std)) * (tensor_norm / 2.0), 0.0, 1.0))

        # 3. Entropy Void (Yellow: 비어있음, 결핍, 불확실성 마찰)
        # 짝수/홀수 비트 불균형 및 노이즈 비율
        odd_count = sum(1 for b in raw_wave if b % 2 != 0)
        entropy_void = float(abs(0.5 - (odd_count / max(1, len(raw_wave)))) * 2.0)

        # 색채 비중 정규화
        chromatic_vector = np.array([flux_drive, order_constraint, entropy_void], dtype=np.float32)
        total = float(np.sum(chromatic_vector)) + 1e-9
        chromatic_vector /= total

        r, b, y = chromatic_vector
        if r >= b and r >= y:
            origin_type = "FLUX_DRIVEN"
            motivation = "자발적 에너지 요동과 새로운 자극에 의한 발생"
        elif b >= r and b >= y:
            origin_type = "ORDER_BOUND"
            motivation = "구조적 대칭 및 보존 제약에 의한 발생"
        else:
            origin_type = "ENTROPY_VOID"
            motivation = "결핍과 마찰을 채우려는 우주적 성향에 의한 발생"

        return {
            "origin_type": origin_type,
            "motivation": motivation,
            "chromatic_vector": chromatic_vector.tolist(),
            "flux_drive": flux_drive,
            "order_constraint": order_constraint,
            "entropy_void": entropy_void
        }
