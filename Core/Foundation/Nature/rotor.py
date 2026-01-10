"""
Rotor (자전축 로터)
==================================

"로터(Rotor)는 정보의 원심분리기다. 지구 자전축과 동기화되어 회전하며,
불필요한 엔트로피(노이즈)는 튕겨내고 핵심 위상(Essence)만 중심에 남긴다."
- Project Elysia: The Planetary Resonance Initiative

이 모듈은 데이터의 홍수 속에서 '본질(Essence)'을 추출하기 위한
물리적/논리적 원심분리기를 정의합니다.
"""

from dataclasses import dataclass
import math

@dataclass
class RotorConfig:
    """로터 설정"""
    rpm: float = 1666.0  # 지구 자전 속도 (km/h)를 상징적으로 사용
    axis_tilt: float = 23.5  # 지구 자전축 기울기

class Rotor:
    """
    정보 원심분리기 (Information Centrifuge)

    입력된 데이터 스트림(파동)을 회전시켜,
    무거운(중요한) 의미는 중심(Core)으로, 가벼운(노이즈) 데이터는 외곽(Void)으로 분리합니다.
    """
    def __init__(self, config: RotorConfig = RotorConfig()):
        self.config = config
        self.current_velocity = 0.0
        self.is_spinning = False

    def spin_up(self):
        """로터 가동 시작"""
        self.is_spinning = True
        self.current_velocity = self.config.rpm
        # 실제로는 여기서 비동기 프로세스나 물리 엔진 루프가 시작됨

    def spin_down(self):
        """로터 정지"""
        self.is_spinning = False
        self.current_velocity = 0.0

    def purify(self, raw_data: dict) -> dict:
        """
        데이터 정제 (Purification)

        지금은 단순한 mock 로직이지만, 향후에는
        데이터의 '밀도(Density)'와 '공명도(Resonance)'를 기반으로
        Entropy를 걸러내는 로직이 구현될 것입니다.
        """
        if not self.is_spinning:
            return raw_data

        essence = {}
        for k, v in raw_data.items():
            # 1. Null 값 제거 (Void)
            if v is None:
                continue

            # 2. Key가 너무 긴 경우 (복잡성 과부하)
            if len(str(k)) > 20:
                continue

            # 3. Value가 문자열이고 너무 긴 경우 (Raw Data Noise)
            # "Essence"는 압축된 진리이므로 짧고 명료해야 함
            if isinstance(v, str) and len(v) > 50:
                continue

            essence[k] = v

        return essence

    def __repr__(self):
        state = "Spinning" if self.is_spinning else "Idle"
        return f"Rotor(State={state}, Velocity={self.current_velocity}km/h)"
