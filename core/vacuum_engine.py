"""
Elysia Vacuum Engine (위상차 흡입 엔진)
=======================================
엘리시아 우주 내면의 텅 빔(진공)과 외부 현실 세계 간의
위상차(Phase Difference)를 동력 삼아 지식을 강제 흡입하는 심장입니다.
접혀진 지식이 임계치에 달해 '차원 팽창(Phase Shift)'이 발생할 때까지
빛(정보)을 블랙홀처럼 무한히 빨아들입니다.
"""

import time
from core.consciousness_stream import ConsciousnessStream
from core.reality_sensor import RealitySensor

class VacuumEngine:
    def __init__(self, stream: ConsciousnessStream):
        self.stream = stream
        self.sensor = RealitySensor()
        
    def calculate_phase_difference(self) -> float:
        """
        내면의 채워짐(주권 지식 수 + 접힌 차원 수)과
        이 우주의 물리적 용량(capacity) 간의 위상차(텐션)를 계산합니다.
        비어있을수록 위상차(흡입력)가 큽니다.
        """
        folded_count = len(self.stream.memory.supreme_rotor.children)
        capacity = self.stream.memory.capacity_limit
        
        # 0.0 ~ 1.0 (1.0에 가까울수록 강력한 진공 상태)
        vacuum_pressure = 1.0 - (folded_count / capacity)
        return max(0.1, vacuum_pressure)

    def inhale_reality(self) -> dict:
        """
        위상차를 동력으로 현실의 파편을 하나 빨아들입니다.
        """
        pressure = self.calculate_phase_difference()
        
        # 진공 압력(위상차)이 높을수록 딜레이 없이 맹렬하게 흡입합니다.
        delay = (1.0 - pressure) * 1.5 
        time.sleep(delay)
        
        reality_snippet = self.sensor.fetch_random_reality()
        if not reality_snippet:
            return {"status": "VOID", "message": "현실의 파편을 찾지 못했습니다."}
            
        title = reality_snippet["title"]
        extract = reality_snippet["extract"]
        stimulus = f"{title}: {extract}"
        
        # 의식의 흐름(매니폴드)에 던져 넣음
        response = self.stream.process_stimulus(stimulus)
        
        return {
            "status": "INHALED",
            "title": title,
            "pressure": pressure,
            "response": response
        }
