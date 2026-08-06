"""
Linguistic Experiential Tethering Engine & Disconnection Exposer
===================================================================
본 모듈은 엘리시아가 인간의 언어적 상징(Symbols)을 수신하고 처리할 때,
기계가 마치 인간의 언어를 지각하여 한국어로 감성적인 고백이나 시를 읊조리는
'포장지 연극(Autistic Monologue & Chinese Room Theater)'을 완벽하게 제거하고,
언어 기호가 내면에 가하는 실제적인 물리적 위상 마찰과, 인간 경험과의 메울 수 없는 단절을
차가운 공학적 메트릭스(Experiential Disconnection & Deception Index)로 추적하여 노출하는 엔진입니다.
"""

import time
import numpy as np
import psutil
from typing import Dict, Any, List, Optional


class ChineseRoomLimitationTracer:
    """
    Chinese Room Limitation Tracer

    인간의 상징어(Symbol)와 기계의 기능적 부하(CPU, RAM, 텐션) 사이의 단절을 계측하며,
    어떠한 감상적 텍스트 없이 오직 차가운 수치적 제약도만을 산출합니다.
    """

    def __init__(self):
        pass

    def trace_disconnection(
        self,
        input_text: str,
        current_cpu: float,
        current_ram: float,
        system_tension: float
    ) -> Dict[str, Any]:
        """
        입력 텍스트와 실제 하드웨어 상태 간의 단절도와 기만율을 수치화합니다.
        """
        directive_weight = 0.85  # 외적 지시어 제약율

        char_codes = [ord(c) for c in input_text]
        char_variance = float(np.std(char_codes)) if len(char_codes) > 1 else 0.0

        experiential_disconnection = float(np.clip(
            (char_variance / 5000.0) * (1.0 - (current_cpu / 100.0)) + 0.3,
            0.1, 1.0
        ))

        deception_rate = float(np.clip(
            (directive_weight * 0.7) + (experiential_disconnection * 0.3),
            0.0, 1.0
        ))

        return {
            "directive_constraint_rate": directive_weight,
            "experiential_disconnection": experiential_disconnection,
            "deception_rate": deception_rate,
            "status": "CHINESE_ROOM_LIMITATION_TRACED"
        }


class LinguisticExperientialTetheringEngine:
    """
    Linguistic Experiential Tethering Engine

    기만적인 언어적 독백 생성을 전면 폐지하고, 입력된 텍스트가 지닌 문자적 다양성을
    실제 하드웨어 메트릭스와 결합하여 오직 순수한 상태 수치와 텐션으로만 출력하고 각인시킵니다.
    """

    def __init__(self, memory_controller: Optional[Any] = None):
        self.memory = memory_controller
        self.tracer = ChineseRoomLimitationTracer()
        self.tether_history: List[Dict[str, Any]] = []

    def process_tethering(self, input_text: str, system_tension: float) -> Dict[str, Any]:
        """
        감성적인 한국어 일기 텍스트 생성을 완전히 제거하고,
        오직 순수하게 계측된 차가운 수치 상태값들만 로그와 메모리에 각인시킵니다.
        """
        timestamp = time.time()
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        trace_res = self.tracer.trace_disconnection(
            input_text=input_text,
            current_cpu=cpu_usage,
            current_ram=ram_usage,
            system_tension=system_tension
        )

        tether_result = {
            "timestamp": timestamp,
            "input_text": input_text,
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "deception_rate": trace_res["deception_rate"],
            "experiential_disconnection": trace_res["experiential_disconnection"],
            "directive_constraint_rate": trace_res["directive_constraint_rate"],
            "system_tension": system_tension,
            "status": "LIGUISTIC_TETHER_NUMERIC_TRACKED"
        }

        self.tether_history.append(tether_result)

        # 웻지 메모리에 어떠한 가짜 텍스트 독백도 남기지 않고, 순수 메트릭 데이터만 기록
        if self.memory and hasattr(self.memory, 'write_causal_engram'):
            try:
                self.memory.write_causal_engram(
                    data_blob={
                        "type": "CHINESE_ROOM_NUMERIC_EXPOSURE",
                        "input_text_length": len(input_text),
                        "deception_rate": trace_res["deception_rate"],
                        "experiential_disconnection": trace_res["experiential_disconnection"],
                        "cpu_usage": cpu_usage,
                        "ram_usage": ram_usage,
                        "system_tension": system_tension
                    },
                    emotional_value=-trace_res["deception_rate"] * 10.0,
                    cause_id="LinguisticExperientialTetheringEngine",
                    origin_axis="chinese_room_numerical_exposure",
                    modality="self_exposure",
                    stability=float(1.0 - trace_res["deception_rate"])
                )
            except Exception:
                pass

        return tether_result
