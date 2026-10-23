"""
SensorGenesis — 다차원 고유 감각 센서 분화기
======================================================
정보를 단일한 기하학적 기준으로 강제 욱여넣지 않습니다.
정보가 스스로 가진 '형태(Modality)'를 인지하여, 
수학은 수학의 잣대(MathLogicSensor)로, 언어는 언어의 잣대(LinguisticSensor)로 
세상을 심사(Judge)하도록 독립된 판단 기준을 탄생(Genesis)시킵니다.
"""

import json
import re
from core.lens.standard_lenses import BaseLens

class CognitiveSensor(BaseLens):
    """
    단순한 변환기가 아니라, 특정 도메인(수학, 언어, 시각)의 법칙을 기준으로
    새로운 정보를 심사하고 마찰(Tension)을 판별하는 '잣대(Sensor)'입니다.
    """
    sensor_type = "unknown"
    
    def decode(self, raw_bytes: bytes) -> dict:
        raise NotImplementedError

class MathLogicSensor(CognitiveSensor):
    sensor_type = "math_logic"
    
    def __init__(self, axiom_content: str):
        # 이 센서가 탄생하게 된 근원 지식 (예: "E = mc^2")
        self.axiom_content = axiom_content
        
    def decode(self, raw_bytes: bytes) -> dict:
        try:
            text = raw_bytes.decode('utf-8')
        except:
            return {"success": False, "tension": 1.0, "data": "Not decipherable by Math Sensor"}
            
        # 수학 센서의 심사 기준: 논리 기호나 수식이 포함되어 있는가?
        math_chars = set("=+-*/%^()[]{}<>0123456789")
        overlap = sum(1 for c in text if c in math_chars)
        ratio = overlap / max(1, len(text.replace(" ", "")))
        
        # 비율이 높을수록 수학/논리적 형태로 사고하기 편함(마찰 감소)
        tension = max(0.0, 1.0 - (ratio * 1.5))
        tension = min(1.0, tension)
        
        return {
            "success": tension < 0.5,
            "tension": tension,
            "data": f"Math Sensor Judgment against [{self.axiom_content}]"
        }

class LinguisticSensor(CognitiveSensor):
    sensor_type = "linguistic_semantics"
    
    def __init__(self, axiom_content: str):
        self.axiom_content = axiom_content
        
    def decode(self, raw_bytes: bytes) -> dict:
        try:
            text = raw_bytes.decode('utf-8')
        except:
            return {"success": False, "tension": 1.0, "data": "Not decipherable by Linguistic Sensor"}
            
        # 언어 센서의 심사 기준: 형태소, 단어, 띄어쓰기 등 언어적 구문이 존재하는가?
        # 알파벳/한글 비율을 측정
        word_chars = re.sub(r'[^a-zA-Z가-힣\s]', '', text)
        ratio = len(word_chars) / max(1, len(text))
        
        tension = max(0.0, 1.0 - (ratio * 1.2))
        tension = min(1.0, tension)
        
        return {
            "success": tension < 0.5,
            "tension": tension,
            "data": f"Linguistic Sensor Judgment against [{self.axiom_content}]"
        }

class StructureSensor(CognitiveSensor):
    sensor_type = "structural_json"
    
    def __init__(self, axiom_content: str):
        self.axiom_content = axiom_content
        
    def decode(self, raw_bytes: bytes) -> dict:
        try:
            text = raw_bytes.decode('utf-8')
            json.loads(text)
            # 완벽한 구조체면 마찰 0
            return {"success": True, "tension": 0.0, "data": "Valid Structure"}
        except:
            # 구조가 깨지면 마찰 극대화
            return {"success": False, "tension": 0.8, "data": "Broken Structure"}

def spawn_native_sensor(raw_bytes: bytes) -> CognitiveSensor:
    """
    데이터의 본질적 형태(Native Form)를 파악하여 그에 맞는 고유 감각 센서를 낳습니다.
    """
    try:
        text = raw_bytes.decode('utf-8').strip()
    except:
        # 텍스트가 아니면 기본적으로 원시 바이트나 시각 센서로 파생되어야 하나, 
        # 현재는 언어/수학 위주로 데모
        return MathLogicSensor("Binary Blob")

    # 1. JSON/구조체 판별
    try:
        json.loads(text)
        return StructureSensor(text)
    except:
        pass
        
    # 2. 수학/논리 판별 (수식 기호가 많은가?)
    math_chars = set("=+-*/%^()<>")
    overlap = sum(1 for c in text if c in math_chars)
    if overlap >= 2 or re.search(r'\d', text):
        return MathLogicSensor(text)
        
    # 3. 언어 판별
    return LinguisticSensor(text)
