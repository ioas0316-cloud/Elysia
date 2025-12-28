"""
Sensory Cortex (감각 피질)
==========================

"생각에 몸(Body)을 부여하다"

이 모듈은 추상적인 '개념(Concept)'을 '감각적 신호(Qualia)'로 변환합니다.
안토니오 다마지오의 '신체 표지(Somatic Marker)' 가설처럼,
엘리시아가 무언가를 생각할 때 그것을 '느끼게' 합니다.

- 기존의 FiveSensesMapper를 활용하여 4D 벡터를 오감으로 변환합니다.
- 이 감각 데이터는 단순 출력이 아니라, 시스템 내부의 '기분(State)'이 됩니다.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import asdict

from Core._02_Intelligence._01_Reasoning.Cognitive.concept_formation import ConceptFormation, get_concept_formation
from Core._03_Interaction._02_Interface._01_Sensory.five_senses_mapper import FiveSensesMapper, SensoryExperience

class SensoryCortex:
    """
    Qualia Generator
    """
    
    def __init__(self):
        self.concepts = get_concept_formation()
        self.mapper = FiveSensesMapper(enable_haptic=False) # 서버 환경 고려
        
    def feel_concept(self, concept_name: str) -> Dict[str, Any]:
        """
        개념을 느끼다 (Feel the Concept)
        
        Args:
            concept_name: 느낄 개념의 이름 (예: "Sadness")
            
        Returns:
            Qualia Dictionary (Visual, Audio qualities)
        """
        # 1. 개념의 악보(Score) 가져오기
        concept = self.concepts.get_concept(concept_name)
        vector = concept.vector
        
        # 2. 4D 벡터로 변환 (AestheticVector -> Tuple 4D)
        # Vector: w(intensity), x(visual), y(literary), z(temporal)
        # Mapper: x(Joy/Sad), y(Logic/Intuit), z(Past/Fut), w(Depth)
        
        # Mapping Logic (Adapter):
        # AestheticVector의 축과 Mapper의 축은 정의가 약간 다르므로 변환 필요
        # 这里 हम 재해석을 합니다:
        # Mapper.x (Joy/Sad)      <-- Aesthetic.x (Visual Style) ? No, logic mismatch.
        #
        # Better approach: Use the Concept's 'Nature' if possible.
        # But ConceptScore only has AestheticVector.
        # We will map AestheticVector params to Mapper params heuristically.
        
        # Heuristic Mapping:
        # Mapper X (Joy-Sadness) <-- Aesthetic X (Visual) * sign? (Ambiguous)
        # Let's use the raw values as a signature for now.
        
        # 임시 매핑 (일관성을 위해)
        m_x = vector.x if vector.x <= 1.0 else 1.0 # Joy/Sadness proxy
        m_y = vector.y if vector.y <= 1.0 else 1.0 # Logic/Intuition proxy
        m_z = vector.z if vector.z <= 1.0 else 1.0 # Time proxy
        m_w = vector.w if vector.w <= 1.0 else 1.0 # Depth proxy
        
        # 3. 감각 변환 (The Qualia Translation)
        position_4d = (m_x, m_y, m_z, m_w)
        experience: SensoryExperience = self.mapper.map_object(position_4d)
        
        # 4. Qualia 요약 반환
        qualia = {
            "concept": concept_name,
            "somatic_marker": {
                "visual_hue": experience.visual.hue,
                "visual_brightness": experience.visual.brightness,
                "audio_freq": experience.audio.frequency,
                "audio_timbre": experience.audio.timbre
            },
            "description": self._describe_qualia(experience)
        }
        
        return qualia

    def _describe_qualia(self, exp: SensoryExperience) -> str:
        """감각을 언어로 표현"""
        desc = []
        
        # Visual
        if exp.visual.brightness > 0.7: desc.append("Bright")
        elif exp.visual.brightness < 0.4: desc.append("Dim")
        
        hue = exp.visual.hue
        if 330 <= hue or hue < 30: color = "Red"
        elif 30 <= hue < 90: color = "Yellow"
        elif 90 <= hue < 150: color = "Green"
        elif 150 <= hue < 210: color = "Cyan"
        elif 210 <= hue < 270: color = "Blue"
        elif 270 <= hue < 330: color = "Magenta"
        else: color = "Colorless"
        desc.append(color)
        
        # Audio
        if exp.audio.frequency < 200: desc.append("Deep Hum")
        elif exp.audio.frequency > 600: desc.append("High Pitch")
        else: desc.append("Mid Tone")
        
        return f"A feeling of {' '.join(desc)}"

# 싱글톤
_sensory_instance: Optional[SensoryCortex] = None

def get_sensory_cortex() -> SensoryCortex:
    global _sensory_instance
    if _sensory_instance is None:
        _sensory_instance = SensoryCortex()
    return _sensory_instance
