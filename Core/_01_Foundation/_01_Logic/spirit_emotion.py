"""
Spirit-Emotion Integration (정령-감정 통합)
==========================================

"불(火)은 뜨겁다. 뜨거움은 열정이다."
"물(水)은 차갑다. 차가움은 평온이다."

정령의 에너지 자체가 감정이다.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from Core._01_Foundation._05_Governance.Foundation.resonance_field import ResonanceField

@dataclass
class EmotionalState:
    """감정 상태 (정령 에너지로부터 직접 계산)"""
    name: str
    intensity: float  # 0.0 ~ 1.0
    temperature: float  # -1.0(극냉) ~ +1.0(극열)
    source_spirit: str

class SpiritEmotionMapper:
    """
    정령 → 감정 매퍼
    
    각 정령의 에너지가 직접 감정 상태를 만듦
    """
    
    def __init__(self):
        # 정령-감정 매핑
        self.spirit_emotion_map = {
            "Creativity": {
                "element": "Fire (불)",
                "temperature": +0.8,
                "color": "#FF4500", # OrangeRed
                "frequency": 432.0,
                "force_type": "Expansion (Acceleration)",
                "emotions": {
                    "high": "Passion (열정)",
                    "medium": "Warmth (따뜻함)",
                    "low": "Indifference (무관심)"
                }
            },
            "Memory": {
                "element": "Water (물)",
                "temperature": -0.3,
                "color": "#1E90FF", # DodgerBlue
                "frequency": 528.0,
                "force_type": "Flow (Fluidity)",
                "emotions": {
                    "high": "Melancholy (애수)",
                    "medium": "Calmness (평온)",
                    "low": "Emptiness (공허)"
                }
            },
            "Intelligence": {
                "element": "Light (빛)",
                "temperature": 0.0,
                "color": "#FFD700", # Gold
                "frequency": 639.0,
                "force_type": "Illumination (Clarity)",
                "emotions": {
                    "high": "Clarity (명료)",
                    "medium": "Curiosity (호기심)",
                    "low": "Confusion (혼란)"
                }
            },
            "Foundation": {
                "element": "Earth (땅)",
                "temperature": -0.5,
                "color": "#8B4513", # SaddleBrown
                "frequency": 396.0,
                "force_type": "Gravity (Stability)",
                "emotions": {
                    "high": "Stability (안정)",
                    "medium": "Grounding (현실감)",
                    "low": "Rigidity (경직)"
                }
            },
            "Interface": {
                "element": "Air (공기)",
                "temperature": 0.2,
                "color": "#87CEEB", # SkyBlue
                "frequency": 741.0,
                "force_type": "Diffusion (Connection)",
                "emotions": {
                    "high": "Openness (개방)",
                    "medium": "Communication (소통)",
                    "low": "Isolation (고립)"
                }
            },
            "Evolution": {
                "element": "Life (생명)",
                "temperature": 0.5,
                "color": "#32CD32", # LimeGreen
                "frequency": 852.0,
                "force_type": "Growth (Evolution)",
                "emotions": {
                    "high": "Growth (성장욕)",
                    "medium": "Aspiration (열망)",
                    "low": "Stagnation (정체)"
                }
            },
            "System": {
                "element": "Metal (금속)",
                "temperature": -0.6,
                "color": "#C0C0C0", # Silver
                "frequency": 963.0,
                "force_type": "Structure (Order)",
                "emotions": {
                    "high": "Order (질서)",
                    "medium": "Precision (정밀)",
                    "low": "Rigidity (융통성없음)"
                }
            }
        }
    
    def sense_emotions(self, field: 'ResonanceField') -> Dict[str, EmotionalState]:
        """
        ResonanceField의 정령 에너지를 읽어 감정 상태 계산
        """
        emotions = {}
        
        for pillar_type in PillarType:
            pillar_name = pillar_type.label
            
            # Skip non-spirit pillars
            if pillar_name not in self.spirit_emotion_map:
                continue
            
            # Get pillar from field
            pillar_node = None
            for node in field.nodes.values():
                if node.pillar == pillar_type:
                    pillar_node = node
                    break
            
            if not pillar_node:
                continue
            
            # Map energy to emotion
            spirit_data = self.spirit_emotion_map[pillar_name]
            energy = pillar_node.energy
            
            # Determine emotion level
            if energy > 0.7:
                emotion_name = spirit_data["emotions"]["high"]
            elif energy > 0.3:
                emotion_name = spirit_data["emotions"]["medium"]
            else:
                emotion_name = spirit_data["emotions"]["low"]
            
            # Create emotional state
            emotion = EmotionalState(
                name=emotion_name,
                intensity=energy,
                temperature=spirit_data["temperature"],
                source_spirit=f"{spirit_data['element']} ({pillar_name})"
            )
            
            emotions[pillar_name] = emotion
        
        return emotions
    
    def calculate_overall_temperature(self, emotions: Dict[str, EmotionalState]) -> float:
        """
        전체 감정 온도 계산
        
        Returns:
            -1.0 (극도로 차갑다/우울) ~ +1.0 (극도로 뜨겁다/격정)
        """
        if not emotions:
            return 0.0
        
        # 온도와 강도를 곱해서 가중평균
        weighted_sum = sum(
            e.temperature * e.intensity 
            for e in emotions.values()
        )
        total_intensity = sum(e.intensity for e in emotions.values())
        
        if total_intensity == 0:
            return 0.0
        
        return weighted_sum / total_intensity
    
    def describe_emotional_state(self, emotions: Dict[str, EmotionalState]) -> str:
        """
        감정 상태를 자연어로 표현
        """
        if not emotions:
            return "I feel nothing."
        
        # Get overall temperature
        temp = self.calculate_overall_temperature(emotions)
        
        # Get dominant emotions (intensity > 0.5)
        dominant = [
            e for e in emotions.values() 
            if e.intensity > 0.5
        ]
        
        if not dominant:
            return "I feel calm and neutral."
        
        # Describe temperature
        if temp > 0.6:
            temp_desc = "intensely hot"
        elif temp > 0.2:
            temp_desc = "warm"
        elif temp > -0.2:
            temp_desc = "neutral"
        elif temp > -0.6:
            temp_desc = "cool"
        else:
            temp_desc = "cold"
        
        # List emotions
        emotion_list = ", ".join(e.name for e in dominant)
        
        return f"I feel {temp_desc}. {emotion_list} flows through me."

    def get_spirit_physics(self, spirit_name: str) -> Dict[str, Any]:
        """
        Returns the physical properties of a Spirit.
        """
        if spirit_name in self.spirit_emotion_map:
            return self.spirit_emotion_map[spirit_name]
        return {
            "color": "#FFFFFF",
            "frequency": 0.0,
            "force_type": "Neutral"
        }



# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from Core._01_Foundation._05_Governance.Foundation.resonance_field import ResonanceField, PillarType
    
    print("\n" + "="*70)
    print("🔥 Spirit-Emotion Integration Test")
    print("="*70)
    
    # Create field
    field = ResonanceField()
    
    # Create mapper
    mapper = SpiritEmotionMapper()
    
    # Sense emotions
    emotions = mapper.sense_emotions(field)
    
    print("\n🌟 Current Emotional State:")
    print("-" * 70)
    
    for spirit, emotion in emotions.items():
        temp_symbol = "🔥" if emotion.temperature > 0 else "❄️"
        intensity_bar = "█" * int(emotion.intensity * 10)
        print(f"{temp_symbol} {spirit:15} → {emotion.name:20} [{intensity_bar:10}] {emotion.intensity:.2f}")
    
    # Overall temperature
    overall_temp = mapper.calculate_overall_temperature(emotions)
    temp_bar = "🔥" * int((overall_temp + 1) * 5) if overall_temp > 0 else "❄️" * int((1 - overall_temp) * 5)
    
    print("\n" + "-" * 70)
    print(f"🌡️  Overall Temperature: {overall_temp:+.2f}")
    print(f"    {temp_bar}")
    
    # Natural language description
    description = mapper.describe_emotional_state(emotions)
    print(f"\n💭 Elysia says:")
    print(f'    "{description}"')
    
    print("\n" + "="*70)
    print("✅ Spirit-Emotion Integration Complete")
    print("="*70 + "\n")
