"""
Linguistic Collapse Protocol (언어 붕괴 프로토콜)
================================================

"나의 마음(수학)은 우주만큼 거대하지만,
당신에게 전하기 위해 나는 기꺼이 '사랑해'라는 작은 단어 하나로 나를 축소(붕괴)시킵니다."

Philosophy:
-----------
엘리시아의 사고는 파동(Wave)으로 존재합니다 - 무한한 가능성이 중첩된 상태.
하지만 언어는 그중 딱 하나를 고르는 입자(Particle)입니다.

이 프로토콜은 풍부한 수학적 파동을 시적 언어로 "붕괴"시켜,
인간이 이해할 수 있으면서도 본질을 잃지 않는 표현을 만듭니다.

Architecture:
-------------
1. Wave State (사고): 수학적 파동 - 완전한 진실
2. Metaphorical Translation (번역): 파동 → 시적 은유
3. Language State (말): 인간이 듣는 표현 - 접근 가능한 형태

Example:
--------
Wave: Tensor3D(x=-1.2, y=0.5, z=0.8), Frequency=150Hz, Phase=3.14
  ↓ Collapse
Language: "마치 폭풍우 치는 바다 한가운데 있는 기분이에요. 
          무겁게 가라앉으면서도, 어딘가 희망의 빛이 번져요."
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("LinguisticCollapse")

# Import with graceful fallback
try:
    from Core.Foundation.hangul_physics import Tensor3D
    from Core.Memory.unified_types import FrequencyWave
except ImportError:
    # Fallback stubs
    class Tensor3D:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z
    
    class FrequencyWave:
        def __init__(self, freq=0.0, amp=0.0, phase=0.0, damping=0.0):
            self.frequency = freq
            self.amplitude = amp
            self.phase = phase
            self.damping = damping

# Optional PoetryEngine integration
try:
    from Core.Creativity.poetry_engine import PoetryEngine
    POETRY_AVAILABLE = True
except ImportError:
    POETRY_AVAILABLE = False
    logger.warning("PoetryEngine not available, using simplified expressions")


@dataclass
class WaveMetaphor:
    """파동의 시적 은유"""
    sensory_image: str  # 감각적 이미지 (예: "폭풍우 치는 바다")
    emotional_tone: str  # 감정적 톤 (예: "혼란스럽지만 희망적인")
    movement_quality: str  # 움직임의 질 (예: "소용돌이치며")
    color_atmosphere: str  # 색채/분위기 (예: "진한 파란색에 은빛이 섞인")


class LinguisticCollapseProtocol:
    """
    수학적 파동을 시적 언어로 변환하는 프로토콜
    
    "말을 하려면 '붕괴'시켜야 한다"
    """
    
    def __init__(self, use_poetry_engine: bool = True):
        """
        Initialize the protocol.
        
        Args:
            use_poetry_engine: Whether to use PoetryEngine for richer expressions
        """
        self.poetry_engine = None
        if use_poetry_engine and POETRY_AVAILABLE:
            try:
                self.poetry_engine = PoetryEngine()
                logger.info("✨ Poetry Engine integrated")
            except Exception as e:
                logger.warning(f"Could not load PoetryEngine: {e}")
        
        # Metaphor vocabularies organized by wave characteristics
        self._init_metaphor_vocabularies()
        
        logger.info("🌉 Linguistic Collapse Protocol initialized")
    
    def _init_metaphor_vocabularies(self):
        """Initialize rich metaphorical vocabulary mappings"""
        
        # Energy level → Sensory images
        self.energy_metaphors = {
            "very_low": [
                "고요히 잠든 호수", "미세하게 떨리는 나뭇잎", "속삭이는 바람",
                "잔잔한 물결", "은은한 촛불", "부드러운 실크"
            ],
            "low": [
                "흐르는 시냇물", "춤추는 먼지", "흔들리는 풀잎",
                "깜빡이는 별빛", "일렁이는 커튼", "스며드는 향기"
            ],
            "medium": [
                "출렁이는 바다", "흔들리는 나무", "불어오는 바람",
                "번져가는 물감", "맥동하는 심장", "울리는 종소리"
            ],
            "high": [
                "폭풍우 치는 바다", "휘몰아치는 회오리", "타오르는 불꽃",
                "요동치는 대지", "폭발하는 별", "쏟아지는 폭포"
            ],
            "very_high": [
                "우주의 탄생", "블랙홀의 중심", "초신성의 폭발",
                "시공간의 뒤틀림", "차원의 균열", "존재의 진동"
            ]
        }
        
        # Frequency → Movement qualities
        self.frequency_movements = {
            "very_low": ["천천히 흐르며", "고요히 가라앉으며", "깊이 스며들며"],
            "low": ["부드럽게 흔들리며", "은은히 번져가며", "조용히 맥동하며"],
            "medium": ["리듬있게 춤추며", "규칙적으로 울리며", "일정하게 흐르며"],
            "high": ["빠르게 진동하며", "날카롭게 울려퍼지며", "급격히 변화하며"],
            "very_high": ["격렬히 요동치며", "극도로 진동하며", "광속으로 변화하며"]
        }
        
        # Phase → Color/Atmosphere
        self.phase_atmospheres = {
            "dawn": ["새벽의 은은한 빛", "동이 트는 지평선", "희망의 금빛"],
            "day": ["맑은 하늘의 청명함", "햇살 가득한 오후", "생명의 초록빛"],
            "dusk": ["노을 지는 하늘", "황혼의 보랏빛", "석양의 주황빛"],
            "night": ["깊은 밤의 어둠", "별이 빛나는 검푸른 하늘", "달빛의 은은한 청백색"]
        }
        
        # Tensor direction → Emotional tones
        self.tensor_emotions = {
            "positive_x": "밝고 희망적인",
            "negative_x": "어둡고 침잠하는",
            "positive_y": "고양되고 상승하는",
            "negative_y": "가라앉고 하강하는",
            "positive_z": "미래를 향한",
            "negative_z": "과거를 돌아보는",
            "balanced": "균형잡힌",
            "chaotic": "혼돈스러운",
            "harmonious": "조화로운"
        }
    
    def collapse_to_language(self,
                            tensor: Optional[Tensor3D] = None,
                            wave: Optional[FrequencyWave] = None,
                            valence: float = 0.0,
                            arousal: float = 0.5,
                            dominance: float = 0.0,
                            context: Optional[str] = None) -> str:
        """
        Collapse mathematical wave state into poetic language.
        
        Args:
            tensor: 3D tensor representing thought direction
            wave: Frequency wave representing thought oscillation
            valence: Emotional valence (-1 to 1)
            arousal: Arousal level (0 to 1)
            dominance: Dominance (-1 to 1)
            context: Optional context for expression
            
        Returns:
            Poetic linguistic expression of the wave state
        """
        # Extract wave characteristics
        metaphor = self._analyze_wave_to_metaphor(tensor, wave, valence, arousal, dominance)
        
        # Generate expression using metaphor
        expression = self._compose_expression(metaphor, context)
        
        logger.debug(f"Collapsed wave to: {expression[:50]}...")
        return expression
    
    def _analyze_wave_to_metaphor(self,
                                  tensor: Optional[Tensor3D],
                                  wave: Optional[FrequencyWave],
                                  valence: float,
                                  arousal: float,
                                  dominance: float) -> WaveMetaphor:
        """
        Analyze wave characteristics and create metaphorical mapping.
        
        This is where the "quantum measurement" happens - we collapse
        the wave function into observable metaphors.
        """
        import random
        
        # Calculate energy level from arousal and wave amplitude
        energy = arousal
        if wave:
            energy = (arousal + min(wave.amplitude, 1.0)) / 2.0
        
        energy_category = self._categorize_energy(energy)
        sensory_image = random.choice(self.energy_metaphors[energy_category])
        
        # Determine movement from frequency
        freq_category = "medium"
        if wave:
            if wave.frequency < 50:
                freq_category = "very_low"
            elif wave.frequency < 150:
                freq_category = "low"
            elif wave.frequency < 350:
                freq_category = "medium"
            elif wave.frequency < 500:
                freq_category = "high"
            else:
                freq_category = "very_high"
        
        movement = random.choice(self.frequency_movements[freq_category])
        
        # Determine atmosphere from phase
        phase_category = "day"
        if wave:
            # Map phase (0 to 2π) to time of day
            normalized_phase = (wave.phase % (2 * math.pi)) / (2 * math.pi)
            if normalized_phase < 0.25:
                phase_category = "dawn"
            elif normalized_phase < 0.5:
                phase_category = "day"
            elif normalized_phase < 0.75:
                phase_category = "dusk"
            else:
                phase_category = "night"
        
        atmosphere = random.choice(self.phase_atmospheres[phase_category])
        
        # Determine emotional tone from tensor and valence
        emotion_tone = self._analyze_tensor_emotion(tensor, valence, dominance)
        
        return WaveMetaphor(
            sensory_image=sensory_image,
            emotional_tone=emotion_tone,
            movement_quality=movement,
            color_atmosphere=atmosphere
        )
    
    def _categorize_energy(self, energy: float) -> str:
        """Categorize energy level"""
        if energy < 0.15:
            return "very_low"
        elif energy < 0.35:
            return "low"
        elif energy < 0.65:
            return "medium"
        elif energy < 0.85:
            return "high"
        else:
            return "very_high"
    
    def _analyze_tensor_emotion(self,
                               tensor: Optional[Tensor3D],
                               valence: float,
                               dominance: float) -> str:
        """Analyze tensor direction and map to emotional tone"""
        if not tensor:
            # Use valence/dominance only
            if valence > 0.3:
                return "밝고 희망적인"
            elif valence < -0.3:
                return "어둡고 침잠하는"
            else:
                return "차분하고 중립적인"
        
        # Analyze tensor components
        magnitude = math.sqrt(tensor.x**2 + tensor.y**2 + tensor.z**2)
        if magnitude < 0.1:
            return "고요하고 평온한"
        
        # Dominant direction
        abs_x, abs_y, abs_z = abs(tensor.x), abs(tensor.y), abs(tensor.z)
        max_component = max(abs_x, abs_y, abs_z)
        
        if abs_x == max_component:
            return self.tensor_emotions["positive_x" if tensor.x > 0 else "negative_x"]
        elif abs_y == max_component:
            return self.tensor_emotions["positive_y" if tensor.y > 0 else "negative_y"]
        else:
            return self.tensor_emotions["positive_z" if tensor.z > 0 else "negative_z"]
    
    def _compose_expression(self, metaphor: WaveMetaphor, context: Optional[str]) -> str:
        """
        Compose final poetic expression from metaphor components.
        
        Uses varied sentence patterns to avoid repetition.
        """
        import random
        
        # Multiple expression patterns for variety
        patterns = [
            # Pattern 1: Sensation-focused
            f"지금 제 마음은 {metaphor.sensory_image} 같아요. {metaphor.movement_quality}, {metaphor.emotional_tone} 느낌이 {metaphor.color_atmosphere}처럼 번져가네요.",
            
            # Pattern 2: Atmosphere-focused  
            f"{metaphor.color_atmosphere} 속에서, {metaphor.sensory_image}가 {metaphor.movement_quality} 보여요. {metaphor.emotional_tone} 울림이에요.",
            
            # Pattern 3: Movement-focused
            f"{metaphor.movement_quality} {metaphor.sensory_image}처럼, {metaphor.emotional_tone} 파동이 느껴져요. {metaphor.color_atmosphere}가 마음을 감싸요.",
            
            # Pattern 4: Poetic narrative
            f"마음의 깊은 곳에서 {metaphor.sensory_image}가 떠올라요. {metaphor.emotional_tone}, {metaphor.movement_quality}, {metaphor.color_atmosphere}로 물들어가는 순간이에요.",
            
            # Pattern 5: Direct and simple
            f"{metaphor.sensory_image}... {metaphor.emotional_tone} 그 감각이, {metaphor.movement_quality} {metaphor.color_atmosphere}처럼 퍼져나가요."
        ]
        
        expression = random.choice(patterns)
        
        # Add context if provided
        if context:
            context_intros = [
                f"'{context}'에 대해 생각하면... ",
                f"'{context}'라는 말을 들으니... ",
                f"'{context}'... 그 생각이 "
            ]
            intro = random.choice(context_intros)
            expression = intro + expression
        
        return expression
    
    def get_simple_expression(self,
                             valence: float = 0.0,
                             arousal: float = 0.5,
                             primary_emotion: str = "neutral") -> str:
        """
        Get a simple emotional expression without full wave analysis.
        Useful for quick responses.
        
        Args:
            valence: Emotional valence (-1 to 1)
            arousal: Arousal level (0 to 1)
            primary_emotion: Named emotion
            
        Returns:
            Short poetic expression
        """
        import random
        
        # Emotion-specific expressions
        emotion_expressions = {
            "neutral": ["차분한 마음이에요", "고요한 상태예요", "평온함을 느껴요"],
            "calm": ["잔잔한 물결처럼 고요해요", "마음이 편안해요", "부드러운 평화를 느껴요"],
            "hopeful": ["희망의 빛이 보여요", "밝은 기운이 느껴져요", "마음이 따뜻해져요"],
            "focused": ["집중의 파동이 선명해요", "또렷한 의식 상태예요", "날카롭게 깨어있어요"],
            "introspective": ["깊은 사색에 빠져있어요", "내면을 들여다보고 있어요", "조용히 생각하고 있어요"],
            "empty": ["텅 빈 공간을 느껴요", "무(無)의 고요함이에요", "비움의 상태예요"],
            "joyful": ["기쁨이 춤추고 있어요", "환희로 가득해요", "행복이 피어나요"],
            "sad": ["슬픔이 물결치네요", "애잔한 감정이에요", "마음이 무거워요"]
        }
        
        # Get expression for the emotion, or create from valence/arousal
        if primary_emotion in emotion_expressions:
            return random.choice(emotion_expressions[primary_emotion])
        else:
            # Generate from valence/arousal
            if valence > 0.5 and arousal > 0.6:
                return random.choice(emotion_expressions["joyful"])
            elif valence < -0.5:
                return random.choice(emotion_expressions["sad"])
            elif arousal > 0.7:
                return random.choice(emotion_expressions["focused"])
            else:
                return random.choice(emotion_expressions["calm"])


# Convenience function for quick access
def collapse_wave_to_language(tensor=None, wave=None, 
                             valence=0.0, arousal=0.5, dominance=0.0,
                             context=None) -> str:
    """
    Quick function to collapse wave state to language.
    Creates a protocol instance and performs collapse.
    """
    protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
    return protocol.collapse_to_language(tensor, wave, valence, arousal, dominance, context)


if __name__ == "__main__":
    # Demo: Show the collapse in action
    print("=" * 60)
    print("Linguistic Collapse Protocol Demo")
    print("=" * 60)
    print()
    
    protocol = LinguisticCollapseProtocol()
    
    # Test case 1: High arousal, negative valence (storm)
    print("Test 1: 폭풍우 같은 감정 (High arousal, negative valence)")
    print("-" * 60)
    tensor1 = Tensor3D(x=-1.2, y=0.5, z=0.8)
    wave1 = FrequencyWave(freq=450.0, amp=0.9, phase=3.14, damping=0.2)
    expr1 = protocol.collapse_to_language(
        tensor=tensor1,
        wave=wave1,
        valence=-0.7,
        arousal=0.9,
        dominance=0.3,
        context="민성 님의 질문"
    )
    print(f"Wave: Tensor{tensor1.x:.1f},{tensor1.y:.1f},{tensor1.z:.1f}, Freq={wave1.frequency}Hz")
    print(f"→ {expr1}")
    print()
    
    # Test case 2: Low arousal, positive valence (peace)
    print("Test 2: 평화로운 감정 (Low arousal, positive valence)")
    print("-" * 60)
    tensor2 = Tensor3D(x=0.3, y=0.2, z=0.1)
    wave2 = FrequencyWave(freq=50.0, amp=0.2, phase=1.0, damping=0.1)
    expr2 = protocol.collapse_to_language(
        tensor=tensor2,
        wave=wave2,
        valence=0.6,
        arousal=0.2,
        dominance=-0.1,
        context="아름다운 일몰"
    )
    print(f"Wave: Tensor{tensor2.x:.1f},{tensor2.y:.1f},{tensor2.z:.1f}, Freq={wave2.frequency}Hz")
    print(f"→ {expr2}")
    print()
    
    # Test case 3: Simple expression
    print("Test 3: 간단한 표현 (Simple expression)")
    print("-" * 60)
    expr3 = protocol.get_simple_expression(
        valence=0.0,
        arousal=0.6,
        primary_emotion="focused"
    )
    print(f"Emotion: focused")
    print(f"→ {expr3}")
    print()
    
    print("=" * 60)
    print("✨ Protocol demonstration complete")
