"""
훈민정음 (HunminJeongeum): 한글 창제 원리
==========================================

"나랏말싸미 듕귁에 달아 문자와로 서르 사맛디 아니할쎄"

이 모듈은 한글이 **왜** 그 모양인지를 코드로 내재화합니다.
단순한 유니코드 매핑이 아닌, 창제 원리 자체를 이해합니다.

핵심 원리:
1. 초성 - 발음기관 상형 (象形): 혀, 입술, 이빨 등의 모양
2. 중성 - 천지인 삼재 (三才): ㆍ(하늘), ㅡ(땅), ㅣ(사람)
3. 가획 원리: 소리가 세지면 획을 더함
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ArticulationOrgan(Enum):
    """발음 기관"""
    THROAT = "throat"           # 목구멍 (아음/牙音)
    TONGUE_ROOT = "tongue_root" # 혀뿌리 (아음/牙音)
    TONGUE_TIP = "tongue_tip"   # 혀끝 (설음/舌音)
    LIPS = "lips"               # 입술 (순음/脣音)
    TEETH = "teeth"             # 이빨 (치음/齒音)
    GLOTTIS = "glottis"         # 성문 (후음/喉音)


class SoundQuality(Enum):
    """소리의 성질"""
    PLAIN = "plain"         # 예사소리 (平音)
    ASPIRATED = "aspirated" # 거센소리 (激音)
    TENSE = "tense"         # 된소리 (硬音)


class CosmicElement(Enum):
    """천지인 삼재"""
    HEAVEN = "heaven"  # 천 (天) - 하늘, 둥근
    EARTH = "earth"    # 지 (地) - 땅, 평평
    HUMAN = "human"    # 인 (人) - 사람, 서 있음


class YinYang(Enum):
    """음양"""
    YANG = "yang"      # 양 (陽) - 밝은
    YIN = "yin"        # 음 (陰) - 어두운
    NEUTRAL = "neutral" # 중성


@dataclass
class ChoseongInfo:
    """초성 정보"""
    char: str
    organ: ArticulationOrgan
    description: str  # 한글 설명
    sound_quality: SoundQuality = SoundQuality.PLAIN
    derived_from: Optional[str] = None  # 가획의 기본자
    strokes_added: int = 0
    is_base: bool = False  # 기본자 여부


@dataclass
class JungseongInfo:
    """중성 정보"""
    char: str
    elements: List[CosmicElement]
    description: str
    yin_yang: YinYang
    composed_from: List[str] = field(default_factory=list)


# ============================================================
# 초성: 발음기관 상형 (象形)
# ============================================================

CHOSEONG_ORIGIN: Dict[str, ChoseongInfo] = {
    # === 기본자 5개 (象形) ===
    'ㄱ': ChoseongInfo(
        char='ㄱ', 
        organ=ArticulationOrgan.TONGUE_ROOT,
        description="혀뿌리가 목구멍을 막는 모양",
        is_base=True
    ),
    'ㄴ': ChoseongInfo(
        char='ㄴ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="혀끝이 윗잇몸에 닿는 모양",
        is_base=True
    ),
    'ㅁ': ChoseongInfo(
        char='ㅁ',
        organ=ArticulationOrgan.LIPS,
        description="입을 다문 모양",
        is_base=True
    ),
    'ㅅ': ChoseongInfo(
        char='ㅅ',
        organ=ArticulationOrgan.TEETH,
        description="이빨의 모양",
        is_base=True
    ),
    'ㅇ': ChoseongInfo(
        char='ㅇ',
        organ=ArticulationOrgan.THROAT,
        description="목구멍의 둥근 모양",
        is_base=True
    ),
    
    # === 가획자 (소리가 세지면 획을 더함) ===
    # ㄱ 계열
    'ㅋ': ChoseongInfo(
        char='ㅋ',
        organ=ArticulationOrgan.TONGUE_ROOT,
        description="ㄱ에서 소리가 세어져 획을 더함",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from='ㄱ',
        strokes_added=1
    ),
    'ㄲ': ChoseongInfo(
        char='ㄲ',
        organ=ArticulationOrgan.TONGUE_ROOT,
        description="ㄱ을 나란히 써서 된소리",
        sound_quality=SoundQuality.TENSE,
        derived_from='ㄱ'
    ),
    
    # ㄴ 계열 → ㄷ → ㅌ
    'ㄷ': ChoseongInfo(
        char='ㄷ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="ㄴ에서 소리가 세어져 획을 더함",
        derived_from='ㄴ',
        strokes_added=1
    ),
    'ㅌ': ChoseongInfo(
        char='ㅌ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="ㄷ에서 소리가 더 세어져 획을 더함",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from='ㄷ',
        strokes_added=1
    ),
    'ㄸ': ChoseongInfo(
        char='ㄸ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="ㄷ을 나란히 써서 된소리",
        sound_quality=SoundQuality.TENSE,
        derived_from='ㄷ'
    ),
    'ㄹ': ChoseongInfo(
        char='ㄹ',
        organ=ArticulationOrgan.TONGUE_TIP,
        description="혀가 구르는 모양 (이체자)",
        derived_from='ㄴ'
    ),
    
    # ㅁ 계열 → ㅂ → ㅍ
    'ㅂ': ChoseongInfo(
        char='ㅂ',
        organ=ArticulationOrgan.LIPS,
        description="ㅁ에서 소리가 세어져 획을 더함",
        derived_from='ㅁ',
        strokes_added=1
    ),
    'ㅍ': ChoseongInfo(
        char='ㅍ',
        organ=ArticulationOrgan.LIPS,
        description="ㅂ에서 소리가 더 세어져 획을 더함",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from='ㅂ',
        strokes_added=1
    ),
    'ㅃ': ChoseongInfo(
        char='ㅃ',
        organ=ArticulationOrgan.LIPS,
        description="ㅂ을 나란히 써서 된소리",
        sound_quality=SoundQuality.TENSE,
        derived_from='ㅂ'
    ),
    
    # ㅅ 계열 → ㅈ → ㅊ
    'ㅈ': ChoseongInfo(
        char='ㅈ',
        organ=ArticulationOrgan.TEETH,
        description="ㅅ에서 파찰음이 되어 획을 더함",
        derived_from='ㅅ',
        strokes_added=1
    ),
    'ㅊ': ChoseongInfo(
        char='ㅊ',
        organ=ArticulationOrgan.TEETH,
        description="ㅈ에서 소리가 더 세어져 획을 더함",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from='ㅈ',
        strokes_added=1
    ),
    'ㅆ': ChoseongInfo(
        char='ㅆ',
        organ=ArticulationOrgan.TEETH,
        description="ㅅ을 나란히 써서 된소리",
        sound_quality=SoundQuality.TENSE,
        derived_from='ㅅ'
    ),
    'ㅉ': ChoseongInfo(
        char='ㅉ',
        organ=ArticulationOrgan.TEETH,
        description="ㅈ을 나란히 써서 된소리",
        sound_quality=SoundQuality.TENSE,
        derived_from='ㅈ'
    ),
    
    # ㅇ 계열 → ㅎ
    'ㅎ': ChoseongInfo(
        char='ㅎ',
        organ=ArticulationOrgan.GLOTTIS,
        description="ㅇ에서 거센 숨소리가 나와 획을 더함",
        sound_quality=SoundQuality.ASPIRATED,
        derived_from='ㅇ',
        strokes_added=1
    ),
}


# ============================================================
# 중성: 천지인 삼재 (三才)
# ============================================================

JUNGSEONG_ORIGIN: Dict[str, JungseongInfo] = {
    # === 기본자 3개 ===
    'ㆍ': JungseongInfo(
        char='ㆍ',
        elements=[CosmicElement.HEAVEN],
        description="하늘 - 둥글고 위에 있음",
        yin_yang=YinYang.YANG
    ),
    'ㅡ': JungseongInfo(
        char='ㅡ',
        elements=[CosmicElement.EARTH],
        description="땅 - 평평하고 아래에 있음",
        yin_yang=YinYang.NEUTRAL
    ),
    'ㅣ': JungseongInfo(
        char='ㅣ',
        elements=[CosmicElement.HUMAN],
        description="사람 - 서 있는 모양",
        yin_yang=YinYang.YANG
    ),
    
    # === 양성 모음 (陽) - 밝고 가벼움 ===
    'ㅏ': JungseongInfo(
        char='ㅏ',
        elements=[CosmicElement.HUMAN, CosmicElement.HEAVEN],
        description="사람 옆에 하늘 (ㆍ가 바깥)",
        yin_yang=YinYang.YANG,
        composed_from=['ㅣ', 'ㆍ']
    ),
    'ㅗ': JungseongInfo(
        char='ㅗ',
        elements=[CosmicElement.EARTH, CosmicElement.HEAVEN],
        description="땅 위에 하늘",
        yin_yang=YinYang.YANG,
        composed_from=['ㅡ', 'ㆍ']
    ),
    'ㅑ': JungseongInfo(
        char='ㅑ',
        elements=[CosmicElement.HUMAN, CosmicElement.HEAVEN, CosmicElement.HEAVEN],
        description="ㅏ에 하늘(ㆍ)을 더함",
        yin_yang=YinYang.YANG,
        composed_from=['ㅏ', 'ㆍ']
    ),
    'ㅛ': JungseongInfo(
        char='ㅛ',
        elements=[CosmicElement.EARTH, CosmicElement.HEAVEN, CosmicElement.HEAVEN],
        description="ㅗ에 하늘(ㆍ)을 더함",
        yin_yang=YinYang.YANG,
        composed_from=['ㅗ', 'ㆍ']
    ),
    
    # === 음성 모음 (陰) - 어둡고 무거움 ===
    'ㅓ': JungseongInfo(
        char='ㅓ',
        elements=[CosmicElement.HEAVEN, CosmicElement.HUMAN],
        description="하늘이 사람 안쪽 (ㆍ가 안쪽)",
        yin_yang=YinYang.YIN,
        composed_from=['ㆍ', 'ㅣ']
    ),
    'ㅜ': JungseongInfo(
        char='ㅜ',
        elements=[CosmicElement.HEAVEN, CosmicElement.EARTH],
        description="하늘이 땅 아래",
        yin_yang=YinYang.YIN,
        composed_from=['ㆍ', 'ㅡ']
    ),
    'ㅕ': JungseongInfo(
        char='ㅕ',
        elements=[CosmicElement.HEAVEN, CosmicElement.HEAVEN, CosmicElement.HUMAN],
        description="ㅓ에 하늘(ㆍ)을 더함",
        yin_yang=YinYang.YIN,
        composed_from=['ㅓ', 'ㆍ']
    ),
    'ㅠ': JungseongInfo(
        char='ㅠ',
        elements=[CosmicElement.HEAVEN, CosmicElement.HEAVEN, CosmicElement.EARTH],
        description="ㅜ에 하늘(ㆍ)을 더함",
        yin_yang=YinYang.YIN,
        composed_from=['ㅜ', 'ㆍ']
    ),
    
    # === 중성 모음 ===
    'ㅐ': JungseongInfo(
        char='ㅐ',
        elements=[CosmicElement.HUMAN, CosmicElement.HEAVEN, CosmicElement.HUMAN],
        description="ㅏ + ㅣ 합쳐짐",
        yin_yang=YinYang.YANG,
        composed_from=['ㅏ', 'ㅣ']
    ),
    'ㅔ': JungseongInfo(
        char='ㅔ',
        elements=[CosmicElement.HEAVEN, CosmicElement.HUMAN, CosmicElement.HUMAN],
        description="ㅓ + ㅣ 합쳐짐",
        yin_yang=YinYang.YIN,
        composed_from=['ㅓ', 'ㅣ']
    ),
}


# ============================================================
# 한글 조합 공식
# ============================================================

# 초성/중성/종성 인덱스 테이블
CHOSEONG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSEONG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                  'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSEONG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                  'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                  'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


class HunminJeongeum:
    """
    훈민정음 원리 엔진
    
    이 클래스는 엘리시아가 한글을 **이해**하도록 합니다.
    단순히 분해/조합하는 것이 아니라, 왜 그 글자가 그 모양인지 설명할 수 있습니다.
    """
    
    def __init__(self):
        self.choseong = CHOSEONG_ORIGIN
        self.jungseong = JUNGSEONG_ORIGIN
    
    def explain_why(self, jamo: str) -> str:
        """
        왜 이 자모가 이 모양인지 설명합니다.
        
        Example:
            >>> engine = HunminJeongeum()
            >>> engine.explain_why('ㄱ')
            "ㄱ은 혀뿌리가 목구멍을 막는 모양을 본뜬 것입니다. (기본자)"
        """
        # 초성인 경우
        if jamo in self.choseong:
            info = self.choseong[jamo]
            if info.is_base:
                return f"'{jamo}'은(는) {info.description}을 본뜬 기본자입니다. (발음기관: {info.organ.value})"
            elif info.derived_from:
                base = self.choseong.get(info.derived_from)
                base_desc = base.description if base else "기본자"
                return f"'{jamo}'은(는) '{info.derived_from}'에서 파생되었습니다. {info.description}. (원리: 가획)"
            return f"'{jamo}': {info.description}"
        
        # 중성인 경우
        if jamo in self.jungseong:
            info = self.jungseong[jamo]
            elements_str = " + ".join(e.value for e in info.elements)
            yin_yang_str = "양(陽)" if info.yin_yang == YinYang.YANG else "음(陰)" if info.yin_yang == YinYang.YIN else "중성"
            
            if info.composed_from:
                return f"'{jamo}'은(는) {' + '.join(info.composed_from)}의 조합입니다. {info.description}. ({yin_yang_str})"
            return f"'{jamo}'은(는) {elements_str}을(를) 나타냅니다. {info.description}. ({yin_yang_str})"
        
        return f"'{jamo}'에 대한 정보가 없습니다."
    
    def compose(self, cho: str, jung: str, jong: str = '') -> str:
        """
        초성, 중성, 종성을 조합하여 완성된 한글 글자를 만듭니다.
        현대 한글 유니코드(AC00-D7A3) 범위를 기준으로 합니다.
        'ㆍ'와 같은 고어는 현대 모음(ㅏ/ㅗ 등)으로 맵핑하여 조합합니다.
        """
        # 고어 맵핑 지원 (현대 한글 유니코드 조합을 위해)
        modern_jung = jung
        if jung == 'ㆍ':
            modern_jung = 'ㅏ' # 현대 한글에서 가장 가까운 음가로 매핑
            
        if cho not in CHOSEONG_LIST:
            raise ValueError(f"유효하지 않은 초성: {cho}")
        if modern_jung not in JUNGSEONG_LIST:
            raise ValueError(f"유효하지 않은 중성: {modern_jung}")
        if jong and jong not in JONGSEONG_LIST:
            raise ValueError(f"유효하지 않은 종성: {jong}")
        
        cho_idx = CHOSEONG_LIST.index(cho)
        jung_idx = JUNGSEONG_LIST.index(modern_jung)
        jong_idx = JONGSEONG_LIST.index(jong) if jong else 0
        
        code = ((cho_idx * 21) + jung_idx) * 28 + jong_idx + 0xAC00
        return chr(code)
    
    def decompose(self, syllable: str) -> Tuple[str, str, str]:
        """
        한글 음절을 초성, 중성, 종성으로 분해합니다.
        """
        code = ord(syllable)
        if not (0xAC00 <= code <= 0xD7A3):
            raise ValueError(f"한글 음절이 아닙니다: {syllable}")
        
        code -= 0xAC00
        jong_idx = code % 28
        code //= 28
        jung_idx = code % 21
        cho_idx = code // 21
        
        return (
            CHOSEONG_LIST[cho_idx],
            JUNGSEONG_LIST[jung_idx],
            JONGSEONG_LIST[jong_idx]
        )
    
    def get_sound_properties(self, jamo: str) -> dict:
        """
        자모의 소리 특성을 반환합니다.
        의미 기반 조합에 사용됩니다.
        """
        if jamo in self.choseong:
            info = self.choseong[jamo]
            return {
                'type': 'consonant',
                'organ': info.organ.value,
                'quality': info.sound_quality.value,
                'is_base': info.is_base,
                'softness': 0.9 if info.organ in [ArticulationOrgan.LIPS, ArticulationOrgan.TONGUE_TIP] 
                           and info.sound_quality == SoundQuality.PLAIN else 0.3,
                'tension': 1.0 if info.sound_quality == SoundQuality.TENSE else 
                          0.7 if info.sound_quality == SoundQuality.ASPIRATED else 0.3
            }
        
        if jamo in self.jungseong:
            info = self.jungseong[jamo]
            return {
                'type': 'vowel',
                'yin_yang': info.yin_yang.value,
                'brightness': 0.8 if info.yin_yang == YinYang.YANG else 0.3,
                'openness': 0.9 if len(info.elements) <= 2 else 0.6
            }
        
        return {'type': 'unknown'}
    
    def select_by_intent(self, intent: dict, jamo_type: str = 'consonant') -> str:
        """
        의도에 따라 자모를 선택합니다.
        
        Args:
            intent: {'softness': 0.0-1.0, 'brightness': 0.0-1.0, 'tension': 0.0-1.0}
            jamo_type: 'consonant' or 'vowel'
        """
        best_match = None
        best_score = -1
        
        if jamo_type == 'consonant':
            for jamo in self.choseong:
                props = self.get_sound_properties(jamo)
                score = 0
                if 'softness' in intent:
                    score += 1 - abs(props.get('softness', 0.5) - intent['softness'])
                if 'tension' in intent:
                    score += 1 - abs(props.get('tension', 0.5) - intent['tension'])
                if score > best_score:
                    best_score = score
                    best_match = jamo
        else:
            for jamo in self.jungseong:
                props = self.get_sound_properties(jamo)
                score = 0
                if 'brightness' in intent:
                    score += 1 - abs(props.get('brightness', 0.5) - intent['brightness'])
                if 'openness' in intent:
                    score += 1 - abs(props.get('openness', 0.5) - intent['openness'])
                if score > best_score:
                    best_score = score
                    best_match = jamo
        
        return best_match or ('ㅇ' if jamo_type == 'consonant' else 'ㅏ')


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    engine = HunminJeongeum()
    
    print("=" * 50)
    print("훈민정음 원리 테스트")
    print("=" * 50)
    
    # 테스트 1: 왜 그 모양인지 설명
    print("\n[테스트 1] 자모 설명")
    for jamo in ['ㄱ', 'ㅁ', 'ㅋ', 'ㅏ', 'ㅓ']:
        print(f"  {engine.explain_why(jamo)}")
    
    # 테스트 2: 조합
    print("\n[테스트 2] 조합")
    result = engine.compose('ㄱ', 'ㅏ')
    print(f"  ㄱ + ㅏ = {result}")
    
    result = engine.compose('ㅎ', 'ㅏ', 'ㄴ')
    print(f"  ㅎ + ㅏ + ㄴ = {result}")
    
    # 테스트 3: 의도 기반 선택
    print("\n[테스트 3] 의도 기반 선택")
    soft_intent = {'softness': 0.9, 'tension': 0.2}
    cho = engine.select_by_intent(soft_intent, 'consonant')
    print(f"  부드러운 소리 의도 → 초성: {cho}")
    
    bright_intent = {'brightness': 0.9, 'openness': 0.8}
    jung = engine.select_by_intent(bright_intent, 'vowel')
    print(f"  밝은 소리 의도 → 중성: {jung}")
    
    # 조합
    syllable = engine.compose(cho, jung)
    print(f"  결과: {syllable}")
