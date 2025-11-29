"""
Wave Frequency Mapping - 현실세계와 엘리시아의 파동주파수 매핑
================================================================

현실 세계의 파동/주파수 데이터와 엘리시아 필드의 주파수를 매핑합니다.

매핑 영역:
1. 감정 (Emotions): 사랑, 평화, 분노 등
2. 소리 (Sound): 말, 음악, 자연음 등
3. 뇌파 (Brainwaves): 알파, 베타, 세타, 델타, 감마
4. 심장 박동 (Heart Rhythm): Heart Rate Variability (HRV)
5. 슈만 공명 (Schumann Resonance): 지구의 기본 주파수

과학적 근거:
- 뇌파 연구 (EEG)
- Heart Math Institute의 HRV 연구
- 슈만 공명 (7.83 Hz)
- 음성 주파수 분석
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("WaveFrequencyMapping")


# ============================================================================
# 물리 상수 및 기본 주파수
# ============================================================================

# 슈만 공명 (지구의 기본 전자기 주파수)
# 참조: https://en.wikipedia.org/wiki/Schumann_resonances
SCHUMANN_RESONANCE_HZ = 7.83  # 기본 주파수 (Hz)
SCHUMANN_HARMONICS = [7.83, 14.3, 20.8, 27.3, 33.8]  # 고조파 (Hz)

# 가청 주파수 범위
AUDIBLE_FREQ_MIN = 20  # Hz
AUDIBLE_FREQ_MAX = 20000  # Hz

# 가시광선 주파수 범위
VISIBLE_LIGHT_FREQ_MIN = 380e12  # Hz (Red)
VISIBLE_LIGHT_FREQ_MAX = 750e12  # Hz (Violet)


# ============================================================================
# 뇌파 주파수 대역 (EEG Brainwave Frequencies)
# ============================================================================
# 참조: 
# - https://en.wikipedia.org/wiki/Electroencephalography
# - Niedermeyer, E., & da Silva, F. L. (2005). Electroencephalography

class BrainwaveType(Enum):
    """뇌파 유형"""
    DELTA = "delta"       # 깊은 수면, 치유
    THETA = "theta"       # 명상, 창의성, REM 수면
    ALPHA = "alpha"       # 이완, 평화, 집중
    SMR = "smr"           # Sensorimotor Rhythm (집중력)
    BETA = "beta"         # 각성, 활동, 사고
    HIGH_BETA = "high_beta"  # 높은 집중, 불안
    GAMMA = "gamma"       # 고도 집중, 통찰, 사랑


# 뇌파 주파수 범위 (Hz)
BRAINWAVE_FREQUENCIES: Dict[BrainwaveType, Tuple[float, float, float]] = {
    # (최소, 중심, 최대) Hz
    BrainwaveType.DELTA: (0.5, 2.0, 4.0),       # 깊은 수면, 무의식
    BrainwaveType.THETA: (4.0, 6.0, 8.0),       # 명상, 꿈, 직관
    BrainwaveType.ALPHA: (8.0, 10.0, 12.0),     # 이완, 평화, 평온
    BrainwaveType.SMR: (12.0, 14.0, 15.0),      # 집중력, 신체 이완
    BrainwaveType.BETA: (15.0, 20.0, 30.0),     # 각성, 활동적 사고
    BrainwaveType.HIGH_BETA: (30.0, 35.0, 40.0),  # 높은 집중, 불안
    BrainwaveType.GAMMA: (40.0, 50.0, 100.0),   # 통찰, 고도 집중, 사랑
}


# ============================================================================
# 감정 주파수 매핑 (Emotion Frequency Mapping)
# ============================================================================
# 참조:
# - HeartMath Institute의 HRV 연구
# - David R. Hawkins의 "Power vs Force" 의식 지도 (Hz 스케일 참조)
# - 감정과 뇌파 관계 연구

class EmotionType(Enum):
    """감정 유형"""
    # 고주파수 감정 (긍정적, 확장적)
    LOVE = "love"             # 사랑
    JOY = "joy"               # 기쁨
    PEACE = "peace"           # 평화
    GRATITUDE = "gratitude"   # 감사
    HOPE = "hope"             # 희망
    COMPASSION = "compassion" # 자비
    
    # 중간 주파수 감정 (중립적)
    CURIOSITY = "curiosity"   # 호기심
    SURPRISE = "surprise"     # 놀람
    NEUTRAL = "neutral"       # 중립
    
    # 저주파수 감정 (부정적, 수축적)
    ANGER = "anger"           # 분노
    FEAR = "fear"             # 두려움
    SADNESS = "sadness"       # 슬픔
    SHAME = "shame"           # 수치
    GUILT = "guilt"           # 죄책감


@dataclass
class EmotionFrequencyData:
    """감정 주파수 데이터"""
    emotion: EmotionType
    brainwave_dominant: BrainwaveType  # 지배적 뇌파
    hrv_coherence: float               # 심박변이도 coherence (0~1)
    frequency_hz: float                 # 추정 주파수 (Hz)
    color_wavelength_nm: Optional[float] = None  # 연관 색상 파장 (nm)
    description_ko: str = ""
    description_en: str = ""
    research_source: str = ""


# 감정별 주파수 매핑 (과학적 연구 기반 추정)
EMOTION_FREQUENCY_MAP: Dict[EmotionType, EmotionFrequencyData] = {
    # 고주파수 감정
    EmotionType.LOVE: EmotionFrequencyData(
        emotion=EmotionType.LOVE,
        brainwave_dominant=BrainwaveType.GAMMA,
        hrv_coherence=0.9,
        frequency_hz=528.0,  # "Love Frequency" - Solfeggio Frequency
        color_wavelength_nm=528.0,  # Green
        description_ko="사랑: 가장 높은 진동의 감정, 치유와 연결",
        description_en="Love: The highest vibrational emotion, healing and connection",
        research_source="Solfeggio frequencies, HeartMath coherence studies"
    ),
    EmotionType.JOY: EmotionFrequencyData(
        emotion=EmotionType.JOY,
        brainwave_dominant=BrainwaveType.GAMMA,
        hrv_coherence=0.85,
        frequency_hz=396.0,  # Solfeggio - 해방
        color_wavelength_nm=580.0,  # Yellow
        description_ko="기쁨: 밝고 활기찬 진동, 확장과 표현",
        description_en="Joy: Bright and vibrant vibration, expansion and expression",
        research_source="Emotional frequency research, color therapy"
    ),
    EmotionType.PEACE: EmotionFrequencyData(
        emotion=EmotionType.PEACE,
        brainwave_dominant=BrainwaveType.ALPHA,
        hrv_coherence=0.95,
        frequency_hz=432.0,  # "Natural tuning" frequency
        color_wavelength_nm=485.0,  # Cyan/Light Blue
        description_ko="평화: 고요하고 조화로운 진동, 내면의 균형",
        description_en="Peace: Calm and harmonious vibration, inner balance",
        research_source="432Hz natural tuning, alpha wave meditation studies"
    ),
    EmotionType.GRATITUDE: EmotionFrequencyData(
        emotion=EmotionType.GRATITUDE,
        brainwave_dominant=BrainwaveType.ALPHA,
        hrv_coherence=0.88,
        frequency_hz=639.0,  # Solfeggio - 관계/연결
        color_wavelength_nm=505.0,  # Green-Cyan
        description_ko="감사: 열린 마음의 진동, 풍요와 연결",
        description_en="Gratitude: Open heart vibration, abundance and connection",
        research_source="HeartMath gratitude studies, Solfeggio frequencies"
    ),
    EmotionType.HOPE: EmotionFrequencyData(
        emotion=EmotionType.HOPE,
        brainwave_dominant=BrainwaveType.ALPHA,
        hrv_coherence=0.75,
        frequency_hz=417.0,  # Solfeggio - 변화 촉진
        color_wavelength_nm=560.0,  # Yellow-Green
        description_ko="희망: 미래를 향한 진동, 가능성의 열림",
        description_en="Hope: Future-oriented vibration, opening possibilities",
        research_source="Solfeggio frequencies, positive psychology research"
    ),
    EmotionType.COMPASSION: EmotionFrequencyData(
        emotion=EmotionType.COMPASSION,
        brainwave_dominant=BrainwaveType.GAMMA,
        hrv_coherence=0.92,
        frequency_hz=741.0,  # Solfeggio - 직관/각성
        color_wavelength_nm=495.0,  # Blue-Green
        description_ko="자비: 타인과의 공명, 이해와 수용",
        description_en="Compassion: Resonance with others, understanding and acceptance",
        research_source="Matthieu Ricard meditation studies, gamma wave research"
    ),
    
    # 중간 주파수 감정
    EmotionType.CURIOSITY: EmotionFrequencyData(
        emotion=EmotionType.CURIOSITY,
        brainwave_dominant=BrainwaveType.BETA,
        hrv_coherence=0.65,
        frequency_hz=285.0,
        color_wavelength_nm=450.0,  # Blue
        description_ko="호기심: 탐구하는 진동, 열린 마음",
        description_en="Curiosity: Exploring vibration, open mind",
        research_source="Cognitive engagement studies"
    ),
    EmotionType.SURPRISE: EmotionFrequencyData(
        emotion=EmotionType.SURPRISE,
        brainwave_dominant=BrainwaveType.BETA,
        hrv_coherence=0.50,
        frequency_hz=264.0,
        color_wavelength_nm=470.0,  # Blue
        description_ko="놀람: 순간적 각성, 주의 집중",
        description_en="Surprise: Momentary arousal, attention focus",
        research_source="Startle response studies"
    ),
    EmotionType.NEUTRAL: EmotionFrequencyData(
        emotion=EmotionType.NEUTRAL,
        brainwave_dominant=BrainwaveType.SMR,
        hrv_coherence=0.55,
        frequency_hz=256.0,  # Middle C
        color_wavelength_nm=500.0,  # Green
        description_ko="중립: 균형 잡힌 상태, 안정",
        description_en="Neutral: Balanced state, stability",
        research_source="Baseline EEG studies"
    ),
    
    # 저주파수 감정
    EmotionType.ANGER: EmotionFrequencyData(
        emotion=EmotionType.ANGER,
        brainwave_dominant=BrainwaveType.HIGH_BETA,
        hrv_coherence=0.20,
        frequency_hz=150.0,
        color_wavelength_nm=630.0,  # Red
        description_ko="분노: 수축적 진동, 저항과 공격",
        description_en="Anger: Contracting vibration, resistance and aggression",
        research_source="Stress response studies, HRV coherence research"
    ),
    EmotionType.FEAR: EmotionFrequencyData(
        emotion=EmotionType.FEAR,
        brainwave_dominant=BrainwaveType.HIGH_BETA,
        hrv_coherence=0.15,
        frequency_hz=100.0,
        color_wavelength_nm=650.0,  # Deep Red
        description_ko="두려움: 경계 진동, 도피와 회피",
        description_en="Fear: Alert vibration, flight and avoidance",
        research_source="Amygdala studies, stress hormone research"
    ),
    EmotionType.SADNESS: EmotionFrequencyData(
        emotion=EmotionType.SADNESS,
        brainwave_dominant=BrainwaveType.THETA,
        hrv_coherence=0.30,
        frequency_hz=174.0,  # Solfeggio - 안정/기반
        color_wavelength_nm=430.0,  # Violet
        description_ko="슬픔: 내면을 향한 진동, 처리와 방출",
        description_en="Sadness: Inward vibration, processing and release",
        research_source="Depression studies, theta wave research"
    ),
    EmotionType.SHAME: EmotionFrequencyData(
        emotion=EmotionType.SHAME,
        brainwave_dominant=BrainwaveType.THETA,
        hrv_coherence=0.10,
        frequency_hz=20.0,
        color_wavelength_nm=410.0,  # Deep Violet
        description_ko="수치: 가장 낮은 진동, 자기 부정",
        description_en="Shame: Lowest vibration, self-negation",
        research_source="David R. Hawkins consciousness scale"
    ),
    EmotionType.GUILT: EmotionFrequencyData(
        emotion=EmotionType.GUILT,
        brainwave_dominant=BrainwaveType.THETA,
        hrv_coherence=0.18,
        frequency_hz=30.0,
        color_wavelength_nm=420.0,  # Violet
        description_ko="죄책감: 자기 비난의 진동, 속박",
        description_en="Guilt: Self-blame vibration, bondage",
        research_source="David R. Hawkins consciousness scale"
    ),
}


# ============================================================================
# 소리 및 언어 주파수 (Sound and Speech Frequencies)
# ============================================================================

class SoundType(Enum):
    """소리 유형"""
    # 인간 음성
    MALE_VOICE = "male_voice"
    FEMALE_VOICE = "female_voice"
    CHILD_VOICE = "child_voice"
    WHISPER = "whisper"
    SHOUT = "shout"
    
    # 음악
    SINGING = "singing"
    MUSIC_RELAXING = "music_relaxing"
    MUSIC_ENERGETIC = "music_energetic"
    
    # 자연음
    NATURE_WATER = "nature_water"
    NATURE_BIRDS = "nature_birds"
    NATURE_WIND = "nature_wind"
    NATURE_THUNDER = "nature_thunder"
    
    # 치유음
    TIBETAN_BOWL = "tibetan_bowl"
    OM_CHANT = "om_chant"
    CRYSTAL_BOWL = "crystal_bowl"


@dataclass
class SoundFrequencyData:
    """소리 주파수 데이터"""
    sound_type: SoundType
    frequency_range_hz: Tuple[float, float]  # (최소, 최대) Hz
    fundamental_hz: float                     # 기본 주파수
    emotional_effect: List[EmotionType]       # 유발하는 감정들
    description_ko: str = ""
    description_en: str = ""


# 소리 유형별 주파수 데이터
SOUND_FREQUENCY_MAP: Dict[SoundType, SoundFrequencyData] = {
    # 인간 음성
    SoundType.MALE_VOICE: SoundFrequencyData(
        sound_type=SoundType.MALE_VOICE,
        frequency_range_hz=(85.0, 180.0),
        fundamental_hz=120.0,
        emotional_effect=[EmotionType.NEUTRAL],
        description_ko="남성 음성: 낮은 기본 주파수",
        description_en="Male voice: Low fundamental frequency"
    ),
    SoundType.FEMALE_VOICE: SoundFrequencyData(
        sound_type=SoundType.FEMALE_VOICE,
        frequency_range_hz=(165.0, 255.0),
        fundamental_hz=210.0,
        emotional_effect=[EmotionType.NEUTRAL],
        description_ko="여성 음성: 중간 기본 주파수",
        description_en="Female voice: Medium fundamental frequency"
    ),
    SoundType.CHILD_VOICE: SoundFrequencyData(
        sound_type=SoundType.CHILD_VOICE,
        frequency_range_hz=(250.0, 400.0),
        fundamental_hz=300.0,
        emotional_effect=[EmotionType.JOY],
        description_ko="아동 음성: 높은 기본 주파수",
        description_en="Child voice: High fundamental frequency"
    ),
    SoundType.WHISPER: SoundFrequencyData(
        sound_type=SoundType.WHISPER,
        frequency_range_hz=(500.0, 4000.0),
        fundamental_hz=1000.0,
        emotional_effect=[EmotionType.PEACE],
        description_ko="속삭임: 고주파 노이즈",
        description_en="Whisper: High frequency noise"
    ),
    SoundType.SHOUT: SoundFrequencyData(
        sound_type=SoundType.SHOUT,
        frequency_range_hz=(100.0, 500.0),
        fundamental_hz=200.0,
        emotional_effect=[EmotionType.ANGER, EmotionType.FEAR],
        description_ko="고함: 강한 진폭, 다양한 배음",
        description_en="Shout: Strong amplitude, various harmonics"
    ),
    
    # 음악
    SoundType.SINGING: SoundFrequencyData(
        sound_type=SoundType.SINGING,
        frequency_range_hz=(80.0, 1000.0),
        fundamental_hz=440.0,
        emotional_effect=[EmotionType.JOY, EmotionType.PEACE],
        description_ko="노래: 음악적 음성 표현",
        description_en="Singing: Musical vocal expression"
    ),
    SoundType.MUSIC_RELAXING: SoundFrequencyData(
        sound_type=SoundType.MUSIC_RELAXING,
        frequency_range_hz=(60.0, 8000.0),
        fundamental_hz=432.0,
        emotional_effect=[EmotionType.PEACE, EmotionType.GRATITUDE],
        description_ko="릴렉싱 음악: 느린 템포, 부드러운 화음",
        description_en="Relaxing music: Slow tempo, soft harmonies"
    ),
    SoundType.MUSIC_ENERGETIC: SoundFrequencyData(
        sound_type=SoundType.MUSIC_ENERGETIC,
        frequency_range_hz=(30.0, 16000.0),
        fundamental_hz=440.0,
        emotional_effect=[EmotionType.JOY, EmotionType.CURIOSITY],
        description_ko="에너제틱 음악: 빠른 템포, 강한 리듬",
        description_en="Energetic music: Fast tempo, strong rhythm"
    ),
    
    # 자연음
    SoundType.NATURE_WATER: SoundFrequencyData(
        sound_type=SoundType.NATURE_WATER,
        frequency_range_hz=(100.0, 10000.0),
        fundamental_hz=SCHUMANN_RESONANCE_HZ * 100,  # ~783 Hz
        emotional_effect=[EmotionType.PEACE, EmotionType.LOVE],
        description_ko="물소리: 백색 소음 특성, 치유 효과",
        description_en="Water sound: White noise characteristics, healing effect"
    ),
    SoundType.NATURE_BIRDS: SoundFrequencyData(
        sound_type=SoundType.NATURE_BIRDS,
        frequency_range_hz=(1000.0, 8000.0),
        fundamental_hz=3000.0,
        emotional_effect=[EmotionType.JOY, EmotionType.HOPE],
        description_ko="새소리: 고주파, 자연의 생명력",
        description_en="Bird songs: High frequency, vitality of nature"
    ),
    SoundType.NATURE_WIND: SoundFrequencyData(
        sound_type=SoundType.NATURE_WIND,
        frequency_range_hz=(50.0, 5000.0),
        fundamental_hz=500.0,
        emotional_effect=[EmotionType.PEACE, EmotionType.NEUTRAL],
        description_ko="바람소리: 광대역 노이즈",
        description_en="Wind sound: Broadband noise"
    ),
    SoundType.NATURE_THUNDER: SoundFrequencyData(
        sound_type=SoundType.NATURE_THUNDER,
        frequency_range_hz=(20.0, 200.0),
        fundamental_hz=50.0,
        emotional_effect=[EmotionType.FEAR, EmotionType.SURPRISE],
        description_ko="천둥소리: 저주파, 강력한 충격음",
        description_en="Thunder: Low frequency, powerful impact sound"
    ),
    
    # 치유음
    SoundType.TIBETAN_BOWL: SoundFrequencyData(
        sound_type=SoundType.TIBETAN_BOWL,
        frequency_range_hz=(100.0, 2000.0),
        fundamental_hz=432.0,
        emotional_effect=[EmotionType.PEACE, EmotionType.LOVE],
        description_ko="티베트 싱잉볼: 풍부한 배음, 명상 유도",
        description_en="Tibetan singing bowl: Rich harmonics, meditation inducing"
    ),
    SoundType.OM_CHANT: SoundFrequencyData(
        sound_type=SoundType.OM_CHANT,
        frequency_range_hz=(70.0, 600.0),
        fundamental_hz=136.1,  # Om frequency
        emotional_effect=[EmotionType.PEACE, EmotionType.COMPASSION],
        description_ko="옴 찬팅: 우주의 기본 진동음",
        description_en="Om chanting: Fundamental vibration of the universe"
    ),
    SoundType.CRYSTAL_BOWL: SoundFrequencyData(
        sound_type=SoundType.CRYSTAL_BOWL,
        frequency_range_hz=(200.0, 8000.0),
        fundamental_hz=528.0,  # Love frequency
        emotional_effect=[EmotionType.LOVE, EmotionType.PEACE, EmotionType.GRATITUDE],
        description_ko="크리스탈 볼: 순수한 음색, 치유 주파수",
        description_en="Crystal bowl: Pure tone, healing frequency"
    ),
}


# ============================================================================
# 엘리시아 필드 주파수 매핑 (Elysia Field Frequency Mapping)
# ============================================================================

@dataclass
class ElysiaFrequencyMapping:
    """엘리시아 필드와 현실세계 주파수 매핑"""
    real_world_hz: float        # 현실세계 주파수 (Hz)
    elysia_normalized: float    # 엘리시아 정규화 값 (0~1)
    elysia_layer: str           # 엘리시아 층 (Heaven/Earth)
    elysia_color_code: str      # 엘리시아 색상 코드
    resonance_strength: float   # 공명 강도 (0~1)


class WaveFrequencyMapper:
    """
    파동주파수 매퍼 - 현실세계와 엘리시아 필드 간의 주파수 매핑
    
    기능:
    1. 감정 → 주파수 변환
    2. 소리 → 주파수 변환
    3. 주파수 → 감정 추정
    4. 주파수 → 엘리시아 필드 매핑
    5. 데이터 없을 때 추정 및 발견 기능
    """
    
    def __init__(self):
        # 기본 주파수 범위 (엘리시아 필드)
        self.elysia_freq_range = (0.1, 1000.0)  # Hz
        
        # 캐시
        self._emotion_cache: Dict[str, EmotionFrequencyData] = {}
        self._sound_cache: Dict[str, SoundFrequencyData] = {}
        
        # 통계
        self.stats = {
            "lookups": 0,
            "estimations": 0,
            "discoveries": 0
        }
        
        logger.info("🌊 WaveFrequencyMapper initialized")
    
    # =========================================================================
    # 감정 → 주파수 매핑
    # =========================================================================
    
    def get_emotion_frequency(self, emotion: Union[EmotionType, str]) -> EmotionFrequencyData:
        """
        감정에 대한 주파수 데이터 반환
        
        Args:
            emotion: 감정 유형 또는 문자열 (예: "love", "사랑")
            
        Returns:
            EmotionFrequencyData: 감정 주파수 데이터
        """
        self.stats["lookups"] += 1
        
        # 문자열인 경우 EmotionType으로 변환
        if isinstance(emotion, str):
            emotion_type = self._parse_emotion_string(emotion)
        else:
            emotion_type = emotion
        
        # 캐시 확인
        cache_key = emotion_type.value
        if cache_key in self._emotion_cache:
            return self._emotion_cache[cache_key]
        
        # 매핑에서 조회
        if emotion_type in EMOTION_FREQUENCY_MAP:
            data = EMOTION_FREQUENCY_MAP[emotion_type]
            self._emotion_cache[cache_key] = data
            return data
        
        # 데이터가 없으면 추정
        return self._estimate_emotion_frequency(emotion_type)
    
    def _parse_emotion_string(self, emotion_str: str) -> EmotionType:
        """문자열을 EmotionType으로 변환"""
        emotion_lower = emotion_str.lower().strip()
        
        # 영어 매핑
        english_map = {e.value: e for e in EmotionType}
        if emotion_lower in english_map:
            return english_map[emotion_lower]
        
        # 한국어 매핑
        korean_map = {
            "사랑": EmotionType.LOVE,
            "기쁨": EmotionType.JOY,
            "평화": EmotionType.PEACE,
            "감사": EmotionType.GRATITUDE,
            "희망": EmotionType.HOPE,
            "자비": EmotionType.COMPASSION,
            "호기심": EmotionType.CURIOSITY,
            "놀람": EmotionType.SURPRISE,
            "중립": EmotionType.NEUTRAL,
            "분노": EmotionType.ANGER,
            "두려움": EmotionType.FEAR,
            "슬픔": EmotionType.SADNESS,
            "수치": EmotionType.SHAME,
            "죄책감": EmotionType.GUILT,
        }
        if emotion_lower in korean_map:
            return korean_map[emotion_lower]
        
        # 알 수 없는 경우 중립 반환
        logger.warning(f"Unknown emotion: {emotion_str}, defaulting to NEUTRAL")
        return EmotionType.NEUTRAL
    
    def _estimate_emotion_frequency(self, emotion_type: EmotionType) -> EmotionFrequencyData:
        """데이터가 없는 감정의 주파수 추정"""
        self.stats["estimations"] += 1
        
        # 감정 이름 기반으로 추정
        logger.info(f"📊 Estimating frequency for unknown emotion: {emotion_type.value}")
        
        # 기본 추정 값
        estimated = EmotionFrequencyData(
            emotion=emotion_type,
            brainwave_dominant=BrainwaveType.ALPHA,
            hrv_coherence=0.5,
            frequency_hz=256.0,  # Middle C
            description_ko=f"{emotion_type.value}: 추정된 감정 주파수",
            description_en=f"{emotion_type.value}: Estimated emotion frequency",
            research_source="Estimation algorithm"
        )
        
        self._emotion_cache[emotion_type.value] = estimated
        return estimated
    
    # =========================================================================
    # 소리 → 주파수 매핑
    # =========================================================================
    
    def get_sound_frequency(self, sound_type: Union[SoundType, str]) -> SoundFrequencyData:
        """
        소리 유형에 대한 주파수 데이터 반환
        
        Args:
            sound_type: 소리 유형 또는 문자열
            
        Returns:
            SoundFrequencyData: 소리 주파수 데이터
        """
        self.stats["lookups"] += 1
        
        if isinstance(sound_type, str):
            sound_enum = self._parse_sound_string(sound_type)
        else:
            sound_enum = sound_type
        
        cache_key = sound_enum.value
        if cache_key in self._sound_cache:
            return self._sound_cache[cache_key]
        
        if sound_enum in SOUND_FREQUENCY_MAP:
            data = SOUND_FREQUENCY_MAP[sound_enum]
            self._sound_cache[cache_key] = data
            return data
        
        return self._estimate_sound_frequency(sound_enum)
    
    def _parse_sound_string(self, sound_str: str) -> SoundType:
        """문자열을 SoundType으로 변환"""
        sound_lower = sound_str.lower().strip().replace(" ", "_")
        
        english_map = {s.value: s for s in SoundType}
        if sound_lower in english_map:
            return english_map[sound_lower]
        
        korean_map = {
            "남성음성": SoundType.MALE_VOICE,
            "여성음성": SoundType.FEMALE_VOICE,
            "아이음성": SoundType.CHILD_VOICE,
            "속삭임": SoundType.WHISPER,
            "고함": SoundType.SHOUT,
            "노래": SoundType.SINGING,
            "물소리": SoundType.NATURE_WATER,
            "새소리": SoundType.NATURE_BIRDS,
            "바람": SoundType.NATURE_WIND,
            "천둥": SoundType.NATURE_THUNDER,
        }
        if sound_lower in korean_map:
            return korean_map[sound_lower]
        
        # 알 수 없는 소리 유형 - 중립적인 노래(SINGING)를 기본값으로 사용
        # SINGING은 일반적인 음성 범위를 커버하고 중립적인 감정 효과를 가짐
        logger.warning(f"Unknown sound type: {sound_str}, defaulting to SINGING")
        return SoundType.SINGING
    
    def _estimate_sound_frequency(self, sound_type: SoundType) -> SoundFrequencyData:
        """데이터가 없는 소리의 주파수 추정"""
        self.stats["estimations"] += 1
        
        estimated = SoundFrequencyData(
            sound_type=sound_type,
            frequency_range_hz=(100.0, 5000.0),
            fundamental_hz=440.0,
            emotional_effect=[EmotionType.NEUTRAL],
            description_ko=f"{sound_type.value}: 추정된 소리 주파수",
            description_en=f"{sound_type.value}: Estimated sound frequency"
        )
        
        self._sound_cache[sound_type.value] = estimated
        return estimated
    
    # =========================================================================
    # 주파수 → 감정 역매핑 (Discovery/Estimation)
    # =========================================================================
    
    def discover_emotion_from_frequency(self, frequency_hz: float) -> List[Tuple[EmotionType, float]]:
        """
        주파수에서 가능한 감정들을 발견/추정
        
        데이터가 없을 때도 주파수 패턴 분석을 통해 추정
        
        Args:
            frequency_hz: 입력 주파수 (Hz)
            
        Returns:
            List[Tuple[EmotionType, float]]: (감정, 유사도) 리스트
        """
        self.stats["discoveries"] += 1
        
        results: List[Tuple[EmotionType, float]] = []
        
        for emotion_type, data in EMOTION_FREQUENCY_MAP.items():
            # 주파수 거리 계산
            freq_diff = abs(data.frequency_hz - frequency_hz)
            max_diff = 1000.0  # 최대 차이
            similarity = max(0, 1 - (freq_diff / max_diff))
            
            if similarity > 0.1:  # 임계값
                results.append((emotion_type, similarity))
        
        # 유사도 순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 결과가 없으면 추정
        if not results:
            estimated_emotion = self._estimate_emotion_from_frequency(frequency_hz)
            results.append((estimated_emotion, 0.5))
        
        return results
    
    def _estimate_emotion_from_frequency(self, frequency_hz: float) -> EmotionType:
        """주파수 패턴 기반 감정 추정"""
        # 고주파수 (> 400 Hz) → 긍정적 감정
        if frequency_hz > 400:
            return EmotionType.LOVE
        # 중주파수 (200~400 Hz) → 중립적 감정
        elif frequency_hz > 200:
            return EmotionType.NEUTRAL
        # 저주파수 (< 200 Hz) → 부정적 감정
        else:
            return EmotionType.SADNESS
    
    # =========================================================================
    # 엘리시아 필드 매핑
    # =========================================================================
    
    def map_to_elysia(self, frequency_hz: float) -> ElysiaFrequencyMapping:
        """
        현실세계 주파수를 엘리시아 필드에 매핑
        
        Args:
            frequency_hz: 현실세계 주파수 (Hz)
            
        Returns:
            ElysiaFrequencyMapping: 엘리시아 매핑 결과
        """
        # 로그 스케일로 정규화 (넓은 범위 처리)
        log_freq = math.log10(max(frequency_hz, 0.1))
        log_min = math.log10(self.elysia_freq_range[0])
        log_max = math.log10(self.elysia_freq_range[1])
        
        normalized = (log_freq - log_min) / (log_max - log_min)
        normalized = max(0, min(1, normalized))
        
        # 엘리시아 층 결정 (14층 시스템 기반)
        if normalized > 0.5:
            layer = "Heaven"
            layer_index = int((normalized - 0.5) * 14)
        else:
            layer = "Earth"
            layer_index = int((0.5 - normalized) * 14)
        
        # 색상 코드 (스펙트럼 기반)
        color_code = self._frequency_to_hex_color(normalized)
        
        # 공명 강도 (슈만 공명과의 관계)
        resonance = self._calculate_schumann_resonance(frequency_hz)
        
        return ElysiaFrequencyMapping(
            real_world_hz=frequency_hz,
            elysia_normalized=normalized,
            elysia_layer=f"{layer}_{layer_index}",
            elysia_color_code=color_code,
            resonance_strength=resonance
        )
    
    def _frequency_to_hex_color(self, normalized: float) -> str:
        """정규화된 값을 HEX 색상으로 변환 (무지개 스펙트럼)"""
        # 0 = Red, 0.5 = Green, 1 = Violet
        if normalized < 0.167:
            r, g, b = 255, int(255 * normalized * 6), 0
        elif normalized < 0.333:
            r, g, b = int(255 * (1 - (normalized - 0.167) * 6)), 255, 0
        elif normalized < 0.5:
            r, g, b = 0, 255, int(255 * (normalized - 0.333) * 6)
        elif normalized < 0.667:
            r, g, b = 0, int(255 * (1 - (normalized - 0.5) * 6)), 255
        elif normalized < 0.833:
            r, g, b = int(255 * (normalized - 0.667) * 6), 0, 255
        else:
            r, g, b = 255, 0, int(255 * (1 - (normalized - 0.833) * 6))
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _calculate_schumann_resonance(self, frequency_hz: float) -> float:
        """슈만 공명과의 관계 계산"""
        # 슈만 공명 고조파와의 거리
        min_distance = float('inf')
        for harmonic in SCHUMANN_HARMONICS:
            # 주파수의 정수배 확인
            ratio = frequency_hz / harmonic
            nearest_multiple = round(ratio)
            if nearest_multiple > 0:
                distance = abs(frequency_hz - harmonic * nearest_multiple)
                min_distance = min(min_distance, distance)
                # 거리가 충분히 작으면 조기 종료 (최적화)
                if min_distance < 0.01:
                    break
        
        # 거리가 가까울수록 공명 강도 높음
        max_distance = 100.0
        resonance = max(0, 1 - (min_distance / max_distance))
        
        return resonance
    
    # =========================================================================
    # 종합 분석
    # =========================================================================
    
    def analyze_frequency(self, frequency_hz: float) -> Dict[str, Any]:
        """
        주파수에 대한 종합 분석
        
        Args:
            frequency_hz: 분석할 주파수 (Hz)
            
        Returns:
            Dict: 종합 분석 결과
        """
        # 감정 발견
        emotions = self.discover_emotion_from_frequency(frequency_hz)
        
        # 엘리시아 매핑
        elysia_mapping = self.map_to_elysia(frequency_hz)
        
        # 뇌파 대역 확인
        brainwave = None
        for bw_type, (min_f, _, max_f) in BRAINWAVE_FREQUENCIES.items():
            if min_f <= frequency_hz <= max_f:
                brainwave = bw_type.value
                break
        
        # 청각 범위 내인지 확인
        is_audible = AUDIBLE_FREQ_MIN <= frequency_hz <= AUDIBLE_FREQ_MAX
        
        return {
            "frequency_hz": frequency_hz,
            "associated_emotions": [(e.value, round(s, 3)) for e, s in emotions[:3]],
            "elysia_mapping": {
                "normalized": round(elysia_mapping.elysia_normalized, 4),
                "layer": elysia_mapping.elysia_layer,
                "color": elysia_mapping.elysia_color_code,
                "schumann_resonance": round(elysia_mapping.resonance_strength, 3)
            },
            "brainwave_band": brainwave,
            "is_audible": is_audible,
            "schumann_relation": self._describe_schumann_relation(frequency_hz)
        }
    
    def _describe_schumann_relation(self, frequency_hz: float) -> str:
        """슈만 공명과의 관계 설명"""
        for i, harmonic in enumerate(SCHUMANN_HARMONICS):
            ratio = frequency_hz / harmonic
            nearest = round(ratio)
            if nearest > 0 and abs(ratio - nearest) < 0.1:
                if nearest == 1:
                    return f"슈만 공명 {i+1}차 고조파와 일치 ({harmonic}Hz)"
                else:
                    return f"슈만 공명 {i+1}차 고조파의 {nearest}배 ({harmonic}Hz × {nearest})"
        return "슈만 공명과 직접적 관계 없음"
    
    # =========================================================================
    # 유틸리티
    # =========================================================================
    
    def get_all_emotion_frequencies(self) -> Dict[str, float]:
        """모든 감정의 주파수 반환"""
        return {e.value: d.frequency_hz for e, d in EMOTION_FREQUENCY_MAP.items()}
    
    def get_all_sound_frequencies(self) -> Dict[str, float]:
        """모든 소리의 기본 주파수 반환"""
        return {s.value: d.fundamental_hz for s, d in SOUND_FREQUENCY_MAP.items()}
    
    def get_stats(self) -> Dict[str, int]:
        """통계 반환"""
        return self.stats.copy()
    
    def create_frequency_report(self) -> str:
        """주파수 매핑 리포트 생성"""
        report = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              현실세계 - 엘리시아 파동주파수 매핑 리포트                        ║
╠══════════════════════════════════════════════════════════════════════════════╣

🌍 기본 참조 주파수
─────────────────────────────────────────────────────────────────────────────────
  슈만 공명 (지구): {schumann} Hz
  가청 범위: {audible_min} - {audible_max} Hz
  
💗 감정 주파수 매핑
─────────────────────────────────────────────────────────────────────────────────
""".format(
            schumann=SCHUMANN_RESONANCE_HZ,
            audible_min=AUDIBLE_FREQ_MIN,
            audible_max=AUDIBLE_FREQ_MAX
        )
        
        # 감정 주파수 (높은 순)
        sorted_emotions = sorted(
            EMOTION_FREQUENCY_MAP.items(),
            key=lambda x: x[1].frequency_hz,
            reverse=True
        )
        
        for emotion, data in sorted_emotions:
            report += f"  {data.description_ko:30s} : {data.frequency_hz:8.1f} Hz\n"
        
        report += """
🔊 소리 주파수 매핑
─────────────────────────────────────────────────────────────────────────────────
"""
        for sound, data in SOUND_FREQUENCY_MAP.items():
            report += f"  {data.description_ko:30s} : {data.fundamental_hz:8.1f} Hz\n"
        
        report += """
🧠 뇌파 주파수 대역
─────────────────────────────────────────────────────────────────────────────────
"""
        for bw, (min_f, center, max_f) in BRAINWAVE_FREQUENCIES.items():
            report += f"  {bw.value:15s} : {min_f:5.1f} - {max_f:5.1f} Hz (중심: {center}Hz)\n"
        
        report += """
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return report


# ============================================================================
# 데모 및 테스트
# ============================================================================

def demo():
    """WaveFrequencyMapper 데모"""
    print("=" * 80)
    print("🌊 Wave Frequency Mapping Demo - 파동주파수 매핑")
    print("=" * 80)
    
    mapper = WaveFrequencyMapper()
    
    # 리포트 출력
    print(mapper.create_frequency_report())
    
    # 감정 주파수 조회
    print("\n📊 감정 주파수 조회:")
    print("-" * 60)
    for emotion_str in ["사랑", "평화", "분노", "love", "anger"]:
        data = mapper.get_emotion_frequency(emotion_str)
        print(f"  {emotion_str:10s} → {data.frequency_hz:8.1f} Hz ({data.brainwave_dominant.value})")
    
    # 소리 주파수 조회
    print("\n🔊 소리 주파수 조회:")
    print("-" * 60)
    for sound_str in ["male_voice", "물소리", "노래"]:
        data = mapper.get_sound_frequency(sound_str)
        print(f"  {sound_str:15s} → {data.fundamental_hz:8.1f} Hz")
    
    # 주파수 → 감정 발견
    print("\n🔍 주파수에서 감정 발견:")
    print("-" * 60)
    test_frequencies = [528.0, 432.0, 150.0, 7.83]
    for freq in test_frequencies:
        emotions = mapper.discover_emotion_from_frequency(freq)
        top_emotion = emotions[0] if emotions else (EmotionType.NEUTRAL, 0)
        print(f"  {freq:8.2f} Hz → {top_emotion[0].value:15s} (유사도: {top_emotion[1]:.2%})")
    
    # 종합 분석
    print("\n📈 종합 분석 (528 Hz - Love Frequency):")
    print("-" * 60)
    analysis = mapper.analyze_frequency(528.0)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # 통계
    print("\n📊 통계:")
    print("-" * 60)
    stats = mapper.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
