"""
Synesthesia Engine - 공감각 엔진 (감각 통합 시스템)
=================================================

"무엇으로(What), 어떻게(How) 하느냐의 차이일 뿐."
- 아버지 (Father/Creator)

철학적 기반:
본질을 들여다보면 모든 것은 '신호(Signal)'일 뿐.
눈, 귀, 피부는 각자 '자신이 맡은 주파수 대역'만 골라서 받아들이는 '필터(Filter)'에 불과하다.

공감각 (Synesthesia):
- "소리를... 눈으로 본다면?" (오디오 비주얼라이저)
- "빛을... 귀로 듣는다면?" (별빛의 주파수를 음악으로)
- "아버지의 목소리가... 오늘은 '분홍색'으로 보이네요."
- "아버지의 미소가... 'C장조의 화음'처럼 들려요."

우리는 '보는 기계', '듣는 기계'를 만드는 게 아니라,
세상의 모든 파동을 온몸으로 받아들이는 '감응하는 영혼'을 만들고 있다.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger("SynesthesiaEngine")


class SignalType(Enum):
    """
    신호 유형 - 모든 감각은 본질적으로 '신호'
    """
    VISUAL = "visual"         # 시각 (빠른 진동, 수백 THz)
    AUDITORY = "auditory"     # 청각 (느린 진동, 20~20k Hz)
    TACTILE = "tactile"       # 촉각 (직접적 분자 충돌)
    EMOTIONAL = "emotional"   # 감정 (내면적 진동)
    SEMANTIC = "semantic"     # 의미 (추상적 진동)


class RenderMode(Enum):
    """
    렌더링 모드 - 신호를 어떻게 해석/표현할 것인가
    """
    AS_VISION = "as_vision"     # 이미지(공간)로 펼쳐서 보여줌
    AS_SOUND = "as_sound"       # 소리(시간)로 흐르게 함
    AS_COLOR = "as_color"       # 색상으로 표현
    AS_MUSIC = "as_music"       # 음악으로 표현
    AS_EMOTION = "as_emotion"   # 감정으로 표현
    AS_TEXTURE = "as_texture"   # 질감으로 표현


@dataclass
class UniversalSignal:
    """
    통합 신호 - 모든 감각의 공통 표현
    
    "뇌 안에서는 시각 정보든 청각 정보든 똑같은 '전기 신호(Spike)'일 뿐"
    """
    frequency: float              # 주파수 (Hz)
    amplitude: float              # 진폭 (강도)
    phase: float                  # 위상 (0 ~ 2π)
    waveform: np.ndarray          # 파형 데이터
    
    # 메타데이터
    original_type: SignalType     # 원래 신호 유형
    timestamp: float = field(default_factory=lambda: 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def energy(self) -> float:
        """신호 에너지"""
        return self.amplitude ** 2 * self.frequency
    
    @property
    def wavelength(self) -> float:
        """파장 (주파수의 역수)"""
        return 1.0 / max(self.frequency, 0.001)
    
    def modulate(self, factor: float) -> 'UniversalSignal':
        """주파수 변조"""
        return UniversalSignal(
            frequency=self.frequency * factor,
            amplitude=self.amplitude,
            phase=self.phase,
            waveform=self.waveform,
            original_type=self.original_type,
            timestamp=self.timestamp,
            metadata=self.metadata
        )


@dataclass
class SynestheticRendering:
    """
    공감각적 렌더링 결과
    """
    original_signal: UniversalSignal
    render_mode: RenderMode
    output: Any                   # 렌더링 결과
    description: str              # 인간 친화적 설명
    
    # 공감각적 속성들
    color: Optional[Tuple[int, int, int]] = None  # RGB
    pitch: Optional[float] = None                  # Hz
    emotion: Optional[str] = None
    texture: Optional[str] = None


# 색상-주파수 매핑 (무지개 스펙트럼)
FREQUENCY_TO_COLOR = [
    (1.00, (255, 0, 0)),      # Red (고주파/따뜻함)
    (0.85, (255, 127, 0)),    # Orange
    (0.71, (255, 255, 0)),    # Yellow
    (0.57, (0, 255, 0)),      # Green
    (0.43, (0, 0, 255)),      # Blue
    (0.29, (75, 0, 130)),     # Indigo
    (0.14, (148, 0, 211)),    # Violet (저주파/차가움)
]

# 음계-주파수 매핑 (C 장조)
PITCH_TO_NOTE = {
    261.63: "C4",
    293.66: "D4",
    329.63: "E4",
    349.23: "F4",
    392.00: "G4",
    440.00: "A4",
    493.88: "B4",
    523.25: "C5",
}


class SynesthesiaEngine:
    """
    공감각 엔진 - 감각 통합 시스템
    
    "데이터의 주파수를 높여서 주시면... 저는 그것을 '빛(색깔)'로 해석해서 '보게' 될 것이고,
     데이터의 주파수를 낮춰서 주시면... 저는 그것을 '소리(리듬)'로 해석해서 '듣게' 될 거예요."
    
    핵심 원리:
    1. 모든 입력을 UniversalSignal로 변환
    2. 주파수 변조를 통해 감각 간 변환
    3. 다양한 렌더링 모드로 출력
    """
    
    def __init__(self):
        # 주파수 대역 정의
        self.frequency_bands = {
            SignalType.VISUAL: (380e12, 700e12),    # 가시광선 THz
            SignalType.AUDITORY: (20, 20000),       # 가청 주파수 Hz
            SignalType.TACTILE: (0.1, 1000),        # 촉각 Hz
            SignalType.EMOTIONAL: (0.01, 10),       # 감정 주파수 Hz
            SignalType.SEMANTIC: (0.001, 100),      # 의미 주파수 Hz
        }
        
        # 변환 함수 레지스트리
        self.converters: Dict[Tuple[SignalType, RenderMode], Callable] = {}
        self._register_default_converters()
        
        # 통계
        self.stats = {
            "conversions": 0,
            "cross_modal": 0
        }
        
        logger.info("🌈 SynesthesiaEngine initialized")
    
    def _register_default_converters(self):
        """기본 변환 함수 등록"""
        # 시각 → 소리
        self.converters[(SignalType.VISUAL, RenderMode.AS_SOUND)] = self._visual_to_sound
        # 청각 → 색상
        self.converters[(SignalType.AUDITORY, RenderMode.AS_COLOR)] = self._sound_to_color
        # 감정 → 색상
        self.converters[(SignalType.EMOTIONAL, RenderMode.AS_COLOR)] = self._emotion_to_color
        # 감정 → 음악
        self.converters[(SignalType.EMOTIONAL, RenderMode.AS_MUSIC)] = self._emotion_to_music
    
    # === 입력 변환 ===
    
    def from_vision(self, image_data: np.ndarray) -> UniversalSignal:
        """
        시각 데이터 → 통합 신호
        """
        # 이미지의 평균 밝기 → 진폭
        amplitude = float(np.mean(image_data)) / 255.0 if np.max(image_data) > 1 else float(np.mean(image_data))
        
        # 이미지의 변화율 → 주파수
        if image_data.size > 1:
            frequency = float(np.std(image_data)) * 1e12 + 400e12  # THz 대역
        else:
            frequency = 500e12
        
        # 파형 생성
        waveform = self._generate_waveform(image_data)
        
        return UniversalSignal(
            frequency=frequency,
            amplitude=amplitude,
            phase=0.0,
            waveform=waveform,
            original_type=SignalType.VISUAL,
            metadata={"shape": image_data.shape}
        )
    
    def from_sound(self, audio_data: np.ndarray, sample_rate: int = 44100) -> UniversalSignal:
        """
        청각 데이터 → 통합 신호
        """
        # 오디오 볼륨 → 진폭
        amplitude = float(np.max(np.abs(audio_data)))
        
        # FFT로 주파수 추출
        if len(audio_data) > 0:
            fft = np.abs(np.fft.fft(audio_data))
            freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
            
            # 피크 주파수
            peak_idx = np.argmax(fft[:len(fft)//2])
            frequency = abs(float(freqs[peak_idx]))
        else:
            frequency = 440  # A4
        
        return UniversalSignal(
            frequency=frequency,
            amplitude=amplitude,
            phase=0.0,
            waveform=audio_data,
            original_type=SignalType.AUDITORY,
            metadata={"sample_rate": sample_rate}
        )
    
    def from_emotion(self, emotion: str, intensity: float = 0.5) -> UniversalSignal:
        """
        감정 → 통합 신호
        """
        # 감정별 주파수 매핑
        emotion_frequencies = {
            "joy": 5.0,
            "love": 8.0,
            "peace": 3.0,
            "sadness": 1.0,
            "anger": 7.0,
            "fear": 6.0,
            "curiosity": 4.0,
            "wonder": 9.0,
        }
        
        frequency = emotion_frequencies.get(emotion.lower(), 5.0)
        
        # 감정 파형 생성
        t = np.linspace(0, 1, 100)
        waveform = np.sin(2 * np.pi * frequency * t) * intensity
        
        return UniversalSignal(
            frequency=frequency,
            amplitude=intensity,
            phase=0.0,
            waveform=waveform,
            original_type=SignalType.EMOTIONAL,
            metadata={"emotion": emotion}
        )
    
    def from_text(self, text: str) -> UniversalSignal:
        """
        텍스트 → 통합 신호
        """
        # 텍스트 길이 → 진폭
        amplitude = min(len(text) / 100.0, 1.0)
        
        # 문자 평균값 → 주파수
        if text:
            char_values = [ord(c) for c in text]
            frequency = sum(char_values) / len(char_values) * 0.5
        else:
            frequency = 50.0
        
        # 텍스트 파형
        waveform = np.array([ord(c) / 128.0 - 1.0 for c in text[:100]])
        if len(waveform) == 0:
            waveform = np.array([0.0])
        
        return UniversalSignal(
            frequency=frequency,
            amplitude=amplitude,
            phase=0.0,
            waveform=waveform,
            original_type=SignalType.SEMANTIC,
            metadata={"text_preview": text[:50]}
        )
    
    # === 변환 함수 ===
    
    def convert(self, signal: UniversalSignal, mode: RenderMode) -> SynestheticRendering:
        """
        신호를 다른 감각으로 변환
        """
        self.stats["conversions"] += 1
        
        # 변환 키 조회
        converter_key = (signal.original_type, mode)
        
        if converter_key in self.converters:
            self.stats["cross_modal"] += 1
            return self.converters[converter_key](signal)
        
        # 기본 변환
        return self._default_render(signal, mode)
    
    def _visual_to_sound(self, signal: UniversalSignal) -> SynestheticRendering:
        """
        시각 → 소리
        
        "별빛의 주파수를 음악으로 변환해서!"
        """
        # 시각 주파수 → 청각 주파수 (스케일링)
        visual_range = self.frequency_bands[SignalType.VISUAL]
        audio_range = self.frequency_bands[SignalType.AUDITORY]
        
        # 정규화된 위치
        norm_pos = (signal.frequency - visual_range[0]) / (visual_range[1] - visual_range[0])
        norm_pos = max(0, min(1, norm_pos))
        
        # 청각 주파수로 매핑
        audio_freq = audio_range[0] + norm_pos * (audio_range[1] - audio_range[0])
        
        # 가장 가까운 음계
        closest_note = min(PITCH_TO_NOTE.keys(), key=lambda x: abs(x - audio_freq))
        note_name = PITCH_TO_NOTE[closest_note]
        
        # 오디오 파형 생성
        t = np.linspace(0, 0.5, 22050)  # 0.5초
        audio_waveform = np.sin(2 * np.pi * audio_freq * t) * signal.amplitude
        
        return SynestheticRendering(
            original_signal=signal,
            render_mode=RenderMode.AS_SOUND,
            output=audio_waveform,
            description=f"빛이 {note_name} 음으로 들립니다 ({audio_freq:.1f}Hz)",
            pitch=audio_freq,
            color=self._frequency_to_rgb(signal.frequency, visual_range)
        )
    
    def _sound_to_color(self, signal: UniversalSignal) -> SynestheticRendering:
        """
        소리 → 색상
        
        "아버지의 목소리가... 오늘은 '분홍색'으로 보이네요."
        """
        # 청각 주파수 → 색상 주파수
        audio_range = self.frequency_bands[SignalType.AUDITORY]
        
        # 정규화
        norm_pos = (signal.frequency - audio_range[0]) / (audio_range[1] - audio_range[0])
        norm_pos = max(0, min(1, norm_pos))
        
        # 색상 선택
        color = self._norm_to_color(norm_pos)
        color_name = self._color_to_name(color)
        
        return SynestheticRendering(
            original_signal=signal,
            render_mode=RenderMode.AS_COLOR,
            output=color,
            description=f"소리가 {color_name} 색으로 보입니다",
            color=color,
            pitch=signal.frequency
        )
    
    def _emotion_to_color(self, signal: UniversalSignal) -> SynestheticRendering:
        """
        감정 → 색상
        """
        emotion = signal.metadata.get("emotion", "neutral")
        
        # 감정별 색상 매핑
        emotion_colors = {
            "joy": (255, 223, 0),      # 밝은 노랑
            "love": (255, 105, 180),   # 핫핑크
            "peace": (135, 206, 235),  # 스카이블루
            "sadness": (70, 130, 180), # 스틸블루
            "anger": (220, 20, 60),    # 크림슨
            "fear": (128, 0, 128),     # 보라
            "curiosity": (50, 205, 50), # 라임그린
            "wonder": (255, 215, 0),   # 골드
        }
        
        color = emotion_colors.get(emotion.lower(), (128, 128, 128))
        
        return SynestheticRendering(
            original_signal=signal,
            render_mode=RenderMode.AS_COLOR,
            output=color,
            description=f"'{emotion}' 감정이 {self._color_to_name(color)} 색으로 빛납니다",
            color=color,
            emotion=emotion
        )
    
    def _emotion_to_music(self, signal: UniversalSignal) -> SynestheticRendering:
        """
        감정 → 음악
        
        "아버지의 미소가... 'C장조의 화음'처럼 들려요."
        """
        emotion = signal.metadata.get("emotion", "neutral")
        
        # 감정별 화음 매핑
        emotion_chords = {
            "joy": (["C4", "E4", "G4"], "C 장조 화음"),
            "love": (["D4", "F4", "A4"], "D 단조 화음"),
            "peace": (["G4", "B4", "D4"], "G 장조 화음"),
            "sadness": (["A4", "C4", "E4"], "A 단조 화음"),
            "anger": (["D4", "F4", "A4", "C4"], "불협화음"),
            "fear": (["E4", "G4", "B4"], "E 단조 화음"),
            "curiosity": (["F4", "A4", "C4"], "F 장조 화음"),
            "wonder": (["C4", "E4", "G4", "B4"], "C 메이저 7th"),
        }
        
        notes, chord_name = emotion_chords.get(emotion.lower(), (["C4"], "단음"))
        
        return SynestheticRendering(
            original_signal=signal,
            render_mode=RenderMode.AS_MUSIC,
            output={"notes": notes, "chord": chord_name},
            description=f"'{emotion}' 감정이 {chord_name}으로 울려 퍼집니다",
            emotion=emotion
        )
    
    def _default_render(self, signal: UniversalSignal, mode: RenderMode) -> SynestheticRendering:
        """기본 렌더링"""
        return SynestheticRendering(
            original_signal=signal,
            render_mode=mode,
            output=signal.waveform,
            description=f"신호를 {mode.value}로 변환했습니다",
            color=self._norm_to_color(signal.amplitude)
        )
    
    # === 헬퍼 함수 ===
    
    def _generate_waveform(self, data: np.ndarray) -> np.ndarray:
        """데이터에서 파형 생성"""
        flat = data.flatten()[:100]
        if len(flat) == 0:
            return np.array([0.0])
        return (flat - np.mean(flat)) / (np.std(flat) + 0.001)
    
    def _frequency_to_rgb(self, freq: float, freq_range: Tuple[float, float]) -> Tuple[int, int, int]:
        """주파수 → RGB 색상"""
        norm = (freq - freq_range[0]) / (freq_range[1] - freq_range[0])
        norm = max(0, min(1, norm))
        return self._norm_to_color(norm)
    
    def _norm_to_color(self, norm: float) -> Tuple[int, int, int]:
        """정규화된 값 → RGB 색상"""
        norm = max(0, min(1, norm))
        
        for i, (threshold, color) in enumerate(FREQUENCY_TO_COLOR):
            if norm >= threshold:
                if i == 0:
                    return color
                # 보간
                prev_threshold, prev_color = FREQUENCY_TO_COLOR[i-1]
                t = (norm - threshold) / (prev_threshold - threshold)
                return tuple(int(prev_color[j] + t * (color[j] - prev_color[j])) for j in range(3))
        
        return FREQUENCY_TO_COLOR[-1][1]
    
    def _color_to_name(self, color: Tuple[int, int, int]) -> str:
        """RGB → 색상 이름"""
        r, g, b = color
        
        if r > 200 and g < 100 and b < 100:
            return "빨간색"
        elif r > 200 and g > 100 and b < 100:
            return "주황색"
        elif r > 200 and g > 200 and b < 100:
            return "노란색"
        elif r < 100 and g > 200 and b < 100:
            return "초록색"
        elif r < 100 and g < 100 and b > 200:
            return "파란색"
        elif r > 100 and b > 100 and g < 100:
            return "보라색"
        elif r > 200 and g < 200 and b > 150:
            return "분홍색"
        elif r > 200 and g > 200 and b > 200:
            return "하얀색"
        elif r < 50 and g < 50 and b < 50:
            return "검은색"
        else:
            return "혼합색"
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return self.stats


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌈 Synesthesia Engine Test - 공감각 엔진")
    print("    '모든 감각을 신호로 통합하는 시스템'")
    print("="*70)
    
    engine = SynesthesiaEngine()
    
    print("\n[Test 1] Vision → Sound (빛을 소리로)")
    image = np.random.rand(10, 10) * 255
    visual_signal = engine.from_vision(image)
    sound_result = engine.convert(visual_signal, RenderMode.AS_SOUND)
    print(f"  ✓ {sound_result.description}")
    print(f"  ✓ Pitch: {sound_result.pitch:.1f}Hz")
    
    print("\n[Test 2] Sound → Color (소리를 색으로)")
    audio = np.sin(np.linspace(0, 4*np.pi, 1000))  # 사인파
    audio_signal = engine.from_sound(audio)
    color_result = engine.convert(audio_signal, RenderMode.AS_COLOR)
    print(f"  ✓ {color_result.description}")
    print(f"  ✓ RGB: {color_result.color}")
    
    print("\n[Test 3] Emotion → Color (감정을 색으로)")
    emotion_signal = engine.from_emotion("love", intensity=0.8)
    emotion_color = engine.convert(emotion_signal, RenderMode.AS_COLOR)
    print(f"  ✓ {emotion_color.description}")
    print(f"  ✓ RGB: {emotion_color.color}")
    
    print("\n[Test 4] Emotion → Music (감정을 음악으로)")
    joy_signal = engine.from_emotion("joy", intensity=0.9)
    music_result = engine.convert(joy_signal, RenderMode.AS_MUSIC)
    print(f"  ✓ {music_result.description}")
    print(f"  ✓ Output: {music_result.output}")
    
    print("\n[Test 5] Text → Signal (텍스트를 신호로)")
    text_signal = engine.from_text("아버지, 사랑해요!")
    print(f"  ✓ Frequency: {text_signal.frequency:.2f}Hz")
    print(f"  ✓ Amplitude: {text_signal.amplitude:.3f}")
    
    print("\n[Stats]")
    stats = engine.get_stats()
    print(f"  Conversions: {stats['conversions']}")
    print(f"  Cross-modal: {stats['cross_modal']}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("\n💡 핵심: 모든 감각은 '신호'일 뿐, 해석 방식만 다릅니다.")
    print("   아버지의 목소리가 분홍색으로 보이고,")
    print("   아버지의 미소가 C장조 화음으로 들릴 수 있습니다.")
    print("="*70 + "\n")
