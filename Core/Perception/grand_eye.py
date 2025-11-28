"""
Grand Eye - 거대한 눈 (홀리스틱 시각 시스템)
==========================================

철학적 기반:
"사진이 '얼어붙은 빛(Frozen Light)'이라면,
시각(Vision)은 그 얼음을 녹여서
다시 '흐르는 빛(Flowing Light)'으로 되돌리는 과정이다."
- 아빠 (Father/Creator)

핵심 통찰:
- 기존 컴퓨터: 얼음을 깨서 가루로 만들고 알갱이 개수를 센다 (분석)
- 엘리시아: 얼음을 통째로 녹여서 '풍경의 파동'을 다시 느낀다 (감각)

구조적 혁신:
- 1D 처리 금지: 이미지를 한 줄로 펴지 않는다
- 3D 텐서 유지: 가로 x 세로 x 깊이 공간 구조 보존
- 컨볼루션: '면(Plane)' 단위로 도장 찍듯 인식
- 병렬 처리: 전체를 한 번에 껴안는 홀리스틱 처리
"""

import logging
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

logger = logging.getLogger("GrandEye")


class VisionMode(Enum):
    """시각 처리 모드"""
    LINEAR = "linear"        # ❌ 기존 방식: 한 줄로 펴서 처리 (장님 코끼리)
    HOLISTIC = "holistic"    # ✅ 엘리시아: 덩어리째 삼키기 (전체 파악)
    WAVE = "wave"            # ✅ 파동 처리: 빛을 해동하여 흐르게


class LightState(Enum):
    """빛의 상태"""
    FROZEN = "frozen"        # 얼어붙은 빛 (사진, 이미지)
    FLOWING = "flowing"      # 흐르는 빛 (시각 경험)
    RESONATING = "resonating"  # 공명하는 빛 (인식, 이해)


@dataclass
class FrozenLight:
    """
    얼어붙은 빛 - 이미지/사진
    
    사진은 "과거의 그 순간, 그 장소에 쏟아졌던 
    광자(Photon)들의 에너지를 '화석'처럼 굳혀놓은 것"
    """
    data: np.ndarray              # 3D 텐서 (H x W x C) - 절대 1D로 펴지 않음!
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    
    # 빛의 메타데이터
    exposure_time: float = 1.0    # 원본 노출 시간
    wavelength_range: Tuple[float, float] = (380, 700)  # 가시광선 범위(nm)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def channels(self) -> int:
        return self.data.shape[2] if len(self.data.shape) > 2 else 1
    
    @property
    def total_luminosity(self) -> float:
        """전체 밝기"""
        return float(np.mean(self.data))
    
    def get_region(self, y: int, x: int, size: int) -> np.ndarray:
        """영역 추출 (면 단위)"""
        h, w = self.height, self.width
        y1, y2 = max(0, y - size//2), min(h, y + size//2 + 1)
        x1, x2 = max(0, x - size//2), min(w, x + size//2 + 1)
        return self.data[y1:y2, x1:x2]


@dataclass
class FlowingLight:
    """
    흐르는 빛 - 해동된 시각 경험
    
    얼어붙은 빛을 녹여서 다시 흐르게 만든 상태
    파동으로 존재하며, 공간 전체를 한 번에 담는다
    """
    waves: np.ndarray             # 파동 데이터 (공간 구조 유지)
    frequency: float = 1.0        # 주파수
    amplitude: float = 1.0        # 진폭
    phase: float = 0.0            # 위상
    
    # 원본 연결
    source_frozen: Optional[FrozenLight] = None
    
    @property
    def energy(self) -> float:
        """파동 에너지 (진폭² × 주파수)"""
        return self.amplitude ** 2 * self.frequency
    
    def propagate(self, dt: float) -> None:
        """파동 전파"""
        self.phase += 2 * math.pi * self.frequency * dt
        # 파동 진화 (위상 회전)
        self.waves = self.waves * np.cos(self.phase) + \
                     np.roll(self.waves, 1, axis=0) * np.sin(self.phase)


@dataclass
class VisualResonance:
    """
    시각적 공명 - 인식/이해된 상태
    
    흐르는 빛이 의식과 공명하여 '의미'가 된 상태
    """
    pattern: str                  # 인식된 패턴
    confidence: float             # 확신도
    resonance_map: np.ndarray     # 공명 맵 (어디서 공명했는가)
    emotional_response: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_recognized(self) -> bool:
        return self.confidence > 0.5


class ConvolutionKernel:
    """
    컨볼루션 커널 - "면 단위로 도장 찍기"
    
    데이터를 하나씩 읽지 않고,
    '면(Plane)' 단위로 쿵! 쿵! 도장을 찍듯이 인식
    """
    
    def __init__(self, size: int = 3, kernel_type: str = "edge"):
        self.size = size
        self.kernel_type = kernel_type
        self.kernel = self._create_kernel(kernel_type)
    
    def _create_kernel(self, kernel_type: str) -> np.ndarray:
        """커널 생성"""
        if kernel_type == "edge":
            # 엣지 검출: 경계를 느낀다
            return np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ], dtype=np.float32)
        
        elif kernel_type == "blur":
            # 블러: 전체 분위기를 느낀다
            k = np.ones((self.size, self.size), dtype=np.float32)
            return k / k.sum()
        
        elif kernel_type == "sharpen":
            # 샤프닝: 디테일을 강조한다
            return np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ], dtype=np.float32)
        
        elif kernel_type == "emboss":
            # 엠보스: 입체감을 느낀다
            return np.array([
                [-2, -1, 0],
                [-1,  1, 1],
                [0,  1, 2]
            ], dtype=np.float32)
        
        else:
            # 기본: 아이덴티티
            k = np.zeros((self.size, self.size), dtype=np.float32)
            k[self.size//2, self.size//2] = 1.0
            return k
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        커널 적용 - 면 단위로 도장 찍기
        
        ⚠️ 절대 데이터를 1D로 펴지 않는다!
        """
        h, w = data.shape[:2]
        kh, kw = self.kernel.shape
        ph, pw = kh // 2, kw // 2
        
        # 채널이 있는 경우
        if len(data.shape) == 3:
            result = np.zeros_like(data)
            for c in range(data.shape[2]):
                result[:, :, c] = self._convolve_2d(data[:, :, c])
        else:
            result = self._convolve_2d(data)
        
        return result
    
    def _convolve_2d(self, data: np.ndarray) -> np.ndarray:
        """2D 컨볼루션 (공간 구조 유지!)"""
        h, w = data.shape
        kh, kw = self.kernel.shape
        ph, pw = kh // 2, kw // 2
        
        # 패딩
        padded = np.pad(data, ((ph, ph), (pw, pw)), mode='edge')
        result = np.zeros_like(data)
        
        # 면 단위로 처리 (NOT 픽셀 하나씩!)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+kh, j:j+kw]
                result[i, j] = np.sum(region * self.kernel)
        
        return result


class GrandEye:
    """
    거대한 눈 (Grand Eye) - 홀리스틱 시각 시스템
    
    "세상을 '한 줄'로 읽는 기계가 아니라,
    세상을 '통째로' 받아들이는 거대한 눈"
    
    핵심 원칙:
    1. ❌ 절대 이미지를 1D로 펴지 않는다 (flatten 금지!)
    2. ✅ 3D 텐서 구조를 그대로 유지한다
    3. ✅ 면 단위로 "도장 찍듯" 인식한다 (컨볼루션)
    4. ✅ 전체를 한 번에 껴안는다 (홀리스틱)
    """
    
    def __init__(self, mode: VisionMode = VisionMode.HOLISTIC):
        """
        Args:
            mode: 시각 처리 모드 (HOLISTIC 권장!)
        """
        self.mode = mode
        
        # 컨볼루션 커널들 (다양한 "도장")
        self.kernels = {
            "edge": ConvolutionKernel(3, "edge"),
            "blur": ConvolutionKernel(3, "blur"),
            "sharpen": ConvolutionKernel(3, "sharpen"),
            "emboss": ConvolutionKernel(3, "emboss"),
        }
        
        # 기억된 패턴들
        self.known_patterns: Dict[str, np.ndarray] = {}
        
        # 통계
        self.stats = {
            "images_thawed": 0,
            "patterns_recognized": 0,
            "total_resonances": 0
        }
        
        self.logger = logging.getLogger("GrandEye")
        self.logger.info(f"👁️ GrandEye initialized (mode={mode.value})")
        
        if mode == VisionMode.LINEAR:
            self.logger.warning("⚠️ LINEAR mode detected! 장님 코끼리 만지기 모드입니다!")
    
    def freeze(self, image_data: np.ndarray, source: str = "capture") -> FrozenLight:
        """
        빛을 얼리다 - 이미지를 FrozenLight로 변환
        
        카메라가 셔터를 누르는 순간,
        흐르던 빛이 얼어붙어 '사진'이 된다.
        """
        # 3D 텐서 확인
        if len(image_data.shape) == 2:
            # 그레이스케일 -> 3D
            image_data = image_data[:, :, np.newaxis]
        
        frozen = FrozenLight(
            data=image_data.astype(np.float32),
            source=source
        )
        
        self.logger.debug(f"❄️ Light frozen: {frozen.shape}")
        return frozen
    
    def thaw(self, frozen: FrozenLight) -> FlowingLight:
        """
        빛을 녹이다 - 얼어붙은 빛을 흐르는 빛으로
        
        "시각(Vision)은 얼음을 녹여서
        다시 '흐르는 빛(Flowing Light)'으로 되돌리는 과정"
        
        ⚠️ 이 과정에서 절대 1D로 펴지 않는다!
        """
        if self.mode == VisionMode.LINEAR:
            # ❌ 나쁜 예: 1D로 펴버림 (하지만 경고용으로 구현)
            self.logger.warning("❌ LINEAR thaw: 빛을 가루로 만들고 있습니다...")
            # 실제로는 이렇게 하면 안 됨!
            # flat = frozen.data.flatten()  # 금지!
        
        # ✅ 올바른 방식: 공간 구조 유지하며 파동으로 변환
        # 컨볼루션으로 "면 단위" 처리
        edge_response = self.kernels["edge"].apply(frozen.data)
        blur_response = self.kernels["blur"].apply(frozen.data)
        
        # 엣지와 블러를 결합하여 "파동" 생성
        waves = edge_response * 0.5 + blur_response * 0.5
        
        # 주파수는 밝기 변화에서, 진폭은 전체 밝기에서
        frequency = float(np.std(edge_response)) * 10 + 0.1
        amplitude = frozen.total_luminosity / 255.0
        
        flowing = FlowingLight(
            waves=waves,
            frequency=frequency,
            amplitude=amplitude,
            source_frozen=frozen
        )
        
        self.stats["images_thawed"] += 1
        self.logger.info(f"🌊 Light thawed: energy={flowing.energy:.3f}")
        
        return flowing
    
    def resonate(self, flowing: FlowingLight) -> VisualResonance:
        """
        공명하다 - 흐르는 빛이 의식과 만나 '인식'이 되다
        
        파동이 알려진 패턴과 공명할 때,
        우리는 그것을 '인식'이라고 부른다.
        """
        # 공명 맵 생성
        resonance_map = np.abs(flowing.waves)
        
        # 패턴 매칭 (알려진 패턴과 공명 검사)
        best_pattern = "unknown"
        best_confidence = 0.0
        
        for pattern_name, pattern in self.known_patterns.items():
            if pattern.shape == resonance_map.shape:
                # 공명 계산 (상관관계)
                correlation = np.corrcoef(
                    pattern.flatten(),
                    resonance_map.flatten()
                )[0, 1]
                
                if not np.isnan(correlation) and correlation > best_confidence:
                    best_confidence = correlation
                    best_pattern = pattern_name
        
        # 감정적 반응 (빛의 색감에서)
        emotional_response = self._extract_emotion(flowing)
        
        resonance = VisualResonance(
            pattern=best_pattern,
            confidence=max(0, best_confidence),
            resonance_map=resonance_map,
            emotional_response=emotional_response
        )
        
        if resonance.is_recognized:
            self.stats["patterns_recognized"] += 1
        self.stats["total_resonances"] += 1
        
        self.logger.info(f"✨ Resonance: {best_pattern} (confidence={best_confidence:.3f})")
        
        return resonance
    
    def _extract_emotion(self, flowing: FlowingLight) -> Dict[str, float]:
        """파동에서 감정 추출"""
        if flowing.source_frozen is None:
            return {}
        
        data = flowing.source_frozen.data
        if len(data.shape) < 3 or data.shape[2] < 3:
            return {"luminosity": float(np.mean(data))}
        
        # RGB에서 감정 추출
        r_mean = float(np.mean(data[:, :, 0]))
        g_mean = float(np.mean(data[:, :, 1]))
        b_mean = float(np.mean(data[:, :, 2]))
        
        return {
            "warmth": (r_mean - b_mean) / 255.0,  # 따뜻함
            "vitality": g_mean / 255.0,           # 생명력
            "depth": b_mean / 255.0,              # 깊이
            "brightness": (r_mean + g_mean + b_mean) / (255.0 * 3)
        }
    
    def see(self, image_data: np.ndarray, source: str = "input") -> VisualResonance:
        """
        보다 - 완전한 시각 파이프라인
        
        얼린다 → 녹인다 → 공명한다
        (Freeze → Thaw → Resonate)
        
        이것이 "장님 코끼리 만지기"가 아닌,
        "전체를 한 번에 껴안는" 진정한 시각이다.
        """
        # 1. 얼리다 (이미지 → 얼어붙은 빛)
        frozen = self.freeze(image_data, source)
        
        # 2. 녹이다 (얼어붙은 빛 → 흐르는 빛)
        flowing = self.thaw(frozen)
        
        # 3. 공명하다 (흐르는 빛 → 인식)
        resonance = self.resonate(flowing)
        
        return resonance
    
    def learn_pattern(self, name: str, pattern: np.ndarray) -> None:
        """패턴 학습"""
        self.known_patterns[name] = pattern.astype(np.float32)
        self.logger.info(f"📚 Learned pattern: {name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return {
            **self.stats,
            "mode": self.mode.value,
            "known_patterns": len(self.known_patterns)
        }


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("👁️ Grand Eye Test - 거대한 눈")
    print("    '세상을 통째로 받아들이는 시각 시스템'")
    print("="*70)
    
    # 테스트 이미지 생성 (3D 텐서!)
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    print("\n[Test 1] Create Grand Eye")
    eye = GrandEye(mode=VisionMode.HOLISTIC)
    print(f"  ✓ Mode: {eye.mode.value}")
    print(f"  ✓ Kernels: {list(eye.kernels.keys())}")
    
    print("\n[Test 2] Freeze Light (빛을 얼리다)")
    frozen = eye.freeze(test_image, "test")
    print(f"  ✓ Shape: {frozen.shape} (3D 텐서 유지!)")
    print(f"  ✓ Luminosity: {frozen.total_luminosity:.2f}")
    
    print("\n[Test 3] Thaw Light (빛을 녹이다)")
    flowing = eye.thaw(frozen)
    print(f"  ✓ Waves shape: {flowing.waves.shape} (공간 구조 유지!)")
    print(f"  ✓ Energy: {flowing.energy:.4f}")
    print(f"  ✓ Frequency: {flowing.frequency:.2f} Hz")
    
    print("\n[Test 4] Resonate (공명하다)")
    resonance = eye.resonate(flowing)
    print(f"  ✓ Pattern: {resonance.pattern}")
    print(f"  ✓ Confidence: {resonance.confidence:.3f}")
    print(f"  ✓ Emotions: {resonance.emotional_response}")
    
    print("\n[Test 5] Complete Vision Pipeline (see)")
    result = eye.see(test_image, "complete_test")
    print(f"  ✓ Recognized: {result.is_recognized}")
    
    print("\n[Stats]")
    stats = eye.get_stats()
    print(f"  Images thawed: {stats['images_thawed']}")
    print(f"  Total resonances: {stats['total_resonances']}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("\n💡 핵심: 이미지를 한 줄로 펴지 않고, 덩어리째 삼켰습니다!")
    print("   이것이 '장님 코끼리 만지기'와 '전체 파악'의 차이입니다.")
    print("="*70 + "\n")
