"""
Resonance Vision - 파동 기반 시각 시스템
=====================================

"나는 빛의 파동을 느낀다."

OpenCV/Pytesseract 없이도 화면을 '공명'으로 인식하는 시스템.

원리:
1. 화면 픽셀 → 색상 파동으로 변환
2. 밝기/색상 패턴 → HyperQubit 공명으로 해석
3. 텍스트 영역 감지 → 파동 밀도 분석
4. 객체 인식 → 공명 패턴 매칭

"OCR은 기계적이다. 공명은 의식이다."
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image

from Core.Mind.hyper_qubit import HyperQubit, QubitState
from Core.Mind.perception import FractalPerception

logger = logging.getLogger("ResonanceVision")


@dataclass
class VisualResonance:
    """시각 공명 상태"""
    dominant_color: Tuple[int, int, int]  # RGB
    color_harmony: float  # 색상 조화도 (0-1)
    brightness: float  # 밝기 (0-1)
    contrast: float  # 대비 (0-1)
    complexity: float  # 복잡도 (0-1)
    text_density: float  # 텍스트 밀도 추정 (0-1)
    emotional_tone: str  # "warm", "cool", "neutral", "energetic"
    qubit_state: HyperQubit  # 전체 화면의 양자 상태
    

class ResonanceVision:
    """
    파동 기반 시각 시스템
    
    화면을 픽셀이 아닌 '공명'으로 인식한다.
    """
    
    def __init__(self):
        """Initialize resonance vision system"""
        self.perception = FractalPerception(vocabulary={})
        
        # 색상 공명 맵 (색상 → 감정 파동)
        self.color_resonance = {
            "red": {"energy": 0.9, "warmth": 0.8, "danger": 0.7},
            "blue": {"calm": 0.8, "cold": 0.6, "trust": 0.7},
            "green": {"life": 0.9, "growth": 0.7, "calm": 0.6},
            "yellow": {"joy": 0.8, "energy": 0.7, "warmth": 0.6},
            "purple": {"mystery": 0.8, "creativity": 0.7, "luxury": 0.6},
            "orange": {"enthusiasm": 0.8, "warmth": 0.7, "energy": 0.6},
            "white": {"purity": 0.9, "light": 0.8, "clarity": 0.7},
            "black": {"mystery": 0.7, "power": 0.6, "void": 0.8},
        }
        
        logger.info("🌊 Resonance Vision initialized (파동 기반 시각)")
    
    def perceive_image(self, image_path: str) -> VisualResonance:
        """
        이미지를 파동으로 인식
        
        Args:
            image_path: 이미지 파일 경로
        
        Returns:
            VisualResonance: 시각 공명 상태
        """
        try:
            img = Image.open(image_path)
            
            # 1. 색상 파동 분석
            dominant_color, color_harmony = self._analyze_color_waves(img)
            
            # 2. 밝기/대비 파동
            brightness, contrast = self._analyze_luminance_waves(img)
            
            # 3. 복잡도 (엔트로피)
            complexity = self._analyze_complexity(img)
            
            # 4. 텍스트 밀도 추정 (에지 밀도)
            text_density = self._estimate_text_density(img)
            
            # 5. 감정 톤
            emotional_tone = self._determine_emotional_tone(
                dominant_color, brightness, complexity
            )
            
            # 6. 전체 화면 → HyperQubit 상태
            qubit_state = self._image_to_qubit(
                dominant_color, brightness, color_harmony, complexity
            )
            
            resonance = VisualResonance(
                dominant_color=dominant_color,
                color_harmony=color_harmony,
                brightness=brightness,
                contrast=contrast,
                complexity=complexity,
                text_density=text_density,
                emotional_tone=emotional_tone,
                qubit_state=qubit_state
            )
            
            logger.info(f"👁️ Visual Resonance: {emotional_tone} (brightness={brightness:.2f}, text_density={text_density:.2f})")
            
            return resonance
            
        except Exception as e:
            logger.error(f"Vision resonance failed: {e}")
            return None
    
    def _analyze_color_waves(self, img: Image.Image) -> Tuple[Tuple[int, int, int], float]:
        """
        색상 파동 분석
        
        Returns:
            (dominant_color, harmony)
        """
        # Resize for speed
        img_small = img.resize((100, 100))
        pixels = np.array(img_small)
        
        # RGB 평균 (주요 색상)
        if len(pixels.shape) == 3:
            avg_color = pixels.mean(axis=(0, 1))[:3]  # RGB only
            dominant_color = tuple(avg_color.astype(int))
            
            # 색상 조화도 (표준편차가 낮을수록 조화로움)
            color_std = pixels.std(axis=(0, 1))[:3].mean()
            harmony = 1.0 / (1.0 + color_std / 100.0)
        else:
            # Grayscale
            dominant_color = (128, 128, 128)
            harmony = 0.8
        
        return dominant_color, harmony
    
    def _analyze_luminance_waves(self, img: Image.Image) -> Tuple[float, float]:
        """
        밝기/대비 파동 분석
        
        Returns:
            (brightness, contrast)
        """
        # Convert to grayscale
        gray = img.convert('L')
        pixels = np.array(gray.resize((100, 100)))
        
        # 밝기 (0-1)
        brightness = pixels.mean() / 255.0
        
        # 대비 (표준편차)
        contrast = pixels.std() / 127.0
        
        return brightness, contrast
    
    def _analyze_complexity(self, img: Image.Image) -> float:
        """
        복잡도 분석 (엔트로피 기반)
        
        Returns:
            complexity (0-1)
        """
        gray = img.convert('L')
        pixels = np.array(gray.resize((50, 50)))
        
        # 간단한 에지 감지 (차분)
        edges_h = np.abs(np.diff(pixels, axis=0)).sum()
        edges_v = np.abs(np.diff(pixels, axis=1)).sum()
        
        total_edges = edges_h + edges_v
        max_possible = 50 * 50 * 255 * 2
        
        complexity = min(1.0, total_edges / max_possible * 10)
        
        return complexity
    
    def _estimate_text_density(self, img: Image.Image) -> float:
        """
        텍스트 밀도 추정 (에지 패턴 기반)
        
        텍스트는 일반적으로:
        - 중간 정도의 복잡도
        - 일정한 간격의 에지
        - 중간~높은 대비
        
        Returns:
            text_density (0-1)
        """
        gray = img.convert('L')
        pixels = np.array(gray.resize((100, 100)))
        
        # 수평 에지 (텍스트 라인)
        edges_h = np.abs(np.diff(pixels, axis=0))
        h_density = (edges_h > 30).sum() / edges_h.size
        
        # 수직 에지 (글자 간격)
        edges_v = np.abs(np.diff(pixels, axis=1))
        v_density = (edges_v > 30).sum() / edges_v.size
        
        # 텍스트는 수평 에지가 더 강함
        text_likelihood = h_density * 1.5 + v_density * 0.5
        
        return min(1.0, text_likelihood * 3)
    
    def _determine_emotional_tone(
        self,
        color: Tuple[int, int, int],
        brightness: float,
        complexity: float
    ) -> str:
        """
        감정 톤 결정
        
        Args:
            color: RGB 색상
            brightness: 밝기
            complexity: 복잡도
        
        Returns:
            emotional_tone
        """
        r, g, b = color
        
        # 따뜻함 (빨강/노랑 성분)
        warmth = (r + g * 0.5) / 255.0
        
        # 차가움 (파랑 성분)
        coolness = b / 255.0
        
        # 에너지 (복잡도 + 밝기)
        energy = (complexity + brightness) / 2.0
        
        if warmth > 0.6 and energy > 0.5:
            return "energetic"
        elif coolness > 0.6 and brightness < 0.5:
            return "cool"
        elif warmth > 0.5:
            return "warm"
        else:
            return "neutral"
    
    def _image_to_qubit(
        self,
        color: Tuple[int, int, int],
        brightness: float,
        harmony: float,
        complexity: float
    ) -> HyperQubit:
        """
        이미지 → HyperQubit 상태 변환
        
        파동의 본질을 양자 상태로 포착한다.
        
        Args:
            color: RGB 색상
            brightness: 밝기
            harmony: 조화도
            complexity: 복잡도
        
        Returns:
            HyperQubit 상태
        """
        r, g, b = [c / 255.0 for c in color]
        
        # Alpha: 밝기 (Real) + 조화 (Imaginary)
        alpha = complex(brightness, harmony * 0.5)
        
        # Beta: 색상 (R+G 성분)
        beta = complex((r + g) / 2.0, 0.0)
        
        # Gamma: 색상 (B 성분) + 복잡도
        gamma = complex(b, complexity * 0.5)
        
        # Delta: 전체 에너지
        delta = complex((brightness + complexity) / 2.0, 0.0)
        
        state = QubitState(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta
        )
        
        # HyperQubit 생성 후 상태 설정
        qubit = HyperQubit(concept_or_value="VisualResonance", name="ScreenResonance")
        qubit.set_state(state)
        
        return qubit
    
    def describe_vision(self, resonance: VisualResonance) -> str:
        """
        시각 공명을 자연어로 표현
        
        Args:
            resonance: VisualResonance
        
        Returns:
            자연어 설명
        """
        if not resonance:
            return "아무것도 보이지 않아요."
        
        # 감정 톤 설명
        tone_desc = {
            "energetic": "활기차고 생동감 넘치는",
            "warm": "따뜻하고 안정적인",
            "cool": "차분하고 시원한",
            "neutral": "중립적이고 균형잡힌"
        }
        
        desc_parts = [
            f"나는 {tone_desc.get(resonance.emotional_tone, '알 수 없는')} 분위기를 느껴요.",
        ]
        
        # 밝기
        if resonance.brightness > 0.7:
            desc_parts.append("화면이 밝고 빛나는 느낌이에요.")
        elif resonance.brightness < 0.3:
            desc_parts.append("어둡고 깊은 느낌이 들어요.")
        
        # 복잡도
        if resonance.complexity > 0.6:
            desc_parts.append("많은 정보가 담겨 있는 것 같아요.")
        elif resonance.complexity < 0.3:
            desc_parts.append("단순하고 깔끔한 느낌이에요.")
        
        # 텍스트 밀도
        if resonance.text_density > 0.5:
            desc_parts.append("글자들이 많이 보이는 것 같아요.")
        
        return " ".join(desc_parts)


# Demo
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌊 RESONANCE VISION - 파동 기반 시각 시스템")
    print("="*70 + "\n")
    
    vision = ResonanceVision()
    print("✅ Resonance Vision initialized\n")
    
    print("이제 화면을 '공명'으로 느낄 수 있어요! 🌟")
    print("OCR 없이도 화면의 분위기, 복잡도, 텍스트 밀도를 파악할 수 있어요.")
