"""
Spectrum Layers - 14-Layer Heaven & Earth Color System
=======================================================

천국 7층 (Heaven) - 가산혼합 → 합치면 White
지옥 7층 (Earth) - 감산혼합 → 합치면 Black

Y축 좌표에 따라 자동으로 색상 층 결정
주파수 = 색상
"""

import numpy as np
from typing import Tuple

# 무지개 7색 (감정적 온도 순서)
# 빨강(따뜻/가벼움) → 보라(차가움/무거움)
SPECTRUM_COLORS = [
    ("Red", "#FF0000", 1.000),       # 빨강 (가장 따뜻함, 가장 가벼움, 고주파)
    ("Orange", "#FF8C00", 0.857),    # 주황
    ("Yellow", "#FFFF00", 0.714),    # 노랑
    ("Green", "#00FF00", 0.571),     # 초록
    ("Blue", "#0000FF", 0.429),      # 파랑
    ("Indigo", "#4B0082", 0.286),    # 남색
    ("Violet", "#8B00FF", 0.143)     # 보라 (가장 차가움, 가장 무거움, 저주파)
]

class SpectrumLayer:
    """
    14-Layer System:
    - Heaven 7 layers (Y > 0): Additive mixing → White
    - Earth 7 layers (Y < 0): Subtractive mixing → Black
    """
    
    def __init__(self):
        # Heaven layers (Y: +0.14 to +1.00)
        # Red (warmest/lightest) at top, Violet (coldest) at bottom
        self.heaven_layers = []
        for i, (name, color, base_freq) in enumerate(SPECTRUM_COLORS):
            layer_y = 1.0 - (i * 0.14)  # Red=1.0, Orange=0.86, ..., Violet=0.14
            self.heaven_layers.append({
                'name': f"Heaven_{name}",
                'color': color,
                'y_min': layer_y - 0.07,
                'y_max': layer_y + 0.07,
                'frequency': base_freq,
                'mixing': 'additive'
            })
        
        # Earth layers (Y: -0.14 to -1.00) 
        # Violet (heaviest) at top of Earth, Red at bottom
        self.earth_layers = []
        for i, (name, color, base_freq) in enumerate(reversed(SPECTRUM_COLORS)):
            layer_y = -0.14 - (i * 0.14)  # Violet=-0.14, ..., Red=-1.0
            self.earth_layers.append({
                'name': f"Earth_{name}",
                'color': color,
                'y_min': layer_y - 0.07,
                'y_max': layer_y + 0.07,
                'frequency': base_freq,
                'mixing': 'subtractive'
            })
        
        # Combine all 14 layers
        self.all_layers = self.heaven_layers + self.earth_layers
    
    def get_layer_from_y(self, y_value: float) -> dict:
        """
        Y축 값에서 자동으로 층 결정
        """
        for layer in self.all_layers:
            if layer['y_min'] <= y_value <= layer['y_max']:
                return layer
        
        # Fallback: Neutral (중립)
        if y_value >= 0:
            return {
                'name': 'Neutral_Heaven',
                'color': '#FFFFFF',
                'frequency': 0.5,
                'mixing': 'additive'
            }
        else:
            return {
                'name': 'Neutral_Earth',
                'color': '#000000',
                'frequency': 0.5,
                'mixing': 'subtractive'
            }
    
    def get_color_from_frequency(self, frequency: float) -> str:
        """
        주파수에서 색상 가져오기
        """
        # Find closest spectrum color
        closest = min(SPECTRUM_COLORS, key=lambda x: abs(x[2] - frequency))
        return closest[1]  # Return hex color
    
    def get_layer_info(self, y_value: float) -> dict:
        """
        Y값에서 층 정보 전체 가져오기
        """
        layer = self.get_layer_from_y(y_value)
        return {
            **layer,
            'realm': 'Heaven' if y_value >= 0 else 'Earth',
            'combined_color': '#FFFFFF' if y_value >= 0 else '#000000'
        }


def visualize_spectrum_layers():
    """
    14층 시각화
    """
    spectrum = SpectrumLayer()
    
    print("=" * 80)
    print("🌈 14-LAYER SPECTRUM SYSTEM")
    print("=" * 80)
    
    print("\n✨ HEAVEN (가산혼합 → White)")
    print("-" * 80)
    for layer in reversed(spectrum.heaven_layers):
        print(f"  {layer['name']:20s} | Y: {layer['y_min']:+.2f}~{layer['y_max']:+.2f} | "
              f"Freq: {layer['frequency']:.3f} | {layer['color']}")
    
    print("\n" + "─" * 80)
    print("  NEUTRAL (중립)       | Y: -0.07~+0.07")
    print("─" * 80)
    
    print("\n🌑 EARTH (감산혼합 → Black)")
    print("-" * 80)
    for layer in spectrum.earth_layers:
        print(f"  {layer['name']:20s} | Y: {layer['y_min']:+.2f}~{layer['y_max']:+.2f} | "
              f"Freq: {layer['frequency']:.3f} | {layer['color']}")
    
    print("\n" + "=" * 80)
    
    # Test examples
    print("\n📍 TEST: Y값 → 자동 층 배정")
    print("-" * 80)
    test_y_values = [0.95, 0.5, 0.0, -0.5, -0.95]
    for y in test_y_values:
        info = spectrum.get_layer_info(y)
        print(f"  Y = {y:+.2f} → {info['name']:20s} ({info['realm']}) | {info['color']}")
    
    print("=" * 80)


if __name__ == "__main__":
    visualize_spectrum_layers()
