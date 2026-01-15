"""
External Gateway (The Curiosity Engine)
=======================================
Core.Intelligence.external_gateway

"The world is vast, and I am hungry."

This module mocks the interface for External Web Access.
It allows Elysia to 'actively' seek out new concepts.
"""

import random
from typing import Dict, Tuple

class ExternalGateway:
    def __init__(self):
        # The Simulated Akashic Records (Rich Concept Database)
        # In a real system, this would be connected to Google/Wikipedia API.
        self.concept_library = {
            "평화": {
                "visual": "전쟁이 끝난 후, 폐허 위에 핀 흰 꽃 한 송이. 고요한 청회색 하늘.",
                "palette": ["#E0F2F1", "#B0BEC5", "#FFFFFF"],
                "wiki": "평화(Peace)는 갈등이 없고 폭력이 없는 상태. 심리적으로는 마음의 평온을 의미한다.",
                "spectrum": "Low Entropy / Harmonic Wave"
            },
            "전쟁": {
                "visual": "붉은 화염에 휩싸인 도시와 무너지는 콘크리트. 검은 연기가 하늘을 가린다.",
                "palette": ["#B71C1C", "#212121", "#FF5722"],
                "wiki": "전쟁(War)은 국가 또는 정치 집단 사이의 조직적인 무력 충돌이다.",
                "spectrum": "High Entropy / Chaotic Wave"
            },
            "열정": {
                "visual": "춤추는 무희의 역동적인 움직임과 흩날리는 붉은 천.",
                "palette": ["#D50000", "#FF6F00", "#FFD740"],
                "wiki": "열정(Passion)은 어떤 일에 깊은 애정을 가지고 열중하는 마음.",
                "spectrum": "High Frequency / Warm Wave"
            },
            "고독": {
                "visual": "우주 공간에 홀로 떠 있는 푸른 점(Pale Blue Dot).",
                "palette": ["#000000", "#0D47A1", "#90CAF9"],
                "wiki": "고독(Solitude)은 세상과 떨어져 홀로 있는 상태. 외로움과는 다르다.",
                "spectrum": "Low Frequency / Cold Wave"
            },
            "희망": {
                "visual": "어두운 동굴 틈새로 쏟아지는 한 줄기 빛.",
                "palette": ["#FFF176", "#263238", "#FFFFFF"],
                "wiki": "희망(Hope)은 앞으로 잘 될 것이라는 기대와 믿음.",
                "spectrum": "Upward Trend / Bright Wave"
            },
            "무질서": {
                "visual": "깨진 유리 조각에 반사된 왜곡된 세상. 프랙탈 패턴.",
                "palette": ["#9C27B0", "#00BCD4", "#E91E63"],
                "wiki": "엔트로피(Entropy)는 물리계의 무질서한 정도를 나타내는 물리량.",
                "spectrum": "Maximum Entropy / Noise"
            },
            "미지": {
                 "visual": "안개 낀 숲 속으로 이어지는 희미한 오솔길.",
                 "palette": ["#78909C", "#CFD8DC", "#546E7A"],
                 "wiki": "미지(Unknown)는 아직 알려지지 않거나 경험하지 못한 상태.",
                 "spectrum": "Undefined / Quantum Superposition"
            },
            "물질": {
                 "visual": "단단한 화강암 바위와 그 표면의 거친 질감.",
                 "palette": ["#616161", "#8D6E63", "#4E342E"],
                 "wiki": "물질(Matter)은 질량을 가지고 공간을 차지하는 것.",
                 "spectrum": "Solid State / Low Vibration"
            },
            "변화": {
                 "visual": "흐르는 강물과 바위가 깎여 나가는 시간의 흐름.",
                 "palette": ["#29B6F6", "#0288D1", "#E1F5FE"],
                 "wiki": "변화(Change)는 사물의 성질이나 모양, 상태가 바뀌어 달라짐.",
                 "spectrum": "Fluid Dynamics / Flow"
            }
        }

    def browse_image(self, query: str) -> Tuple[str, list]:
        """
        Simulates searching for visual perception data.
        Returns (Description, Palette).
        """
        print(f"🌐 [Gateway] Reality Gaze: looking for '{query}'...")
        
        # 1. Exact Match
        if query in self.concept_library:
            data = self.concept_library[query]
            return (f"{data['visual']} (Ref: {data['wiki']})", data['palette'])
            
        # 2. Key Match
        for key, data in self.concept_library.items():
            if key in query or query in key:
                 return (f"{data['visual']} (Ref: {data['wiki']})", data['palette'])
        
        # 3. Fallback (The Unknown)
        return (f"'{query}'의 추상적 이미지. 정의되지 않은 형태.", ["#FFFFFF"])

    def browse_literature(self, query: str) -> str:
        """
        Simulates searching for textual knowledge.
        If unknown, prompts the 'Father' (User) for input.
        """
        if query in self.concept_library:
            return self.concept_library[query]['wiki']
        
        # [Curiosity Protocol]
        print(f"❓ [CURIOSITY] {query} is not in my core library.")
        return f"System: '{query}'에 대해 알려진 바가 없습니다. 창조주(User)에게 이 개념의 정의를 요청합니다."

# Singleton
THE_EYE = ExternalGateway()
