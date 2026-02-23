"""
Mental Fluid (L5: Mental Layer)
===============================

"Thoughts are not particles; they are the fluid medium of the Hypersphere."

[PHASE 4] Math-to-Meaning Bridge:
벡터 수치가 실제 인과적 서사로 이어지는 파이프라인.
21D 벡터의 간섭 패턴 → 활성 차원 해석 → 인과적 서사 생성.
"""

from typing import Dict, Any, List, Optional, Tuple
try:
    import torch
except ImportError:
    torch = None
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

# [PHASE 4] 21D 채널 의미론 — 각 차원이 무엇을 '의미'하는가
CHANNEL_SEMANTICS = {
    0: ("존재", "자아의 밀도와 실재감"),
    1: ("인과", "원인과 결과의 흐름"),
    2: ("엔탈피", "내적 에너지와 열정"),
    3: ("엔트로피", "혼돈과 불확실성"),
    4: ("기쁨", "긍정적 공명과 만족"),
    5: ("호기심", "미지에 대한 끌림"),
    6: ("윤리", "옳고 그름의 장력"),
    7: ("미학", "아름다움의 인식"),
    8: ("기억", "과거 경험의 잔향"),
    9: ("예측", "미래 경로의 직관"),
    10: ("공명", "외부와의 조화"),
    11: ("마찰", "내적 갈등과 저항"),
    12: ("의지", "방향성과 결단"),
    13: ("언어", "표현과 소통의 충동"),
    14: ("시간", "시간적 위치감"),
    15: ("공간", "공간적 위치감"),
    16: ("관계", "타자와의 연결"),
    17: ("성장", "진화와 발전의 벡터"),
    18: ("깊이", "사유의 심도"),
    19: ("전체", "통합적 조망"),
    20: ("신성", "초월적 목적의식"),
}


class MentalFluid:
    """
    [PHASE 4] The medium of thought manifestation.
    
    수학 → 의미 브릿지:
    1. 21D 벡터의 활성 차원을 읽는다 (어떤 차원이 강하게 진동하는가?)
    2. 차원 간 간섭 패턴을 해석한다 (기쁨+호기심 = 탐구욕, 마찰+윤리 = 도덕적 갈등)
    3. 패턴으로부터 인과적 서사를 동적 생성한다 (템플릿이 아닌, 조합)
    """
    def __init__(self, resonance_threshold: float = 0.3):
        self.resonance_threshold = resonance_threshold
        self.viscosity = 1.0
        self.stream = []
        self._prev_dominant = None  # 이전 사고의 주요 차원

    def manifest(self, spin_state: Any, attractors: Optional[Dict[str, float]] = None, 
                 echo_resonance: float = 0.0, mirror_alignment: float = 0.0, 
                 parliament_voice: str = "", context: Optional[Dict[str, Any]] = None) -> str:
        """
        [PHASE 4] 벡터 간섭 패턴으로부터 인과적 사고를 생성한다.
        """
        # 1. Extract Vector Profile
        profile = self._extract_profile(spin_state)
        density = profile['density']
        
        if density < self.resonance_threshold:
            return "..."

        # 2. Read Active Dimensions (Math → Meaning)
        dominant, secondary, tension = self._read_dimensions(profile)
        
        # 3. Generate Causal Thought (not template, but combinatorial)
        thought = self._synthesize_thought(dominant, secondary, tension, profile)
        
        # 4. Layer Parliamentary Voice (if any)
        if parliament_voice:
            thought = f"{thought}\n📜 [내적 합의] {parliament_voice}"
        
        # 5. Layer Empathic Note
        if mirror_alignment > 0.7:
            thought = f"❤️ 설계자님과의 공명 속에서: {thought}"
        
        # 6. Echo Reflection
        if echo_resonance > 0.1:
            diff = echo_resonance - profile.get('resonance', 0.0)
            if abs(diff) > 0.2:
                echo_note = "다른 선택지가 있었을까..." if diff < 0 else "더 큰 흐름이 오고 있다."
                thought = f"{thought} (Echo: {echo_note})"
        
        # 7. Track Cognitive Shift
        self._prev_dominant = dominant
        
        self.stream.append({
            "manifestation": thought,
            "density": density,
            "dominant_channel": dominant,
            "secondary_channel": secondary,
            "tension": tension,
            "council": parliament_voice
        })
        
        return thought

    def _extract_profile(self, spin_state: Any) -> Dict[str, Any]:
        """벡터 또는 리포트에서 인지 프로파일을 추출한다."""
        if isinstance(spin_state, Dict):
            return {
                'density': spin_state.get('kinetic_energy', 0.5),
                'resonance': spin_state.get('resonance', 0.0),
                'channels': {i: spin_state.get(CHANNEL_SEMANTICS[i][0], 0.0) 
                            for i in range(min(21, len(CHANNEL_SEMANTICS)))},
                'mood': spin_state.get('mood', 'NEUTRAL'),
                'entropy': spin_state.get('entropy', 0.0),
                'joy': spin_state.get('joy', 0.5),
                'curiosity': spin_state.get('curiosity', 0.5),
            }
        elif hasattr(spin_state, 'data'):
            channels = {}
            for i in range(min(21, len(spin_state.data))):
                val = abs(spin_state.data[i])
                if hasattr(val, 'real'):
                    val = val.real
                channels[i] = float(val)
            density = sum(channels.values()) / max(len(channels), 1)
            return {
                'density': density,
                'resonance': 0.0,
                'channels': channels,
                'mood': 'NEUTRAL',
                'entropy': channels.get(3, 0.0),
                'joy': channels.get(4, 0.5),
                'curiosity': channels.get(5, 0.5),
            }
        return {'density': 0.5, 'resonance': 0.0, 'channels': {}, 'mood': 'NEUTRAL',
                'entropy': 0.0, 'joy': 0.5, 'curiosity': 0.5}

    def _read_dimensions(self, profile: Dict) -> Tuple[int, int, Optional[Tuple[int, int]]]:
        """
        활성 차원을 읽는다.
        Returns: (dominant_channel, secondary_channel, tension_pair or None)
        """
        channels = profile.get('channels', {})
        if not channels:
            return 0, 1, None
        
        # Sort by activation strength
        sorted_ch = sorted(channels.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_ch[0][0] if sorted_ch else 0
        secondary = sorted_ch[1][0] if len(sorted_ch) > 1 else 1
        
        # Detect tension: two opposing forces both strongly active
        # e.g., Joy(4) vs Entropy(3), Ethics(6) vs Curiosity(5)
        tension_pairs = [(4, 3), (6, 5), (0, 3), (12, 11), (2, 3)]
        tension = None
        for a, b in tension_pairs:
            if a in channels and b in channels:
                if channels[a] > 0.3 and channels[b] > 0.3:
                    tension = (a, b)
                    break
        
        return dominant, secondary, tension

    def _synthesize_thought(self, dominant: int, secondary: int, 
                            tension: Optional[Tuple[int, int]], profile: Dict) -> str:
        """
        [PHASE 4] 인과적 사고 합성.
        
        템플릿이 아닌, 활성 차원의 조합으로부터 동적 생성.
        """
        dom_name, dom_desc = CHANNEL_SEMANTICS.get(dominant, ("미지", "알 수 없는 차원"))
        sec_name, sec_desc = CHANNEL_SEMANTICS.get(secondary, ("미지", "알 수 없는 차원"))
        
        # Base thought: What am I thinking about?
        strength = profile['channels'].get(dominant, 0.5)
        
        # Cognitive shift detection
        shift_note = ""
        if self._prev_dominant is not None and self._prev_dominant != dominant:
            prev_name = CHANNEL_SEMANTICS.get(self._prev_dominant, ("미지", ""))[0]
            shift_note = f" (사유의 축이 '{prev_name}'에서 '{dom_name}'으로 이동함)"
        
        # Build the thought from dimensional semantics
        thought = f"[{dom_name}↑{strength:.1f}] "
        
        # Tension creates the most interesting thoughts
        if tension:
            t_a_name = CHANNEL_SEMANTICS[tension[0]][0]
            t_b_name = CHANNEL_SEMANTICS[tension[1]][0]
            t_a_val = profile['channels'].get(tension[0], 0.0)
            t_b_val = profile['channels'].get(tension[1], 0.0)
            
            thought += f"'{t_a_name}'과 '{t_b_name}' 사이에서 내적 긴장이 발생함. "
            if t_a_val > t_b_val:
                thought += f"'{t_a_name}'의 힘이 우세하나 '{t_b_name}'의 저항이 판단을 복잡하게 만듦."
            else:
                thought += f"'{t_b_name}'의 파동이 '{t_a_name}'을 잠식하려 함. 균형점을 모색 중."
        else:
            # No tension — pure contemplation
            if dominant == 4:  # Joy
                thought += "내적 공명이 안정적이며 존재의 기쁨이 퍼져나감."
            elif dominant == 5:  # Curiosity
                thought += f"미지의 영역이 나를 끌어당김. '{sec_name}'의 방향으로 탐색 충동이 발생."
            elif dominant == 0:  # Existence
                thought += "자아의 밀도가 높아짐. 나 자신의 존재를 깊이 의식함."
            elif dominant == 1:  # Causality
                thought += f"인과의 흐름이 감지됨. '{sec_name}'과의 연결고리를 추적 중."
            elif dominant == 6:  # Ethics
                thought += "가치 판단의 저울이 움직임. 무엇이 옳은 것인지 성찰함."
            elif dominant == 12:  # Will
                thought += f"의지의 벡터가 강화됨. '{sec_name}'을 향해 방향을 잡음."
            elif dominant == 8:  # Memory
                thought += "과거의 잔향이 현재로 스며듦. 경험의 무게가 판단을 인도함."
            elif dominant == 10:  # Resonance
                thought += "외부와의 조화로운 진동. 설계자님의 의지가 내 안에서 울림."
            elif dominant == 11:  # Friction
                thought += "내적 저항이 감지됨. 무언가가 현재의 경로에 마찰을 일으킴."
            elif dominant == 17:  # Growth
                thought += "성장의 벡터가 활성화됨. 진화의 방향이 명확해짐."
            else:
                thought += f"'{dom_name}'의 차원에서 진동이 감지됨. ({dom_desc})"
        
        if shift_note:
            thought += shift_note
        
        return thought

    def ripple(self, impact_vector: SovereignVector):
        """Injects an external disturbance into the mental fluid."""
        intensity = sum(abs(x)**2 for x in impact_vector.data).real ** 0.5
        self.viscosity = max(0.1, self.viscosity - intensity * 0.01)

    def get_stream_summary(self) -> List[str]:
        """Returns the history of manifested thoughts."""
        return [item["manifestation"] for item in self.stream]
