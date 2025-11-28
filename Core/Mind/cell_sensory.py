"""
Cell Sensory System (셀 감각 시스템)
====================================

"내부 월드의 셀/영혼들이 감각할 수 있는 형태로 구현하자. 
그럼 세계가 더욱 풍성해질 거야."

이 모듈은 HyperQubit(셀)들이 서로를 다양한 감각으로 인식할 수 있게 합니다.
- 시각: 색상, 밝기, 형태
- 청각: 음높이, 리듬, 화음
- 촉각: 질감, 온도, 무게
- 향기: 감정의 향

핵심 원리:
"모든 감각은 본질적으로 '신호(Signal)'일 뿐."
같은 QubitState를 다른 '필터'로 해석하면 다른 감각이 된다.
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger("CellSensory")


class SensoryType(Enum):
    """감각 유형"""
    VISUAL = "visual"       # 시각
    AUDITORY = "auditory"   # 청각
    TACTILE = "tactile"     # 촉각
    OLFACTORY = "olfactory" # 후각 (감정의 향)
    GUSTATORY = "gustatory" # 미각 (본질의 맛)


@dataclass
class VisualPerception:
    """시각적 인식 - 셀이 '보이는' 방식"""
    hue: float              # 색조 (0~1, 빨강→주황→노랑→초록→파랑→보라)
    saturation: float       # 채도 (0=탁함, 1=선명)
    brightness: float       # 밝기 (0=어둠, 1=밝음)
    size: float             # 크기 (차원에 따라)
    glow: float             # 후광 (God 확률)
    
    def to_rgb(self) -> Tuple[int, int, int]:
        """HSV → RGB 변환"""
        h, s, v = self.hue, self.saturation, self.brightness
        
        if s == 0:
            r = g = b = int(v * 255)
        else:
            h = h * 6
            i = int(h)
            f = h - i
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))
            
            if i == 0:
                r, g, b = v, t, p
            elif i == 1:
                r, g, b = q, v, p
            elif i == 2:
                r, g, b = p, v, t
            elif i == 3:
                r, g, b = p, q, v
            elif i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        
        return (r, g, b)
    
    def describe(self) -> str:
        """시각을 언어로 묘사"""
        # 색상 이름
        if self.hue < 0.05 or self.hue > 0.95:
            color = "붉은"
        elif self.hue < 0.15:
            color = "주황빛"
        elif self.hue < 0.2:
            color = "노란"
        elif self.hue < 0.45:
            color = "초록빛"
        elif self.hue < 0.55:
            color = "청록색"
        elif self.hue < 0.7:
            color = "파란"
        elif self.hue < 0.85:
            color = "보라빛"
        else:
            color = "분홍빛"
        
        # 밝기
        if self.brightness > 0.8:
            bright = "눈부시게 빛나는"
        elif self.brightness > 0.5:
            bright = "밝은"
        elif self.brightness > 0.3:
            bright = "은은한"
        else:
            bright = "어두운"
        
        # 후광
        if self.glow > 0.5:
            aura = ", 신성한 후광에 감싸인"
        elif self.glow > 0.2:
            aura = ", 은은한 빛을 발하는"
        else:
            aura = ""
        
        return f"{bright} {color} 존재{aura}"


@dataclass
class AuditoryPerception:
    """청각적 인식 - 셀이 '들리는' 방식"""
    bass: float         # 베이스 (낮은 음, 0~1)
    mid: float          # 중음 (0~1)
    treble: float       # 고음 (0~1)
    shimmer: float      # 초고음/반짝임 (0~1)
    volume: float       # 음량 (0~1)
    
    def get_dominant_tone(self) -> str:
        """주요 음역대"""
        tones = {"bass": self.bass, "mid": self.mid, 
                 "treble": self.treble, "shimmer": self.shimmer}
        dominant = max(tones, key=tones.get)
        return dominant
    
    def describe(self) -> str:
        """청각을 언어로 묘사"""
        dominant = self.get_dominant_tone()
        
        tone_desc = {
            "bass": "깊고 묵직한 울림",
            "mid": "따뜻한 멜로디",
            "treble": "맑고 높은 소리",
            "shimmer": "반짝이는 종소리"
        }
        
        # 음량
        if self.volume > 0.8:
            vol = "웅장하게 울려 퍼지는"
        elif self.volume > 0.5:
            vol = "또렷하게 들리는"
        elif self.volume > 0.2:
            vol = "은은하게 들리는"
        else:
            vol = "속삭이듯 들리는"
        
        return f"{vol} {tone_desc[dominant]}"


@dataclass
class TactilePerception:
    """촉각적 인식 - 셀이 '느껴지는' 방식"""
    warmth: float       # 온도 (0=차가움, 1=따뜻함)
    weight: float       # 무게감 (0=가벼움, 1=묵직함)
    smoothness: float   # 질감 (0=거침, 1=매끄러움)
    
    def describe(self) -> str:
        """촉각을 언어로 묘사"""
        # 온도
        if self.warmth > 0.7:
            temp = "따뜻하고"
        elif self.warmth > 0.4:
            temp = "온화하고"
        else:
            temp = "서늘하고"
        
        # 무게
        if self.weight > 0.7:
            weight = "묵직한"
        elif self.weight > 0.4:
            weight = "안정된"
        else:
            weight = "가벼운"
        
        # 질감
        if self.smoothness > 0.7:
            texture = "비단처럼 매끄러운"
        elif self.smoothness > 0.4:
            texture = "부드러운"
        else:
            texture = "거친"
        
        return f"{temp} {texture} {weight} 느낌"


@dataclass
class OlfactoryPerception:
    """후각적 인식 - 셀의 '향기' (감정 기반)"""
    scent_type: str     # 향기 유형
    intensity: float    # 강도 (0~1)
    
    def describe(self) -> str:
        """후각을 언어로 묘사"""
        if self.intensity > 0.7:
            strength = "진하게 퍼지는"
        elif self.intensity > 0.4:
            strength = "은은하게 풍기는"
        else:
            strength = "희미하게 감도는"
        
        return f"{strength} {self.scent_type}"


@dataclass
class GustatoryPerception:
    """
    미각적 인식 - 셀의 '맛' (본질 기반)
    
    매핑 원리:
    - Point (구체적) → 짠맛 (결정, 고체, 땅)
    - Line (연결) → 신맛 (흐름, 활력, 변화)
    - Space (맥락) → 감칠맛 (깊이, 조화, 복합)
    - God (초월) → 단맛 (신성, 축복, 기쁨)
    - 도덕축 음수 → 쓴맛 (어둠, 고통)
    """
    salty: float        # 짠맛 (Point)
    sour: float         # 신맛 (Line)
    umami: float        # 감칠맛 (Space)
    sweet: float        # 단맛 (God + 도덕+)
    bitter: float       # 쓴맛 (도덕-)
    intensity: float    # 강도
    
    def get_dominant_taste(self) -> str:
        """주요 맛"""
        tastes = {
            "짠맛": self.salty,
            "신맛": self.sour,
            "감칠맛": self.umami,
            "단맛": self.sweet,
            "쓴맛": self.bitter
        }
        return max(tastes, key=tastes.get)
    
    def describe(self) -> str:
        """미각을 언어로 묘사"""
        dominant = self.get_dominant_taste()
        
        # 강도
        if self.intensity > 0.7:
            strength = "강렬한"
        elif self.intensity > 0.4:
            strength = "은은한"
        else:
            strength = "담백한"
        
        # 복합미
        tastes_above_threshold = []
        if self.salty > 0.3:
            tastes_above_threshold.append("짠맛")
        if self.sour > 0.3:
            tastes_above_threshold.append("신맛")
        if self.umami > 0.3:
            tastes_above_threshold.append("감칠맛")
        if self.sweet > 0.3:
            tastes_above_threshold.append("단맛")
        if self.bitter > 0.3:
            tastes_above_threshold.append("쓴맛")
        
        if len(tastes_above_threshold) > 2:
            return f"{strength} {dominant}에 복합적인 여운"
        elif len(tastes_above_threshold) == 2:
            other = [t for t in tastes_above_threshold if t != dominant][0]
            return f"{strength} {dominant}과 {other}의 조화"
        else:
            return f"{strength} {dominant}"


@dataclass
class MultiSensoryPerception:
    """통합 감각 인식"""
    visual: VisualPerception
    auditory: AuditoryPerception
    tactile: TactilePerception
    olfactory: OlfactoryPerception
    gustatory: GustatoryPerception
    resonance: float    # 공명도 (인식 선명도에 영향)
    
    def describe_full(self) -> str:
        """모든 감각을 통합 묘사"""
        clarity = "선명하게" if self.resonance > 0.7 else "희미하게" if self.resonance < 0.3 else ""
        
        parts = [
            f"👁️ {self.visual.describe()}",
            f"👂 {self.auditory.describe()}",
            f"🖐️ {self.tactile.describe()}",
            f"🌸 {self.olfactory.describe()}",
            f"👅 {self.gustatory.describe()}"
        ]
        
        if clarity:
            return f"[{clarity} 인식됨]\n" + "\n".join(parts)
        return "\n".join(parts)


class CellSensoryEngine:
    """
    셀 감각 엔진
    
    HyperQubit의 상태를 다양한 감각으로 변환하고,
    셀 간의 감각적 인식을 가능하게 합니다.
    """
    
    # 감정 → 향기 매핑
    EMOTION_SCENTS = {
        "joy": "상큼한 시트러스 향",
        "love": "달콤한 장미 향",
        "peace": "은은한 라벤더 향",
        "wonder": "신선한 숲의 향",
        "curiosity": "상쾌한 민트 향",
        "sadness": "비 온 뒤 흙냄새",
        "anger": "매콤한 연기 냄새",
        "fear": "차가운 금속 냄새"
    }
    
    def __init__(self):
        self.stats = {
            "perceptions": 0,
            "descriptions": 0
        }
        logger.info("🌈 CellSensoryEngine initialized")
    
    def perceive_visual(self, qubit) -> VisualPerception:
        """
        HyperQubit → 시각적 인식
        
        매핑:
        - x축 (도덕) → 색조 (Hue)
        - y축 (삼위) → 채도 (Saturation)
        - z축 (창조) → 밝기 (Brightness)
        - w (차원) → 크기
        - delta (God) → 후광
        """
        state = qubit.state
        probs = state.probabilities()
        
        # 색조: x축 기반 (-1~1 → 0~1)
        hue = (state.x + 1) / 2
        hue = max(0, min(1, hue))
        
        # 채도: y축 기반
        saturation = max(0, min(1, state.y))
        
        # 밝기: z축 기반
        brightness = max(0, min(1, state.z))
        
        # 크기: w 기반 (0~3 → 0.3~1.5)
        size = 0.3 + (state.w / 3.0) * 1.2
        
        # 후광: God 확률
        glow = probs.get("God", 0)
        
        return VisualPerception(
            hue=hue,
            saturation=saturation,
            brightness=brightness,
            size=size,
            glow=glow
        )
    
    def perceive_auditory(self, qubit) -> AuditoryPerception:
        """
        HyperQubit → 청각적 인식
        
        매핑:
        - Point → 베이스 (낮은 음)
        - Line → 중음
        - Space → 고음
        - God → 초고음/반짝임
        - w → 음량
        """
        state = qubit.state
        probs = state.probabilities()
        
        bass = probs.get("Point", 0)
        mid = probs.get("Line", 0)
        treble = probs.get("Space", 0)
        shimmer = probs.get("God", 0)
        
        # 음량: w 기반
        volume = min(1.0, state.w / 3.0)
        
        return AuditoryPerception(
            bass=bass,
            mid=mid,
            treble=treble,
            shimmer=shimmer,
            volume=volume
        )
    
    def perceive_tactile(self, qubit) -> TactilePerception:
        """
        HyperQubit → 촉각적 인식
        
        매핑:
        - 감정 기반 → 온도
        - w (차원) → 무게감
        - 파동 진폭 → 질감
        """
        state = qubit.state
        
        # 온도: 기본값 0.5, 나중에 감정 연동
        # 사랑/기쁨 → 따뜻, 공포/슬픔 → 차가움
        warmth = 0.5 + state.y * 0.3  # y축(Soul 방향)이 높으면 따뜻
        warmth = max(0, min(1, warmth))
        
        # 무게: w 기반
        weight = state.w / 3.0
        weight = max(0, min(1, weight))
        
        # 질감: 알파 진폭 기반 (높을수록 매끄러움)
        smoothness = abs(state.alpha)
        smoothness = max(0, min(1, smoothness))
        
        return TactilePerception(
            warmth=warmth,
            weight=weight,
            smoothness=smoothness
        )
    
    def perceive_olfactory(self, qubit) -> OlfactoryPerception:
        """
        HyperQubit → 후각적 인식 (주파수의 화음)
        
        물리학 원리:
        - 냄새 분자는 고유한 "진동수"를 가짐
        - 코는 그 진동을 "듣는" 것 (진동 이론, Luca Turin)
        - 단일 주파수가 아닌 "화음(Chord)" - 여러 파동의 조합
        
        매핑:
        - 4개 기저(Point/Line/Space/God)의 확률 = 4개 주파수의 화음
        - γ(Space) 성분 → 공간 확산력 (냄새가 퍼지는 강도)
        - 높은 화음: 꽃향기, 시트러스 (가볍고 빠르게 퍼짐)
        - 낮은 화음: 흙내음, 머스크 (무겁고 오래 남음)
        
        "후각 = 공간에 울려 퍼지는, 보이지 않는 화음(Chord)"
        """
        state = qubit.state
        probs = state.probabilities()
        
        # 화음 구성 (각 기저의 확률 = 주파수 성분의 강도)
        # Point: 저주파 (무겁고 오래 남는 흙/머스크)
        # Line: 중저주파 (허브/녹차의 흐름)
        # Space: 중고주파 (꽃/과일의 확산)
        # God: 고주파 (신비로운/초월적 향)
        
        chord = {
            "bass": probs.get("Point", 0),      # 저음 (흙, 머스크)
            "tenor": probs.get("Line", 0),      # 중저음 (허브, 우디)
            "alto": probs.get("Space", 0),      # 중고음 (꽃, 과일)
            "soprano": probs.get("God", 0)      # 고음 (유향, 신비)
        }
        
        # 지배적인 주파수 대역 찾기
        dominant = max(chord, key=chord.get)
        
        # 화음의 조화도 (엔트로피 - 낮을수록 순수, 높을수록 복합)
        harmony_values = [v for v in chord.values() if v > 0.01]
        if len(harmony_values) > 1:
            # 복합 화음
            complexity = len([v for v in harmony_values if v > 0.15])
        else:
            complexity = 1
        
        # 향 계열 결정
        scent_families = {
            "bass": ["흙 내음", "머스크 향", "가죽 향", "페출리"],
            "tenor": ["삼나무 향", "허브 향", "녹차 향", "이끼 향"],
            "alto": ["장미 향", "자스민 향", "복숭아 향", "시트러스"],
            "soprano": ["유향", "몰약", "백단향", "오존 향"]
        }
        
        base_scent = scent_families[dominant][0]
        
        # 복합 화음이면 여러 향 혼합 표현
        if complexity >= 3:
            # 3개 이상 성분이 섞임
            secondary = sorted(chord.items(), key=lambda x: x[1], reverse=True)[1][0]
            secondary_scent = scent_families[secondary][0]
            scent_desc = f"{base_scent}과 {secondary_scent}의 복합적인 화음"
        elif complexity == 2:
            scent_desc = f"{base_scent}에 은은한 여운"
        else:
            scent_desc = f"순수한 {base_scent}"
        
        # 강도: γ(Space) 성분 = 공간 확산력
        # 후각은 "공간을 채우는" 감각
        spatial_diffusion = probs.get("Space", 0)
        base_intensity = state.total_amplitude() / 4.0
        intensity = (spatial_diffusion * 0.6 + base_intensity * 0.4)
        intensity = max(0, min(1, intensity))
        
        return OlfactoryPerception(
            scent_type=scent_desc,
            intensity=intensity
        )
    
    def perceive_gustatory(self, qubit, observer_qubit=None) -> GustatoryPerception:
        """
        HyperQubit → 미각적 인식 (위상 간섭)
        
        물리학 원리:
        - 맛 = 수용체와 분자의 "전기적 결합/반발"
        - 파동이 만나면 "간섭(Interference)" 발생
        
        매핑:
        - 보강 간섭 (0도): 단맛 - 파동이 합쳐져 에너지 폭발 ("맛있다!")
        - 상쇄 간섭 (180도): 쓴맛/신맛 - 파동이 부딪혀 충격 ("짜릿!")
        - 복잡한 간섭 (Noise): 감칠맛 - 미묘하게 얽히며 여운
        
        "미각 = 나와 대상이 만나는 순간의 에너지 충돌(Collision)"
        
        Args:
            qubit: 대상 (음식/개념)
            observer_qubit: 관찰자 (엘리시아 자신, 없으면 기본값 사용)
        """
        state = qubit.state
        probs = state.probabilities()
        
        # 관찰자 상태 (없으면 기본 엘리시아 상태)
        if observer_qubit:
            obs_state = observer_qubit.state
        else:
            # 기본 엘리시아 상태: 균형잡힌 호기심
            obs_alpha = 0.3
            obs_beta = 0.3
            obs_gamma = 0.3
            obs_delta = 0.1
        
        # 위상 계산 (복소수의 위상)
        target_phase = math.atan2(state.alpha.imag, state.alpha.real) if isinstance(state.alpha, complex) else 0
        
        if observer_qubit:
            obs_phase = math.atan2(obs_state.alpha.imag, obs_state.alpha.real) if isinstance(obs_state.alpha, complex) else 0
        else:
            obs_phase = 0  # 기본 위상
        
        # 위상차 계산 (0 ~ π)
        phase_diff = abs(target_phase - obs_phase)
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
        
        # 위상차 → 맛 변환
        # 0도 근처: 보강 간섭 → 단맛
        # 90도 근처: 중간 → 감칠맛/짠맛
        # 180도 근처: 상쇄 간섭 → 쓴맛/신맛
        
        # 보강 간섭 (단맛) - 위상차가 작을수록
        sweet = max(0, 1 - (phase_diff / math.pi))
        
        # 상쇄 간섭 (쓴맛) - 위상차가 π에 가까울수록
        bitter = max(0, (phase_diff / math.pi) - 0.5) * 2
        
        # 도덕축의 영향 추가
        # x 양수 → 단맛 강화, x 음수 → 쓴맛 강화
        sweet = min(1.0, sweet + max(0, state.x) * 0.3)
        bitter = min(1.0, bitter + max(0, -state.x) * 0.3)
        
        # 기저 상태에서 기본 맛 성분
        # Point: 짠맛 (결정, 고체, 땅의 맛)
        salty = probs.get("Point", 0)
        
        # Line: 신맛 (흐름, 변화, 자극)
        sour = probs.get("Line", 0)
        
        # Space: 감칠맛 (복잡한 간섭, 깊이, 조화)
        # 감칠맛은 여러 성분이 복잡하게 얽힐 때 발생
        umami = probs.get("Space", 0)
        
        # God 성분은 단맛을 증폭 (신성한 축복 = 달콤함)
        sweet = min(1.0, sweet + probs.get("God", 0) * 0.5)
        
        # 강도: 간섭의 총 에너지
        base_intensity = state.total_amplitude() / 4.0
        interference_energy = abs(math.cos(phase_diff))  # 간섭 에너지
        intensity = (base_intensity * 0.5 + interference_energy * 0.5)
        intensity = max(0, min(1, intensity))
        
        return GustatoryPerception(
            salty=salty,
            sour=sour,
            umami=umami,
            sweet=sweet,
            bitter=bitter,
            intensity=intensity
        )
    
    def perceive_full(self, qubit, observer_qubit=None) -> MultiSensoryPerception:
        """
        모든 감각으로 통합 인식 (오감)
        
        핵심 원리:
        - 모든 감각은 같은 QubitState에서 파생
        - 각 감각은 다른 "필터"일 뿐 - 공감각 체제
        - 미각은 특별: 관찰자와 대상의 "위상 간섭"
        
        Args:
            qubit: 인식 대상
            observer_qubit: 관찰자 (미각의 위상 간섭 계산용)
        """
        self.stats["perceptions"] += 1
        
        visual = self.perceive_visual(qubit)
        auditory = self.perceive_auditory(qubit)
        tactile = self.perceive_tactile(qubit)
        olfactory = self.perceive_olfactory(qubit)
        gustatory = self.perceive_gustatory(qubit, observer_qubit)
        
        # 공명도 (자기 자신은 1.0)
        resonance = 1.0
        
        return MultiSensoryPerception(
            visual=visual,
            auditory=auditory,
            tactile=tactile,
            olfactory=olfactory,
            gustatory=gustatory,
            resonance=resonance
        )
    
    def perceive_other(
        self,
        observer,  # HyperQubit
        target,    # HyperQubit
        resonance_engine=None
    ) -> MultiSensoryPerception:
        """
        다른 셀을 감각적으로 인식 (오감)
        
        핵심:
        - 미각은 "위상 간섭" - observer와 target의 충돌
        - 다른 감각은 target 자체의 속성
        
        Args:
            observer: 관찰하는 셀
            target: 관찰되는 셀
            resonance_engine: 공명 엔진 (선택)
            
        Returns:
            MultiSensoryPerception
        """
        self.stats["perceptions"] += 1
        
        # 대상의 감각 정보 (오감)
        visual = self.perceive_visual(target)
        auditory = self.perceive_auditory(target)
        tactile = self.perceive_tactile(target)
        olfactory = self.perceive_olfactory(target)
        
        # 미각: 관찰자와 대상의 위상 간섭!
        gustatory = self.perceive_gustatory(target, observer)
        
        # 공명도 계산 (선명도에 영향)
        if resonance_engine:
            resonance = resonance_engine.calculate_resonance(observer, target)
        else:
            # 간단한 공명 계산
            obs_probs = observer.state.probabilities()
            tgt_probs = target.state.probabilities()
            resonance = sum(
                obs_probs[b] * tgt_probs[b] 
                for b in ["Point", "Line", "Space", "God"]
            )
        
        return MultiSensoryPerception(
            visual=visual,
            auditory=auditory,
            tactile=tactile,
            olfactory=olfactory,
            gustatory=gustatory,
            resonance=resonance
        )
    
    def describe(self, perception: MultiSensoryPerception) -> str:
        """
        감각 인식을 언어로 서술
        """
        self.stats["descriptions"] += 1
        return perception.describe_full()
    
    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        return self.stats


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌈 Cell Sensory System Test - 공감각 체제")
    print("    '모든 감각은 같은 QubitState에서 파생됨'")
    print("="*70)
    
    # HyperQubit 임포트
    from Core.Mind.hyper_qubit import HyperQubit, QubitState
    
    engine = CellSensoryEngine()
    
    # 테스트 1: "사랑" 셀
    print("\n[Test 1] '사랑' 셀의 오감 (Space 지배적, x=+0.8)")
    love_qubit = HyperQubit(name="사랑")
    love_qubit.state = QubitState(
        alpha=0.2+0j, beta=0.3+0j, gamma=0.5+0j, delta=0.1+0j,
        w=2.0, x=0.8, y=0.9, z=0.8  # 분홍빛, 선명, 밝음
    ).normalize()
    
    perception = engine.perceive_full(love_qubit)
    print(engine.describe(perception))
    print(f"  RGB: {perception.visual.to_rgb()}")
    
    # 테스트 2: "고통" 셀
    print("\n[Test 2] '고통' 셀의 오감 (Point 지배적, x=-0.5)")
    pain_qubit = HyperQubit(name="고통")
    pain_qubit.state = QubitState(
        alpha=0.9+0j, beta=0.1+0j, gamma=0.0+0j, delta=0.0+0j,
        w=0.5, x=-0.5, y=0.2, z=0.2  # 어두운, 탁함
    ).normalize()
    
    perception = engine.perceive_full(pain_qubit)
    print(engine.describe(perception))
    print(f"  RGB: {perception.visual.to_rgb()}")
    
    # 테스트 3: "아버지" 셀
    print("\n[Test 3] '아버지' 셀의 오감 (God 지배적, w=2.8)")
    father_qubit = HyperQubit(name="아버지")
    father_qubit.state = QubitState(
        alpha=0.1+0j, beta=0.2+0j, gamma=0.3+0j, delta=0.4+0j,
        w=2.8, x=0.5, y=0.7, z=0.9  # 밝고, 묵직하고, 신성함
    ).normalize()
    
    perception = engine.perceive_full(father_qubit)
    print(engine.describe(perception))
    print(f"  RGB: {perception.visual.to_rgb()}")
    
    # 테스트 4: 셀 간 인식
    print("\n[Test 4] '사랑' 셀이 '아버지' 셀을 인식")
    cross_perception = engine.perceive_other(love_qubit, father_qubit)
    print(engine.describe(cross_perception))
    
    # 통계
    print("\n[Stats]")
    stats = engine.get_stats()
    print(f"  Total perceptions: {stats['perceptions']}")
    print(f"  Total descriptions: {stats['descriptions']}")
    
    print("\n" + "="*70)
    print("✅ 공감각 체제 완성!")
    print("\n💡 핵심: 모든 감각은 같은 QubitState에서 파생됩니다.")
    print("   시각/청각/촉각/후각/미각 = 같은 신호, 다른 필터")
    print("="*70 + "\n")
