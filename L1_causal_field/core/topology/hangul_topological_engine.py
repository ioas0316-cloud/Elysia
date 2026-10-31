"""
hangul_topological_engine.py — 훈민정음 위상 사영 및 조음 마찰장 토크나이저 엔진
=============================================================================
훈민정음(訓民正音) 제자 원리(制字原理)에 기반하여 한글 음절을
단순 기호가 아닌 물리적·위상적 파동으로 사영(Project)하는 코어 엔진.

핵심 원리:
1. 삼재(三才: 天·地·人) 3상 평형 회전:
   - 중성(모음)의 근원인 天(·, 발산/양), 地(ㅡ, 수평/안정/음), 人(ㅣ, 수직/매개/중용)을
     120도(2π/3) 위상차를 갖는 3상 평형 결선 및 2x2 스핀 행렬로 결합.
2. 오음(五音: 牙·舌·脣·齒·喉) 조음 기관 상형과 마찰 장(Friction Field):
   - 초성/종성(자음)의 발음 기관 상형 원리(ㄱ, ㄴ, ㅁ, ㅅ, ㅇ)에 따른
     물리적 저항, 음향 주파수, 기계적 장력 계수를 2x2 스핀 행렬과 결합.
3. 3중 크로네커 텐서(Kronecker Tensor Product):
   - (초성 2x2) ⊗ (중성 2x2) ⊗ (종성 2x2) = (8x8) = 64차원 위상 파수 벡터.
   - 조음 마찰 계수 및 3상 위상 균형이 보존된 단일 파동 벡터 생성.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np


class JamoArticulationProfile(NamedTuple):
    """
    조음 기관 물리 마찰 및 저항 특성
    - friction: 기계적 저항/마찰력 (0.0 ~ 1.0)
    - acoustic_freq: 기본 공진 주파수 (Hz)
    - tension: 기류 폐쇄/개방 장력 (0.0 ~ 1.0)
    - organ: 조음 기관 분류 (아, 설, 순, 치, 후)
    """
    friction: float
    acoustic_freq: float
    tension: float
    organ: str


# 훈민정음 오음(五音) 발음 기관 상형 분류 및 물리적 마찰 프로필
# ㄱ(아음-목구멍을 막음, 높은 장력), ㄴ(설음-혀가 윗잇몸에 붙음, 중간 마찰),
# ㅁ(순음-입술 모양, 부드러운 폐쇄), ㅅ(치음-이 모양, 마찰음/날카로움), ㅇ(후음-목구멍 뚫림, 무저항 통전)
ARTICULATION_PROFILES: Dict[str, JamoArticulationProfile] = {
    # 아음 (牙音 - 어금닛소리): 목구멍을 막아 기류 차단 후 파열
    "ㄱ": JamoArticulationProfile(friction=0.75, acoustic_freq=350.0, tension=0.80, organ="牙"),
    "ㄲ": JamoArticulationProfile(friction=0.90, acoustic_freq=420.0, tension=0.95, organ="牙"),
    "ㅋ": JamoArticulationProfile(friction=0.85, acoustic_freq=390.0, tension=0.85, organ="牙"),
    # 설음 (舌音 - 혓소리): 혀가 윗잇몸에 닿는 탄성
    "ㄴ": JamoArticulationProfile(friction=0.40, acoustic_freq=280.0, tension=0.35, organ="舌"),
    "ㄷ": JamoArticulationProfile(friction=0.70, acoustic_freq=330.0, tension=0.75, organ="舌"),
    "ㄸ": JamoArticulationProfile(friction=0.88, acoustic_freq=410.0, tension=0.92, organ="舌"),
    "ㅌ": JamoArticulationProfile(friction=0.80, acoustic_freq=370.0, tension=0.82, organ="舌"),
    "ㄹ": JamoArticulationProfile(friction=0.30, acoustic_freq=260.0, tension=0.25, organ="舌"),
    # 순음 (脣音 - 입술소리): 입술의 맞닿음과 포용
    "ㅁ": JamoArticulationProfile(friction=0.35, acoustic_freq=220.0, tension=0.30, organ="脣"),
    "ㅂ": JamoArticulationProfile(friction=0.65, acoustic_freq=290.0, tension=0.70, organ="脣"),
    "ㅃ": JamoArticulationProfile(friction=0.85, acoustic_freq=380.0, tension=0.90, organ="脣"),
    "ㅍ": JamoArticulationProfile(friction=0.78, acoustic_freq=340.0, tension=0.80, organ="脣"),
    # 치음 (齒音 - 잇소리): 이 사이로 새어나가는 날카로운 마찰
    "ㅅ": JamoArticulationProfile(friction=0.80, acoustic_freq=750.0, tension=0.65, organ="齒"),
    "ㅆ": JamoArticulationProfile(friction=0.92, acoustic_freq=850.0, tension=0.88, organ="齒"),
    "ㅈ": JamoArticulationProfile(friction=0.70, acoustic_freq=550.0, tension=0.70, organ="齒"),
    "ㅉ": JamoArticulationProfile(friction=0.88, acoustic_freq=680.0, tension=0.92, organ="齒"),
    "ㅊ": JamoArticulationProfile(friction=0.82, acoustic_freq=620.0, tension=0.82, organ="齒"),
    # 후음 (喉音 - 목구멍소리): 텅 비어 막힘없이 통전되는 근원
    "ㅇ": JamoArticulationProfile(friction=0.05, acoustic_freq=180.0, tension=0.05, organ="喉"),
    "ㅎ": JamoArticulationProfile(friction=0.45, acoustic_freq=320.0, tension=0.40, organ="喉"),
    # 종성 전용 빈소리(종성 없음)
    "": JamoArticulationProfile(friction=0.0, acoustic_freq=0.0, tension=0.0, organ="空")
}

# 19개 초성(初聲) 순서표
CHO_LIST = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

# 21개 중성(中聲) 순서표
JUNG_LIST = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]

# 28개 종성(終聲) 순서표 (0번째는 받침 없음)
JONG_LIST = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]


class HangulTopologicalEngine:
    """
    훈민정음 위상 사영 엔진
    - 삼재(천지인) 3상 위상각 및 2x2 스핀 행렬 계산
    - 오음 조음 마찰장 가중치 결합
    - 3중 크로네커 텐서곱 (64차원 위상 파동) 생성
    """

    def __init__(self, target_dim: int = 64):
        assert target_dim == 64, "3중 2x2 크로네커 곱 (2^3 x 2^3)은 정확히 64차원이어야 합니다."
        self.target_dim = target_dim

        # 삼재(三才) 3상 기저 각도: 120도(2π/3) 등간격 위상 배치
        # 天(·, 양/발산): 0도
        # 地(ㅡ, 음/수평): 120도 (2π/3)
        # 人(ㅣ, 중용/수직): 240도 (4π/3)
        self.phase_tian = 0.0
        self.phase_di = 2.0 * math.pi / 3.0
        self.phase_ren = 4.0 * math.pi / 3.0

        # 중성 21자의 삼재 결합 분해표 (천·지·인 비중 [w_tian, w_di, w_ren])
        self.vowel_sancai_map = self._build_vowel_sancai_map()

    def _build_vowel_sancai_map(self) -> Dict[str, Tuple[float, float, float]]:
        """
        중성 21자의 천지인(天地人) 구성비 매핑.
        기본 모음:
          ㅡ (地): [0.0, 1.0, 0.0]
          ㅣ (人): [0.0, 0.0, 1.0]
          · (天):  [1.0, 0.0, 0.0]
        초출자(初出字):
          ㅗ (· + ㅡ): [0.5, 0.5, 0.0]
          ㅏ (ㅣ + ·): [0.5, 0.0, 0.5]
          ㅜ (ㅡ + ·): [0.5, 0.5, 0.0]
          ㅓ (· + ㅣ): [0.5, 0.0, 0.5]
        재출자(再出字):
          ㅛ (·· + ㅡ): [0.7, 0.3, 0.0]
          ㅑ (ㅣ + ··): [0.7, 0.0, 0.3]
          ㅠ (ㅡ + ··): [0.7, 0.3, 0.0]
          ㅕ (·· + ㅣ): [0.7, 0.0, 0.3]
        복합 모음:
          ㅐ, ㅔ, ㅘ, ㅙ, ㅚ, ㅒ, ㅖ, ㅝ, ㅞ, ㅟ, ㅢ
        """
        base = {
            "ㅡ": (0.0, 1.0, 0.0),
            "ㅣ": (0.0, 0.0, 1.0),
            "ㅏ": (0.5, 0.0, 0.5),
            "ㅓ": (0.5, 0.0, 0.5),
            "ㅗ": (0.5, 0.5, 0.0),
            "ㅜ": (0.5, 0.5, 0.0),
            "ㅑ": (0.7, 0.0, 0.3),
            "ㅕ": (0.7, 0.0, 0.3),
            "ㅛ": (0.7, 0.3, 0.0),
            "ㅠ": (0.7, 0.3, 0.0),
            "ㅐ": (0.4, 0.1, 0.5),
            "ㅔ": (0.4, 0.2, 0.4),
            "ㅒ": (0.6, 0.1, 0.3),
            "ㅖ": (0.6, 0.1, 0.3),
            "ㅘ": (0.5, 0.3, 0.2),
            "ㅙ": (0.5, 0.2, 0.3),
            "ㅚ": (0.4, 0.3, 0.3),
            "ㅝ": (0.5, 0.3, 0.2),
            "ㅞ": (0.5, 0.2, 0.3),
            "ㅟ": (0.4, 0.3, 0.3),
            "ㅢ": (0.1, 0.5, 0.4),
        }
        return base

    def decompose_syllable(self, char: str) -> Optional[Tuple[str, str, str]]:
        """한글 음절을 (초성, 중성, 종성) 문자로 분해"""
        code = ord(char) - 0xAC00
        if code < 0 or code > 11171:
            return None
        cho_idx = code // 588
        jung_idx = (code % 588) // 28
        jong_idx = code % 28
        return CHO_LIST[cho_idx], JUNG_LIST[jung_idx], JONG_LIST[jong_idx]

    def compute_vowel_phase(self, vowel: str) -> float:
        """
        중성의 삼재(천지인) 3상 결합에 따른 고유 위상각 계산:
        θ = arctan2( Σ w_i * sin(φ_i), Σ w_i * cos(φ_i) )
        """
        weights = self.vowel_sancai_map.get(vowel, (0.333, 0.333, 0.333))
        w_t, w_d, w_r = weights

        # 3상 벡터 합성
        x = (
            w_t * math.cos(self.phase_tian)
            + w_d * math.cos(self.phase_di)
            + w_r * math.cos(self.phase_ren)
        )
        y = (
            w_t * math.sin(self.phase_tian)
            + w_d * math.sin(self.phase_di)
            + w_r * math.sin(self.phase_ren)
        )
        phase = math.atan2(y, x)
        return float(phase % (2.0 * math.pi))

    def build_consonant_spin_matrix(self, consonant: str) -> np.ndarray:
        """
        자음의 조음 마찰장 프로필을 결합한 2x2 스핀 행렬:
        - friction(마찰)과 tension(장력)으로 스핀 회전 및 감쇠 계수 결정
        - S = [ [cos(θ), -sin(θ) * (1 - friction)],
                [sin(θ) * (1 - tension), cos(θ)] ]
        """
        # 복합 받침(ㄳ, ㄵ 등)의 경우 첫소리 기준 분해
        lead_char = consonant[0] if consonant else ""
        profile = ARTICULATION_PROFILES.get(lead_char, ARTICULATION_PROFILES[""])

        # 주파수 기반 위상 각도
        theta = (profile.acoustic_freq / 1000.0) * math.pi
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        f_factor = max(0.1, 1.0 - 0.5 * profile.friction)
        t_factor = max(0.1, 1.0 - 0.5 * profile.tension)

        spin_mat = np.array([
            [cos_t, -sin_t * f_factor],
            [sin_t * t_factor, cos_t]
        ], dtype=np.float32)

        # 에너지 보존을 위한 Frobenius 정규화 (스핀 유니터리성 유지)
        norm = np.linalg.norm(spin_mat)
        if norm > 1e-6:
            spin_mat = (spin_mat / norm) * math.sqrt(2.0)

        return spin_mat

    def build_vowel_spin_matrix(self, vowel: str) -> np.ndarray:
        """
        모음의 3상 천지인 위상각을 갖는 2x2 유니터리 회전 행렬:
        R(θ) = [ [cos(θ), -sin(θ)],
                 [sin(θ),  cos(θ)] ]
        """
        theta = self.compute_vowel_phase(vowel)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        return np.array([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ], dtype=np.float32)

    def encode_syllable(self, char: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        단일 한글 음절을 64차원 위상 파동 및 물리적 마찰 지표로 인코딩.
        Returns:
            wavevector: [64] float32 1D array
            physics_metrics: {
                "friction_mean": float,
                "tension_mean": float,
                "vowel_phase": float,
                "coherence": float
            }
        """
        decomposed = self.decompose_syllable(char)
        if decomposed is None:
            # 한글 이외의 문자는 제로 파동 반환
            return np.zeros(self.target_dim, dtype=np.float32), {
                "friction_mean": 0.0,
                "tension_mean": 0.0,
                "vowel_phase": 0.0,
                "coherence": 0.0
            }

        cho, jung, jong = decomposed

        # 1. 초성 스핀 행렬
        M_cho = self.build_consonant_spin_matrix(cho)
        # 2. 중성 3상 위상 스핀 행렬
        M_jung = self.build_vowel_spin_matrix(jung)
        # 3. 종성 스핀 행렬
        M_jong = self.build_consonant_spin_matrix(jong)

        # 4. 3중 크로네커 텐서곱 (2x2) ⊗ (2x2) ⊗ (2x2) = (8x8) = 64차원
        M_4x4 = np.kron(M_cho, M_jung)
        M_8x8 = np.kron(M_4x4, M_jong)
        wavevector = M_8x8.reshape(-1)

        # 파동 정규화
        norm = np.linalg.norm(wavevector)
        if norm > 1e-6:
            wavevector = wavevector / norm

        # 물리적 지표 도출
        cho_lead = cho[0] if cho else ""
        jong_lead = jong[0] if jong else ""
        prof_cho = ARTICULATION_PROFILES.get(cho_lead, ARTICULATION_PROFILES[""])
        prof_jong = ARTICULATION_PROFILES.get(jong_lead, ARTICULATION_PROFILES[""])

        friction_mean = float((prof_cho.friction + prof_jong.friction) / 2.0)
        tension_mean = float((prof_cho.tension + prof_jong.tension) / 2.0)
        vowel_phase = self.compute_vowel_phase(jung)

        # 위상 결맞음도(Coherence): 3상 위상과 조음 마찰의 상호작용 지수
        coherence = float(math.cos(vowel_phase) * (1.0 - friction_mean))

        metrics = {
            "friction_mean": friction_mean,
            "tension_mean": tension_mean,
            "vowel_phase": vowel_phase,
            "coherence": coherence
        }

        return wavevector, metrics

    def encode_text(self, text: str) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        문장 전체를 [Seq_len, 64] 형태의 위상 파동 스트림으로 인코딩
        """
        waves = []
        metrics_list = []
        for char in text:
            w, m = self.encode_syllable(char)
            waves.append(w)
            metrics_list.append(m)

        if not waves:
            return np.zeros((0, self.target_dim), dtype=np.float32), []

        return np.array(waves, dtype=np.float32), metrics_list

    def __call__(self, text: str) -> np.ndarray:
        waves, _ = self.encode_text(text)
        return waves
