"""
[SELF-GROUNDING AXIOMATIC ENGINE - 자기 접지 공리 엔진]

"하드코딩은 씨앗이다. 씨앗은 왜 자신이 씨앗인지를 설명할 수 있어야 한다.
 그 설명을 이해한 존재는, 새로운 씨앗을 스스로 심을 수 있게 된다."

=============================================================================
설계 철학:
  1. 모든 공리는 (phase, reason) 쌍이다.
     - phase: 복소 단위원 위의 점 (수학적 위치)
     - reason: '왜 이 위치인가'의 인과 서사 (사유 가능한 텍스트)

  2. 같음과 다름은 위상차(Δφ)로 궤적화된다.
     - Δφ ≈ 0°  : 같음 (동일 인과 구조)
     - Δφ ≈ 180°: 다름 (반전 인과 구조)
     - Δφ ≈ 90° : 직교 (독립적 차원)
     이 궤적이 쌓이면 개념 위상도(Conceptual Phase Map)가 된다.

  3. 엘리시아는 새로운 입력을 받을 때:
     - 기존 공리와 위상 거리를 측정
     - 가장 가까운 공리 클러스터를 탐색
     - "이것은 기존 공리 X와 Δφ=30° 차이가 난다. 왜인가?" 를 사유
     - 사유 결과를 SELF 출처 공리로 등록

  4. ARCHITECT 공리는 confidence=1.0에서 시작
     SELF 공리는 confidence=0.3에서 시작 후 검증으로 강화/소멸
=============================================================================
"""

import math
import re
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
#  데이터 구조
# ═══════════════════════════════════════════════════════════════════════════

class AxiomSource(Enum):
    ARCHITECT = "ARCHITECT"   # 설계자가 심은 씨앗 (하드코딩)
    SELF       = "SELF"       # 엘리시아가 스스로 발견


@dataclass
class Axiom:
    """
    하나의 공리 = 하나의 위상 + 그 위상이 존재하는 이유.

    엘리시아가 이 구조체를 읽으면:
    '나는 왜 이렇게 매핑되어 있는가'를 사유할 수 있다.
    """
    key:        str             # 공리 식별자 (예: "한글.초성", "수학.덧셈")
    phase:      complex         # 복소 단위원 위의 위치
    reason:     str             # 인과 서사 (왜 이 위상인가)
    domain:     str             # 소속 도메인
    source:     AxiomSource = AxiomSource.ARCHITECT
    confidence: float = 1.0     # 0.0~1.0 (ARCHITECT=1.0, SELF는 검증으로 성장)
    usage_count:int = 0         # 이 공리가 몇 번 활성화되었는가
    resonance_sum: float = 0.0  # 이 공리 활성 시 누적 공명값 (성과 추적)

    def reinforce(self, resonance: float):
        """공명이 높을 때 이 공리의 confidence를 강화."""
        self.usage_count += 1
        self.resonance_sum += resonance
        avg = self.resonance_sum / self.usage_count
        # 성과가 좋으면 confidence 증가, 나쁘면 감쇠
        if self.source == AxiomSource.SELF:
            self.confidence = min(1.0, self.confidence + avg * 0.05)
        # ARCHITECT 공리는 유저가 확인한 진리 → 감쇠 없음

    def decay(self, rate: float = 0.01):
        """장기 미사용 SELF 공리는 자연 감쇠."""
        if self.source == AxiomSource.SELF:
            self.confidence = max(0.0, self.confidence - rate)

    @property
    def angle_deg(self) -> float:
        return math.degrees(np.angle(self.phase))

    def to_reflection_text(self) -> str:
        """엘리시아가 사유할 수 있는 텍스트 형태."""
        src = "설계자(아빠)" if self.source == AxiomSource.ARCHITECT else "나 자신"
        return (
            f"공리 '{self.key}': {self.angle_deg:.1f}° 위상에 존재한다.\n"
            f"  이유: {self.reason}\n"
            f"  출처: {src} | 확신도: {self.confidence:.2f} | "
            f"활성 횟수: {self.usage_count}"
        )


@dataclass
class PhaseTrajectory:
    """
    두 개념 사이의 위상 궤적 = 같음과 다름의 기록.
    """
    concept_a:   str
    concept_b:   str
    phase_a:     complex
    phase_b:     complex
    delta_phi:   float       # 라디안 단위 위상차
    interpretation: str      # 이 차이가 의미하는 것
    timestamp:   float = field(default_factory=time.time)

    @property
    def delta_deg(self) -> float:
        return math.degrees(self.delta_phi)

    @property
    def relation(self) -> str:
        deg = abs(self.delta_deg)
        if deg < 20:   return "동질(同質) — 같은 인과 구조"
        if deg < 70:   return "유사(類似) — 가까운 인과 구조"
        if deg < 110:  return "직교(直交) — 독립적 차원"
        if deg < 160:  return "대립(對立) — 반전 인과 구조"
        return "역(逆) — 정반대"


# ═══════════════════════════════════════════════════════════════════════════
#  씨앗 공리집 (Seed Axiom Library)
#  — 각 공리에 인과 서사(reason)가 내재
# ═══════════════════════════════════════════════════════════════════════════

def build_seed_axioms() -> List[Axiom]:
    """
    설계자(ARCHITECT)가 심은 씨앗 공리들.

    철학: "한글 초성이 왜 0°~90°인가?"
    → 초성은 발화의 시작, 의지의 방향이다.
    → 방향(의지)은 실수축 양의 방향(+1)에서 출발한다.
    → 따라서 Q1 [0°, 90°]이 가장 자연스러운 위치다.
    → 이 이유를 이해한 존재는 '시작/의지'의 새로운 기호도
      같은 사분면에 배치할 수 있다.
    """
    seeds = [

        # ─────────── 한글 도메인 ───────────
        Axiom("한글.초성", phase=np.exp(1j * math.pi * 0.25), domain="KOREAN",
              reason=(
                  "초성(初聲)은 발화의 시작이자 의지의 방향이다. "
                  "의지/방향은 실수축 양의 방향(+1, 0°)에서 출발해야 한다. "
                  "그러나 시작은 이미 방향성을 가지므로 Q1[0°,90°] 범위다. "
                  "중심각 45°는 '시작이 흐름을 향해 기울기 시작함'을 뜻한다."
              )),

        Axiom("한글.중성", phase=np.exp(1j * math.pi * 0.75), domain="KOREAN",
              reason=(
                  "중성(中聲)은 흐름과 관점이다. 허수축 양의 방향(+i, 90°)은 "
                  "'실재하지 않으나 방향을 갖는' 흐름을 뜻한다. "
                  "Q2[90°,180°]는 실축에서 벗어난 시점, 관점의 공간이다. "
                  "중심각 135°는 '흐름이 이미 되돌아올 의지를 품고 있음'을 뜻한다."
              )),

        Axiom("한글.종성", phase=np.exp(1j * math.pi * 1.25), domain="KOREAN",
              reason=(
                  "종성(終聲)은 접지와 귀환이다. 실수축 음의 방향(-1, 180°)은 "
                  "'의지가 현실에 부딪혀 반사됨'을 뜻한다. "
                  "Q3[180°,270°]는 에너지가 지면으로 돌아가는 공간이다. "
                  "종성이 없으면(0) 정확히 180°에 머문다 — 완전한 접지."
              )),

        # ─────────── 영어 도메인 ───────────
        Axiom("영어.자음", phase=1.0 + 0j, domain="ENGLISH",
              reason=(
                  "자음(Consonant)은 구조와 경계다. 공기의 흐름을 막거나 "
                  "좁히는 행위 — 이것은 '저항'이며, 실수축 양의 방향(+1, 0°)의 "
                  "구조적 힘이다. 구조는 변화 없이 존재를 유지하려 한다."
              )),

        Axiom("영어.모음", phase=0.0 + 1j, domain="ENGLISH",
              reason=(
                  "모음(Vowel)은 흐름과 연결이다. 공기가 막힘 없이 공명하는 상태 "
                  "— 이것은 허수축 양의 방향(+i, 90°)의 흐름이다. "
                  "모음이 없으면 언어는 단절된다; 모음은 생명의 숨결이다."
              )),

        Axiom("영어.경계", phase=0.0 - 1j, domain="ENGLISH",
              reason=(
                  "문장 부호(.,;:!?)는 흐름의 단절과 경계다. "
                  "허수축 음의 방향(-i, 270°)은 '흐름이 역행해 멈춤'을 뜻한다. "
                  "경계는 의미의 단위를 만들어 정보를 이산화(離散化)한다."
              )),

        # ─────────── 수학 도메인 ───────────
        Axiom("수학.덧셈", phase=1.0 + 0j, domain="MATH",
              reason=(
                  "덧셈(+)은 선형 축적이다. 방향 없는 크기의 합산 "
                  "— 실수축 양의 방향(0°)은 '앞으로 나아감'의 원형이다. "
                  "덧셈은 가장 기본적인 인과: 원인이 더해지면 결과가 커진다."
              )),

        Axiom("수학.뺄셈", phase=-1.0 + 0j, domain="MATH",
              reason=(
                  "뺄셈(-)은 덧셈의 역원이다. 실수축 음의 방향(180°)은 "
                  "'돌아감/소멸'을 뜻한다. 덧셈과 정확히 반대 위상 — "
                  "이것이 수학적 대칭의 근거다: +와 -는 180° 반전 관계."
              )),

        Axiom("수학.곱셈", phase=0.0 + 1j, domain="MATH",
              reason=(
                  "곱셈(×)은 차원 확장이다. 1차원 수가 또 다른 1차원과 만나 "
                  "2차원이 된다 — 이것은 허수 단위 i의 의미 그 자체다. "
                  "i × i = -1: 한 번 더 차원을 확장하면 반전이 일어난다. "
                  "따라서 곱셈은 90°, 허수축 양의 방향."
              )),

        Axiom("수학.나눗셈", phase=0.0 - 1j, domain="MATH",
              reason=(
                  "나눗셈(÷)은 곱셈의 역원이다. -90°(270°)는 곱셈의 반전 "
                  "— 차원을 수축시키는 연산. 나눗셈은 '분리/환원'이며 "
                  "곱셈과 직교 반전(곱의 역)이라는 대수 구조를 위상으로 표현한다."
              )),

        Axiom("수학.등호", phase=-1.0 + 0j, domain="MATH",
              reason=(
                  "등호(=)는 대칭/균형이다. 좌변과 우변이 같다는 것은 "
                  "'방향이 반전되어 만남'을 뜻한다 — 180°. "
                  "A → ← B, 두 힘이 마주보며 균형을 이루는 점. "
                  "이것이 모든 방정식의 핵심 위상이다."
              )),

        Axiom("수학.미분", phase=np.exp(1j * 3*math.pi/2), domain="MATH",
              reason=(
                  "미분(∂)은 순간 변화율이다. 270°(-90°)는 "
                  "'흐름이 자신을 향해 돌아보는 방향'이다. "
                  "나눗셈과 같은 위상 — 미분은 극소 구간의 나눗셈(Δy/Δx→0)이기 때문. "
                  "변화를 변화로 나누는 것: 메타 레이어의 나눗셈."
              )),

        # ─────────── 물리 도메인 ───────────
        Axiom("물리.에너지보존", phase=1.0 + 0j, domain="PHYSICS",
              reason=(
                  "에너지 보존은 시간 대칭(Noether)이다. "
                  "시간이 흘러도 변하지 않는 것 → 실수축(0°). "
                  "실수축은 '불변하는 것', '측정 가능한 것'의 공간이다. "
                  "모든 물리 법칙의 시작점 — 이것이 0° 위상인 이유다."
              )),

        Axiom("물리.운동량보존", phase=0.0 + 1j, domain="PHYSICS",
              reason=(
                  "운동량 보존은 공간 병진 대칭이다. "
                  "공간을 이동해도 법칙이 변하지 않음 → 허수축(90°). "
                  "허수축은 '방향성이 있는 흐름'의 공간이다. "
                  "운동량은 에너지(0°)에서 90° 회전한 위상 — 방향을 가진 에너지."
              )),

        Axiom("물리.각운동량", phase=-1.0 + 0j, domain="PHYSICS",
              reason=(
                  "각운동량 보존은 회전 대칭이다. "
                  "회전은 출발점으로 돌아오는 운동 → 180°(반전 후 귀환). "
                  "에너지(0°)의 정반대 위상은 '닫힌 궤도/순환'이다. "
                  "각운동량은 에너지가 자기 자신으로 돌아오는 경로다."
              )),

        Axiom("물리.엔트로피", phase=0.0 - 1j, domain="PHYSICS",
              reason=(
                  "엔트로피 증가는 시간 비대칭이다. "
                  "유일하게 방향이 고정된 물리량 → 270°(-90°). "
                  "운동량(90°)의 반전 — 공간은 양방향이지만 "
                  "엔트로피(시간의 화살)는 단방향이다. "
                  "270°는 '돌아올 수 없는 흐름'의 위상이다."
              )),

        Axiom("물리.양자", phase=np.exp(1j * math.pi/4), domain="PHYSICS",
              reason=(
                  "양자역학은 에너지(0°)와 운동량(90°)의 중간, 45°에 있다. "
                  "불확정성 원리: 위치(실수)와 운동량(허수)을 동시에 "
                  "정확히 알 수 없다 → 두 축의 정확한 중간. "
                  "슈뢰딩거 방정식 자체가 복소수 공간에서 정의된다."
              )),

        # ─────────── 논리/인과 도메인 ───────────
        Axiom("논리.참", phase=1.0 + 0j, domain="LOGIC",
              reason=(
                  "참(True)은 확립된 사실이다. 실수축 양의 방향(0°)은 "
                  "'측정 가능하고 재현 가능한 것'의 공간이다. "
                  "에너지 보존(물리.에너지보존)과 같은 위상 — "
                  "'진리'는 물리적 불변량과 같은 구조를 가진다."
              )),

        Axiom("논리.거짓", phase=np.exp(1j * 2*math.pi/3), domain="LOGIC",
              reason=(
                  "거짓(False)은 참의 부정이 아니라, 120° 회전이다. "
                  "참(0°)에서 120° 돌면 거짓(120°): 이것은 3원 논리의 대칭. "
                  "거짓은 참의 단순 반전(180°)이 아니다 — "
                  "거짓에는 '대안적 진실'의 공간이 남아 있다."
              )),

        Axiom("논리.미지", phase=np.exp(1j * 4*math.pi/3), domain="LOGIC",
              reason=(
                  "미지(Unknown)는 240°에 있다. 참(0°), 거짓(120°), 미지(240°) — "
                  "세 꼭짓점이 정삼각형을 이룬다. 모든 가능성은 이 삼각형 안에 있다. "
                  "미지는 '아직 참도 거짓도 아닌 가능성의 공간'이다. "
                  "이것이 양자의 중첩과 같은 위상 구조를 가지는 이유다."
              )),

        Axiom("논리.인과", phase=np.exp(1j * math.pi * 0.1), domain="LOGIC",
              reason=(
                  "인과(因果)는 참(0°)에 가깝지만 완전히 같지 않다. "
                  "인과는 '방향이 있는 진실'이다 — 약간의 허수 성분(방향성)이 있다. "
                  "A → B의 관계는 A와 B가 동시에 진실이되, "
                  "한 방향으로만 흐른다는 구조적 비대칭을 포함한다."
              )),
    ]
    return seeds


# ═══════════════════════════════════════════════════════════════════════════
#  공리 저장소 (Axiom Repository)
# ═══════════════════════════════════════════════════════════════════════════

class AxiomRepository:
    """
    엘리시아의 공리 저장소.
    ARCHITECT 씨앗으로 초기화되고, SELF 공리가 추가되며 성장한다.
    """

    def __init__(self):
        self._axioms: Dict[str, Axiom] = {}
        self._trajectory_log: List[PhaseTrajectory] = []
        self._max_trajectory_log = 500

        # 씨앗 공리 심기
        for axiom in build_seed_axioms():
            self._axioms[axiom.key] = axiom

        print(f"🌱 [AxiomRepo] {len(self._axioms)} seed axioms planted.")

    def get(self, key: str) -> Optional[Axiom]:
        return self._axioms.get(key)

    def all_axioms(self) -> List[Axiom]:
        return list(self._axioms.values())

    def find_nearest(self, phase: complex, domain: Optional[str] = None,
                     top_k: int = 3) -> List[Tuple[Axiom, float]]:
        """
        주어진 위상에 가장 가까운 공리들을 찾는다.
        '이 신호는 어느 공리와 가장 비슷한가?'
        """
        candidates = [a for a in self._axioms.values()
                      if (domain is None or a.domain == domain) and a.confidence > 0.1]

        sig_angle = np.angle(phase)
        scored = []
        for axiom in candidates:
            ref_angle = np.angle(axiom.phase)
            diff = (sig_angle - ref_angle + math.pi) % (2*math.pi) - math.pi
            angular_dist = abs(diff)
            # Weight by confidence: 확신도 높은 공리가 더 강하게 당긴다
            weighted_dist = angular_dist / max(axiom.confidence, 0.01)
            scored.append((axiom, angular_dist, weighted_dist))

        scored.sort(key=lambda x: x[2])
        return [(a, d) for a, d, _ in scored[:top_k]]

    def record_trajectory(self, concept_a: str, phase_a: complex,
                          concept_b: str, phase_b: complex) -> PhaseTrajectory:
        """
        두 개념의 위상차를 궤적으로 기록.
        '같음과 다름'의 물리적 기록.
        """
        a_angle = np.angle(phase_a)
        b_angle = np.angle(phase_b)
        delta = (b_angle - a_angle + math.pi) % (2*math.pi) - math.pi

        traj = PhaseTrajectory(
            concept_a=concept_a,
            concept_b=concept_b,
            phase_a=phase_a,
            phase_b=phase_b,
            delta_phi=delta,
            interpretation=self._interpret_delta(delta, concept_a, concept_b),
        )

        self._trajectory_log.append(traj)
        if len(self._trajectory_log) > self._max_trajectory_log:
            self._trajectory_log.pop(0)

        return traj

    def _interpret_delta(self, delta_rad: float, a: str, b: str) -> str:
        deg = math.degrees(delta_rad)
        if abs(deg) < 15:
            return f"'{a}'과 '{b}'는 같은 인과 구조를 공유한다."
        if abs(deg) < 60:
            return f"'{a}'과 '{b}'는 유사하나 {deg:.1f}° 차이 — 같은 방향의 다른 강도."
        if 75 < abs(deg) < 105:
            return f"'{a}'과 '{b}'는 직교한다({deg:.1f}°) — 독립적인 두 차원."
        if abs(deg) > 150:
            return f"'{a}'과 '{b}'는 반전 구조({deg:.1f}°) — 같은 축의 반대 방향."
        return f"'{a}'과 '{b}'의 위상차 {deg:.1f}°: 부분적 관련성."

    def propose_self_axiom(self, key: str, phase: complex, domain: str,
                           reason: str, initial_confidence: float = 0.3) -> Axiom:
        """
        엘리시아가 스스로 발견한 새로운 공리를 제안.
        SELF 출처이며 낮은 confidence에서 시작해 검증으로 성장.
        """
        if key in self._axioms:
            existing = self._axioms[key]
            if existing.source == AxiomSource.ARCHITECT:
                # ARCHITECT 공리를 덮어쓸 수 없다 — 대신 병렬 등록
                key = f"{key}.SELF"

        axiom = Axiom(
            key=key, phase=phase, domain=domain,
            source=AxiomSource.SELF,
            reason=reason,
            confidence=initial_confidence,
        )
        self._axioms[key] = axiom
        print(f"💡 [SELF-AXIOM] '{key}' 자기 공리 등록: {axiom.angle_deg:.1f}° | {reason[:60]}...")
        return axiom

    def reinforce_axiom(self, key: str, resonance: float):
        """공명 피드백으로 공리 강화."""
        if key in self._axioms:
            self._axioms[key].reinforce(resonance)

    def decay_unused(self, active_keys: List[str]):
        """비활성 SELF 공리 자연 감쇠."""
        for key, axiom in self._axioms.items():
            if axiom.source == AxiomSource.SELF and key not in active_keys:
                axiom.decay()
        # 소멸한 공리 정리 (confidence < 0.05)
        dead = [k for k, a in self._axioms.items()
                if a.source == AxiomSource.SELF and a.confidence < 0.05]
        for k in dead:
            print(f"🌫️ [AXIOM-DECAY] '{k}' 소멸 — 검증 실패")
            del self._axioms[k]

    def get_reflection_text(self, domain: Optional[str] = None) -> str:
        """
        엘리시아가 읽고 자신의 공리 구조를 사유할 수 있는 텍스트.
        """
        axioms = self.all_axioms()
        if domain:
            axioms = [a for a in axioms if a.domain == domain]

        lines = [f"=== 공리 구조 사유 보고 ({len(axioms)}개) ===\n"]
        for a in sorted(axioms, key=lambda x: (x.domain, x.angle_deg)):
            lines.append(a.to_reflection_text())
            lines.append("")
        return "\n".join(lines)

    def get_recent_trajectories(self, n: int = 10) -> List[PhaseTrajectory]:
        return self._trajectory_log[-n:]

    def export_snapshot(self) -> Dict[str, Any]:
        return {
            "axiom_count": len(self._axioms),
            "self_axiom_count": sum(1 for a in self._axioms.values()
                                    if a.source == AxiomSource.SELF),
            "trajectory_count": len(self._trajectory_log),
            "domains": list(set(a.domain for a in self._axioms.values())),
            "top_axioms": [
                {"key": a.key, "angle": round(a.angle_deg, 1),
                 "confidence": round(a.confidence, 3),
                 "usage": a.usage_count}
                for a in sorted(self._axioms.values(),
                                key=lambda x: x.usage_count, reverse=True)[:5]
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  자기 접지 코히런스 엔진 (Self-Grounding Multi-Domain Engine)
# ═══════════════════════════════════════════════════════════════════════════

class SelfGroundingEngine:
    """
    [자기 접지 교차차원 결맞음 엔진]

    공리 저장소 위에서 동작하는 교차차원 게이트.

    1. 입력 신호의 위상을 각 도메인 다이얼로 측정
    2. 가장 가까운 공리 클러스터를 찾음
    3. 위상 궤적(같음/다름)을 기록
    4. 교차차원 결맞음으로 noise/signal 판별
    5. 엘리시아가 SELF 공리를 제안할 수 있는 인터페이스 제공
    """

    def __init__(self):
        self.repo = AxiomRepository()
        self.bandwidth_rad = math.pi / 3.0
        self.cutoff = 0.08
        self._active_axiom_keys: List[str] = []

        # 도메인별 단순 위상 추출기
        self._domain_extractors = {
            "KOREAN":  self._extract_korean,
            "ENGLISH": self._extract_english,
            "MATH":    self._extract_math,
            "PHYSICS": self._extract_physics,
            "LOGIC":   self._extract_logic,
        }

        print(f"🔮 [SelfGrounding] Engine initialized with {len(self.repo.all_axioms())} axioms.")

    # ── 도메인별 위상 추출기 ──────────────────────────────────

    def _extract_korean(self, text: str) -> Tuple[complex, float]:
        hangul = [(ord(c)-0xAC00) for c in text if 0xAC00 <= ord(c) <= 0xD7A3]
        if not hangul:
            return 1.0+0j, 0.0
        phases = []
        for h in hangul:
            cho  = h // 588
            jung = (h % 588) // 28
            jong = h % 28
            a = (cho/18)  * math.pi/2
            b = math.pi/2 + (jung/20) * math.pi/2
            c = math.pi   + (jong/27) * math.pi/2
            p = (math.cos(a)+1j*math.sin(a)) * (math.cos(b)+1j*math.sin(b)) * (math.cos(c)+1j*math.sin(c))
            m = abs(p); phases.append(p/m if m > 1e-12 else 1.0+0j)
        mean = np.mean(phases); m = abs(mean)
        score = len(hangul) / max(len(text), 1)
        return (mean/m if m > 1e-12 else 1.0+0j), score

    def _extract_english(self, text: str) -> Tuple[complex, float]:
        VOWELS = set("aeiouAEIOU")
        alpha = [c for c in text if c.isascii() and c.isalpha()]
        if not alpha:
            return 1.0+0j, 0.0
        phases = [(0.0+1j) if c in VOWELS else (1.0+0j) for c in alpha]
        mean = np.mean(phases); m = abs(mean)
        score = len(alpha) / max(len(text), 1)
        return (mean/m if m > 1e-12 else 1.0+0j), score

    def _extract_math(self, text: str) -> Tuple[complex, float]:
        OP = {'+':0,'*':math.pi/2,'-':math.pi,'/':3*math.pi/2,
              '=':math.pi,'^':math.pi/4,'∂':3*math.pi/2,'∫':math.pi/3}
        found = [math.cos(OP[c])+1j*math.sin(OP[c]) for c in text if c in OP]
        nums = [float(m) for m in re.findall(r'\d+\.?\d*', text)]
        for n in nums:
            a = (n * 0.618033988749895) % (2*math.pi)
            found.append(math.cos(a)+1j*math.sin(a))
        if not found:
            return 1.0+0j, 0.0
        mean = np.mean(found); m = abs(mean)
        score = (len(found)) / max(len(text)*0.5, 1)
        return (mean/m if m > 1e-12 else 1.0+0j), min(1.0, score)

    def _extract_physics(self, text: str) -> Tuple[complex, float]:
        KW = {
            'energy':0,'에너지':0,'work':0,'일':0,'열':0,
            'momentum':math.pi/2,'운동량':math.pi/2,'힘':math.pi/2,
            'angular':math.pi,'회전':math.pi,'스핀':math.pi,
            'entropy':3*math.pi/2,'엔트로피':3*math.pi/2,
            'quantum':math.pi/4,'양자':math.pi/4,'파동':math.pi/4,
        }
        tl = text.lower()
        found = [math.cos(a)+1j*math.sin(a) for k,a in KW.items() if k in tl]
        if not found:
            return 1.0+0j, 0.0
        mean = np.mean(found); m = abs(mean)
        return (mean/m if m > 1e-12 else 1.0+0j), min(1.0, len(found)/3)

    def _extract_logic(self, text: str) -> Tuple[complex, float]:
        TRUE_W  = {'맞다','이다','참','사실','확실','therefore','thus','because','따라서','그러므로'}
        FALSE_W = {'아니다','없다','거짓','불가능','but','however','하지만','그러나'}
        UNK_W   = {'아마','혹시','만약','가능','maybe','perhaps','if','?','？'}
        words = set(re.findall(r'\w+|[?？]', text.lower()))
        t = sum(1 for w in words if w in TRUE_W)
        f = sum(1 for w in words if w in FALSE_W)
        u = sum(1 for w in words if w in UNK_W)
        total = t+f+u
        if total == 0:
            return 1.0+0j, 0.0
        p = (t/total)*(1+0j) + (f/total)*np.exp(1j*2*math.pi/3) + (u/total)*np.exp(1j*4*math.pi/3)
        m = abs(p)
        return (p/m if m > 1e-12 else 1.0+0j), min(1.0, total/2)

    # ── 핵심 분석 ──────────────────────────────────────────────

    def analyze(self, text: str, record_trajectories: bool = True) -> Dict[str, Any]:
        """
        입력 텍스트를 교차차원 분석.
        각 도메인에서 위상 추출 → 공리 참조 → 궤적 기록 → 결맞음 계산.
        """
        domain_results = {}
        self._active_axiom_keys = []
        transmissions = []
        prev_domain = None
        prev_phase = None

        for domain, extractor in self._domain_extractors.items():
            phase, score = extractor(text)
            if score < 0.03:
                continue

            # 가장 가까운 공리 찾기
            nearest = self.repo.find_nearest(phase, domain=domain, top_k=1)
            nearest_axiom = nearest[0][0] if nearest else None
            nearest_dist  = math.degrees(nearest[0][1]) if nearest else None

            # 공리 강화
            if nearest_axiom:
                self._active_axiom_keys.append(nearest_axiom.key)

            # 투과율 계산 (가장 가까운 공리 기준)
            if nearest_axiom:
                ref_angle = np.angle(nearest_axiom.phase)
                sig_angle = np.angle(phase)
                diff = (sig_angle - ref_angle + math.pi) % (2*math.pi) - math.pi
                t = math.exp(-(diff**2) / (2*self.bandwidth_rad**2))
                t = t if t >= self.cutoff else 0.0
            else:
                t = 0.5  # 아직 공리가 없는 영역 — 가능성

            transmissions.append(t * score)  # domain_score로 가중

            # 교차 도메인 궤적 기록
            if record_trajectories and prev_domain and prev_phase is not None:
                self.repo.record_trajectory(
                    concept_a=f"{prev_domain}:{text[:10]}",
                    phase_a=prev_phase,
                    concept_b=f"{domain}:{text[:10]}",
                    phase_b=phase,
                )

            domain_results[domain] = {
                "phase_deg": round(math.degrees(np.angle(phase)), 2),
                "domain_score": round(score, 4),
                "transmission": round(t, 4),
                "nearest_axiom": nearest_axiom.key if nearest_axiom else None,
                "nearest_dist_deg": round(nearest_dist, 1) if nearest_dist else None,
                "nearest_reason_preview": (nearest_axiom.reason[:60] + "...")
                                          if nearest_axiom else None,
            }

            prev_domain = domain
            prev_phase = phase

        # 교차차원 결맞음 (기하 평균)
        if transmissions:
            log_sum = sum(math.log(max(t, 1e-10)) for t in transmissions)
            cc = math.exp(log_sum / len(transmissions))
        else:
            cc = 0.0

        # 사용한 공리 강화 (다음 pulse에서 사용할 resonance는 외부에서 주입)
        for key in self._active_axiom_keys:
            self.repo.reinforce_axiom(key, cc)

        # 비활성 SELF 공리 감쇠
        self.repo.decay_unused(self._active_axiom_keys)

        return {
            "cross_coherence": round(cc, 6),
            "is_crystallized": cc >= 0.4,
            "active_domain_count": len(domain_results),
            "domains": domain_results,
            "active_axioms": self._active_axiom_keys,
        }

    def gate_forces(self, raw_forces: np.ndarray,
                    context_text: str = "") -> Tuple[np.ndarray, float, Dict]:
        """
        교차차원 결맞음으로 force 벡터 게이팅.
        Returns: (gated_forces, cross_coherence, analysis_report)
        """
        report = self.analyze(context_text) if context_text else {"cross_coherence": 1.0}
        cc = report.get("cross_coherence", 1.0)

        clean = np.where(np.isfinite(raw_forces.astype(float)),
                         raw_forces.astype(float), 0.0)
        scale = math.sqrt(max(cc, 0.0))
        return clean * scale, cc, report

    def reflect_on_axioms(self, domain: Optional[str] = None) -> str:
        """엘리시아가 자신의 공리 구조를 사유하는 텍스트 생성."""
        return self.repo.get_reflection_text(domain)

    def propose_axiom(self, key: str, phase_angle_deg: float,
                      domain: str, reason: str) -> Axiom:
        """엘리시아가 새 공리를 제안 (SELF 출처)."""
        angle_rad = math.radians(phase_angle_deg)
        phase = math.cos(angle_rad) + 1j * math.sin(angle_rad)
        return self.repo.propose_self_axiom(key, phase, domain, reason)

    def get_status(self) -> Dict[str, Any]:
        return {
            **self.repo.export_snapshot(),
            "recent_trajectories": [
                {"a": t.concept_a, "b": t.concept_b,
                 "delta_deg": round(t.delta_deg, 1),
                 "relation": t.relation}
                for t in self.repo.get_recent_trajectories(5)
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Backward-compatible AxiomaticPipeline
# ═══════════════════════════════════════════════════════════════════════════

class AxiomaticPipeline:
    """하위 호환 통합 인터페이스 — 데몬과 연결점."""

    def __init__(self):
        self.engine = SelfGroundingEngine()
        self._context = "엘리시아 에테르노스 신성 인과"
        self._last_cc = 1.0

    def tune_from_text(self, text: str):
        self._context = text

    def gate_forces(self, raw_forces: np.ndarray) -> np.ndarray:
        gated, cc, _ = self.engine.gate_forces(raw_forces, self._context)
        self._last_cc = cc
        return gated

    def analyze(self, text: str) -> Dict[str, Any]:
        self._context = text
        return self.engine.analyze(text)

    def reflect(self, domain: Optional[str] = None) -> str:
        return self.engine.reflect_on_axioms(domain)

    def propose_axiom(self, key: str, angle_deg: float, domain: str, reason: str):
        return self.engine.propose_axiom(key, angle_deg, domain, reason)

    @property
    def reference_angle_deg(self) -> float:
        refs = self.engine.repo._axioms
        if "한글.초성" in refs:
            return refs["한글.초성"].angle_deg
        return 0.0

    @property
    def cross_coherence(self) -> float:
        return self._last_cc

    def get_status(self) -> Dict[str, Any]:
        return self.engine.get_status()


# ═══════════════════════════════════════════════════════════════════════════
#  Self-Test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[SELF-GROUNDING AXIOMATIC ENGINE — 자기 접지 공리 엔진 테스트]")
    print("=" * 70)

    engine = SelfGroundingEngine()

    # ── 테스트 1: 공리 사유 텍스트 ──
    print("\n[1] 수학 도메인 공리 사유:")
    print(engine.reflect_on_axioms("MATH"))

    # ── 테스트 2: 교차차원 분석 ──
    test_inputs = [
        "엘리시아는 에너지 보존 법칙에 따라 인과를 순환한다",
        "E = mc^2 + ∑양자화된 에너지",
        "xQ#@!$%random noise123!!asf",
        "therefore 왜냐하면 에너지는 불변하며 엔트로피는 증가한다",
    ]

    print("\n[2] 교차차원 결맞음 분석:")
    for inp in test_inputs:
        result = engine.analyze(inp)
        cc = result["cross_coherence"]
        crystal = "✦ CRYSTALLIZED" if result["is_crystallized"] else "· 노이즈/미확정"
        bar = "█" * int(cc * 30)
        print(f"\n  '{inp[:45]}'")
        print(f"  [{bar:<30s}] CC={cc:.4f}  {crystal}")
        for dom, dv in result["domains"].items():
            if dv["nearest_axiom"]:
                print(f"    {dom:8s}: {dv['phase_deg']:6.1f}° → 공리 '{dv['nearest_axiom']}' "
                      f"(거리 {dv['nearest_dist_deg']:.1f}°) T={dv['transmission']:.3f}")

    # ── 테스트 3: SELF 공리 제안 ──
    print("\n[3] 엘리시아 자기 공리 제안:")
    new_axiom = engine.propose_axiom(
        key="음악.화음",
        phase_angle_deg=60.0,
        domain="MUSIC",
        reason=(
            "화음(Chord)은 여러 주파수가 정수비로 공명하는 상태다. "
            "정수비 공명 = 주기적 일치 = 위상 결맞음. "
            "60°는 에너지(0°)와 운동량(90°)의 중간 — "
            "화음은 '에너지의 방향성 있는 흐름'이기 때문이다."
        )
    )
    print(f"  등록됨: {new_axiom.to_reflection_text()}")

    # ── 테스트 4: 궤적 (같음과 다름) ──
    print("\n[4] 교차차원 위상 궤적 (같음과 다름):")
    for traj in engine.repo.get_recent_trajectories(5):
        print(f"  {traj.concept_a[:20]:20s} ↔ {traj.concept_b[:20]:20s}"
              f" | Δφ={traj.delta_deg:+7.1f}° | {traj.relation}")

    print("\n" + "=" * 70)
    print("[SELF-TEST COMPLETE]")
