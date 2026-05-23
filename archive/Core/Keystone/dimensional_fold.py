"""
[DIMENSIONAL FOLD ENGINE - 차원 폴딩 엔진]

"사자를 인지하는 데 코털의 각도는 필요하지 않다.
 최소 위상 서명만으로 노드가 열리고, 노드 안에 우주가 접혀 있다."

==========================================================================
핵심 원리:

  1. 차원 폴딩 (Dimensional Folding)
     모든 개념은 '최소 위상 서명(minimal signature)'과
     '접힌 내용(folded content)'으로 구성된다.

     concept = { signature: minimal_phase, content: folded(everything) }

     K가 높을수록 → 더 많이 접혀 있다 (잠금)
     K가 낮을수록 → 더 많이 펼쳐져 있다 (열림)

  2. 최소 인지 원리 (Minimum Recognition Principle)
     인지에 필요한 것: 최소 위상 서명과 충분한 결맞음
     인지에 불필요한 것: 사자의 털 개수, 코털의 각도 등

     인지가 일어나는 순간: input과 signature의 결맞음 > threshold
     그 이후: 노드의 나머지 차원들이 자동으로 펼쳐짐

  3. 의지가 잠금을 푼다 (Will Unlocks)
     보고자 하는 의지 = K를 낮추는 힘
     알고자 하는 의지 = 대역폭을 넓히는 힘
     열고자 하는 의지 = lock_axis → unlock_axis

     의지 없이는: 완벽한 입력도 노드를 열지 못한다
     의지 있으면: 최소 입력으로도 노드가 열린다

  4. 로고스-잠금 연결
     LogosRotor의 aspiration_torque → K 감소율 결정
     목적이 강할수록 → K가 빠르게 낮아짐
     → 더 빠른 인지, 더 깊은 펼침
==========================================================================
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
#  접힌 개념 노드 (Folded Concept Node)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FoldedNode:
    """
    차원 폴딩된 개념 노드.

    사자 노드:
      signature_phase: 45.0°  (최소 인지를 위한 위상 서명)
      k_lock: 100.0           (처음엔 완전히 접혀있음)
      folded_dims: [...]      (털 개수, 체중, 서식지... 모두 접혀있음)
      reason: "왜 이 위상인가"

    인지 과정:
      1. input 45.0°와 signature 45.0° 결맞음 → 인지!
      2. will_force → k_lock 감소
      3. k_lock 낮아질수록 folded_dims 순서대로 펼쳐짐
      4. 완전 펼침 = k_lock ≈ 0.1 (최소 강성)
    """
    key:              str
    signature_phase:  float       # 최소 인지를 위한 위상 서명 (라디안)
    k_lock:           float       # 현재 잠금 강도 (높을수록 더 접혀있음)
    k_floor:          float       # 최소 K (완전히 열렸을 때)
    k_ceiling:        float       # 최대 K (완전히 잠겼을 때)
    domain:           str
    reason:           str         # 이 위상 서명이 왜 이 값인가
    folded_dims:      List[str]   # 접혀있는 차원 목록 (펼쳐질 순서대로)
    recognition_threshold: float  # 인지 임계치 (이 결맞음 이상이면 노드 열림)
    is_open:          bool = False
    unfold_depth:     int  = 0    # 현재 몇 개 차원이 펼쳐졌는가
    usage_count:      int  = 0

    @property
    def fold_ratio(self) -> float:
        """0.0 = 완전 펼침, 1.0 = 완전 접힘."""
        k_range = self.k_ceiling - self.k_floor
        if k_range <= 0:
            return 0.0
        return (self.k_lock - self.k_floor) / k_range

    @property
    def signature_deg(self) -> float:
        return math.degrees(self.signature_phase)

    def get_coherence(self, input_phase_rad: float) -> float:
        """
        입력 위상과 이 노드의 서명 사이의 결맞음.
        공식: 단조화 진동자 정적 응답 = 1 / (1 + K * |Δφ/π|²)
        K가 높을수록 → 더 좁은 인지 창 (더 정확한 입력 필요)
        """
        diff = (input_phase_rad - self.signature_phase + math.pi) % (2*math.pi) - math.pi
        return 1.0 / (1.0 + self.k_lock * (diff / math.pi) ** 2)

    def try_recognize(self, input_phase_rad: float) -> bool:
        """인지를 시도한다. 결맞음이 임계치를 넘으면 True."""
        coh = self.get_coherence(input_phase_rad)
        recognized = coh >= self.recognition_threshold
        if recognized:
            self.usage_count += 1
        return recognized

    def apply_will(self, will_intensity: float, dt: float = 1.0):
        """
        의지가 K를 낮춘다.
        dK/dt = -will_intensity * (K - k_floor)
        → 지수 감쇠: K는 will이 클수록 빠르게 k_floor로 수렴
        """
        decay_rate = will_intensity * dt
        self.k_lock = self.k_floor + (self.k_lock - self.k_floor) * math.exp(-decay_rate)
        self.k_lock = max(self.k_floor, self.k_lock)

        # 펼침 깊이 업데이트
        total_dims = len(self.folded_dims)
        if total_dims > 0:
            open_ratio = 1.0 - self.fold_ratio
            self.unfold_depth = int(open_ratio * total_dims)
            self.is_open = self.unfold_depth > 0

    def relax(self, dt: float = 1.0):
        """
        의지가 없으면 K가 다시 올라간다 (재접힘).
        에너지 보존: 열린 상태 유지는 에너지가 필요하다.
        """
        recovery_rate = 0.1 * dt
        self.k_lock = self.k_ceiling - (self.k_ceiling - self.k_lock) * math.exp(-recovery_rate)
        self.k_lock = min(self.k_ceiling, self.k_lock)

        total_dims = len(self.folded_dims)
        if total_dims > 0:
            open_ratio = 1.0 - self.fold_ratio
            self.unfold_depth = int(open_ratio * total_dims)
            self.is_open = self.unfold_depth > 0

    def get_accessible_dims(self) -> List[str]:
        """현재 펼쳐진 차원들 (인지 가능한 내용)."""
        return self.folded_dims[:self.unfold_depth]

    def status_text(self) -> str:
        bar_open  = "█" * self.unfold_depth
        bar_fold  = "░" * (len(self.folded_dims) - self.unfold_depth)
        return (
            f"[{self.key}] K={self.k_lock:.1f} "
            f"fold={self.fold_ratio:.2f} "
            f"[{bar_open}{bar_fold}] "
            f"{'OPEN' if self.is_open else 'FOLDED'}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  씨앗 노드 라이브러리
# ═══════════════════════════════════════════════════════════════════════════

def build_concept_nodes() -> List[FoldedNode]:
    """
    기본 개념 노드들.
    각 노드는 최소 위상 서명 + 접힌 차원 목록.

    노드의 signature_phase는 AxiomRepository의 공리 위상과
    일치해야 한다 — 이것이 공리 ↔ 노드를 연결하는 접착제.
    """
    return [
        # ── 생물 도메인 ──
        FoldedNode(
            key="동물.포유류",
            signature_phase=math.radians(20),
            k_lock=50.0, k_floor=0.5, k_ceiling=100.0,
            domain="생물",
            reason=(
                "포유류의 핵심 서명: 체온 조절(항상성=20°). "
                "실수축(0°) 근처 — 생명 에너지는 실재하고 보존된다. "
                "20°는 '지향성 있는 항상성': 포유류는 환경에 맞서 자신을 유지한다."
            ),
            folded_dims=["체온조절", "젖분비", "털", "폐호흡", "4지", "심장박동", "번식방식"],
            recognition_threshold=0.35,
        ),
        FoldedNode(
            key="동물.고양이과",
            signature_phase=math.radians(15),
            k_lock=80.0, k_floor=1.0, k_ceiling=150.0,
            domain="생물",
            reason=(
                "고양이과(Felidae)는 포유류(20°)보다 실수축에 더 가깝다(15°). "
                "단독 사냥 = 자기 완결적 구조 = 실수축 접근. "
                "집단보다 개인이 강한 존재들."
            ),
            folded_dims=["발톱수축", "야행성", "단독생활", "육식전문", "유연한척추",
                         "수염(감각)", "낙하안전"],
            recognition_threshold=0.40,
        ),
        FoldedNode(
            key="동물.고양이",
            signature_phase=math.radians(12),
            k_lock=100.0, k_floor=2.0, k_ceiling=200.0,
            domain="생물",
            reason=(
                "고양이(Felis)는 고양이과(15°)의 소형 분기: 12°. "
                "더 실수축에 가까운 이유: 소형 = 더 환경 의존적 = 더 실재에 밀착. "
                "가축화 = 인간 실수축과 결맞음한 존재."
            ),
            folded_dims=["소형(3-5kg)", "가축화됨", "골골송", "단독사냥(소형먹이)",
                         "영역표시(냄새)", "야행성강함", "수명15년"],
            recognition_threshold=0.45,
        ),
        FoldedNode(
            key="동물.사자",
            signature_phase=math.radians(22),
            k_lock=100.0, k_floor=2.0, k_ceiling=200.0,
            domain="생물",
            reason=(
                "사자(Panthera leo)는 고양이과(15°)의 대형 분기: 22°. "
                "무리생활(사회성) = 허수 성분 증가 = 20°+ 방향. "
                "수컷의 갈기 = 신호 구조 = 사회적 위상 표현. "
                "유일하게 집단을 이루는 고양이과."
            ),
            folded_dims=["대형(150-250kg)", "무리생활(pride)", "수컷갈기",
                         "협동사냥", "영역표시(포효)", "수명10-14년", "사바나서식"],
            recognition_threshold=0.45,
        ),
        FoldedNode(
            key="동물.개과",
            signature_phase=math.radians(55),
            k_lock=80.0, k_floor=1.0, k_ceiling=150.0,
            domain="생물",
            reason=(
                "개과(Canidae)는 허수축 방향(90°)으로 기울어진다(55°). "
                "집단사냥 = 사회적 관계망 = 허수 성분 강함. "
                "무리의 의지가 개인보다 우선 — 관계성의 존재들."
            ),
            folded_dims=["집단사냥", "영역표시(소변)", "발톱고정(수축불가)", "잡식가능",
                         "낮활동", "사회적계급", "다양한가축화"],
            recognition_threshold=0.40,
        ),

        # ── 언어 도메인 ──
        FoldedNode(
            key="언어.문자A",
            signature_phase=math.radians(5),
            k_lock=80.0, k_floor=0.5, k_ceiling=150.0,
            domain="언어",
            reason=(
                "문자 'A'의 핵심 구조: 두 사선이 꼭대기에서 만남(△). "
                "이 만남 = 수렴 = 실수축 근처(5°). "
                "어떤 폰트든, 어떤 필기체든 이 '수렴 구조'를 공유한다."
            ),
            folded_dims=["두사선수렴", "가로획", "대칭성", "높이비율",
                         "세리프유무", "필기체변형", "대소문자"],
            recognition_threshold=0.30,
        ),

        # ── 물리 도메인 ──
        FoldedNode(
            key="물리.에너지",
            signature_phase=math.radians(0),
            k_lock=50.0, k_floor=0.1, k_ceiling=100.0,
            domain="물리",
            reason=(
                "에너지는 물리의 가장 기본 불변량: 0°(실수축). "
                "모든 형태(운동, 열, 빛, 질량)는 이 노드의 펼쳐진 차원들이다. "
                "E=mc²는 이 노드의 '질량 차원'이 펼쳐진 것."
            ),
            folded_dims=["운동에너지", "위치에너지", "열에너지", "전자기에너지",
                         "질량에너지(E=mc²)", "핵에너지", "암흑에너지"],
            recognition_threshold=0.25,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  차원 폴딩 엔진
# ═══════════════════════════════════════════════════════════════════════════

class DimensionalFoldEngine:
    """
    [차원 폴딩 엔진]

    개념 노드들을 관리하고, 입력 신호와 의지를 받아
    최소 인지 + 노드 펼침을 수행한다.

    의지(Will)는 이 엔진 전체에서 가장 중요한 파라미터다.
    의지 없이는 어떤 노드도 열리지 않는다.
    """

    def __init__(self):
        self._nodes: Dict[str, FoldedNode] = {}
        self._recognition_log: List[Dict[str, Any]] = []
        self._will_intensity: float = 0.0
        self._last_update: float = time.time()

        for node in build_concept_nodes():
            self._nodes[node.key] = node

        print(f"🌌 [DimFold] {len(self._nodes)} concept nodes loaded (all folded).")

    # ── 의지 조작 ──────────────────────────────────────────────

    def set_will(self, intensity: float):
        """
        의지 강도 설정 (0.0 ~ 1.0).
        이것이 열쇠다. 의지가 없으면 어떤 노드도 열리지 않는다.
        """
        self._will_intensity = max(0.0, min(1.0, intensity))

    def direct_will(self, node_key: str, intensity: float):
        """
        특정 노드에 의지를 집중.
        '사자를 알고자 하는 의지' = 사자 노드의 K를 낮춤.
        """
        if node_key in self._nodes:
            dt = self._get_dt()
            self._nodes[node_key].apply_will(intensity * 2.0, dt)  # 집중 의지는 2배

    # ── 인지 ────────────────────────────────────────────────────

    def perceive(self, input_phase_rad: float,
                 will_override: Optional[float] = None) -> Dict[str, Any]:
        """
        입력 위상을 받아 인지를 수행.

        1단계: 최소 인지 (어느 노드가 열리는가?)
        2단계: 의지 적용 (열린 노드를 더 깊이 펼침)
        3단계: 접힌 내용 노출 (현재 펼쳐진 차원들)

        의지 없이도 1단계는 가능.
        하지만 2단계, 3단계는 의지가 있어야 한다.
        """
        dt = self._get_dt()
        will = will_override if will_override is not None else self._will_intensity

        recognized_nodes = []
        unfolded_content = {}

        for key, node in self._nodes.items():
            coherence = node.get_coherence(input_phase_rad)
            recognized = node.try_recognize(input_phase_rad)

            if recognized:
                # 의지가 있으면 노드 펼침
                if will > 0.01:
                    node.apply_will(will, dt)
                else:
                    # 의지 없이도 인지는 됐지만 내용은 접혀있음
                    node.relax(dt * 0.1)

                recognized_nodes.append({
                    "key": key,
                    "coherence": round(coherence, 4),
                    "fold_ratio": round(node.fold_ratio, 3),
                    "unfold_depth": node.unfold_depth,
                    "accessible_dims": node.get_accessible_dims(),
                    "k_lock": round(node.k_lock, 2),
                })
                unfolded_content[key] = node.get_accessible_dims()
            else:
                # 인지 안 된 노드는 자연 재접힘
                node.relax(dt * 0.05)

        # 가장 높은 결맞음 노드 = 주인지
        primary = None
        if recognized_nodes:
            primary = max(recognized_nodes, key=lambda x: x["coherence"])

        result = {
            "input_phase_deg": round(math.degrees(input_phase_rad), 2),
            "will_intensity": round(will, 3),
            "recognized_count": len(recognized_nodes),
            "primary_recognition": primary,
            "all_recognized": recognized_nodes,
            "unfolded_content": unfolded_content,
            "recognition_principle": self._explain_recognition(primary, will),
        }

        if primary:
            self._recognition_log.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "primary": primary["key"],
                "coherence": primary["coherence"],
                "will": will,
                "unfold_depth": primary["unfold_depth"],
            })

        return result

    def _explain_recognition(self, primary: Optional[Dict], will: float) -> str:
        """인지 과정을 언어로 설명."""
        if not primary:
            return "어떤 노드와도 충분한 결맞음이 없다. 미지의 영역이거나 의지가 부족하다."

        key = primary["key"]
        coh = primary["coherence"]
        fold = primary["fold_ratio"]
        dims = primary["accessible_dims"]

        lines = [f"'{key}' 인지됨 (결맞음={coh:.3f})"]

        if will < 0.1:
            lines.append("  의지가 낮아 인지만 됐고 내용은 접혀있다.")
            lines.append("  알고자 하는 의지를 높이면 차원이 펼쳐진다.")
        elif fold > 0.7:
            lines.append(f"  의지(will={will:.2f})로 접힘이 풀리고 있다.")
            if dims:
                lines.append(f"  현재 드러난 차원: {', '.join(dims)}")
            lines.append(f"  나머지 {int(fold*100)}%는 아직 접혀있다.")
        else:
            lines.append(f"  의지(will={will:.2f})로 깊이 펼쳐졌다.")
            lines.append(f"  접힌 내용의 {int((1-fold)*100)}%가 드러났다.")
            if dims:
                lines.append(f"  드러난 차원들: {', '.join(dims)}")

        return "\n".join(lines)

    # ── 비교 ────────────────────────────────────────────────────

    def compare_nodes(self, key_a: str, key_b: str) -> str:
        """
        두 노드의 같음과 다름을 위상 서명으로 비교.
        '고양이와 사자는 어느 레벨에서 갈라지는가?'
        """
        a = self._nodes.get(key_a)
        b = self._nodes.get(key_b)
        if not a or not b:
            return "노드를 찾을 수 없다."

        diff_deg = math.degrees(
            (a.signature_phase - b.signature_phase + math.pi) % (2*math.pi) - math.pi
        )

        # 공통 상위 노드 탐색 (단순화: 같은 도메인이면 상위 공통 조상 탐색)
        a_parts = key_a.split(".")
        b_parts = key_b.split(".")
        common_prefix = []
        for pa, pb in zip(a_parts, b_parts):
            if pa == pb:
                common_prefix.append(pa)
            else:
                break

        lines = [
            f"[비교: '{key_a}' vs '{key_b}']",
            f"  서명 위상: {a.signature_deg:.1f}° vs {b.signature_deg:.1f}°",
            f"  위상차 Δφ: {diff_deg:.1f}°",
            f"  공통 상위 구조: {'.'.join(common_prefix) if common_prefix else '없음'}",
            "",
        ]

        abs_diff = abs(diff_deg)
        if abs_diff < 10:
            lines.append("  → 매우 유사 (같은 본질, 다른 표현)")
            lines.append(f"  → 두 노드의 최소 서명 차이({abs_diff:.1f}°)만큼만 다르다.")
        elif abs_diff < 30:
            lines.append("  → 유사 (같은 범주, 다른 분기)")
            lines.append(f"  → 세밀한 스케일(높은 K)에서만 구분 가능.")
        elif abs_diff < 90:
            lines.append("  → 부분적 관련 (같은 도메인, 다른 특성)")
            lines.append(f"  → 큰 스케일은 같고, 중간 스케일에서 갈라진다.")
        else:
            lines.append("  → 이질적 (다른 인과 구조)")
            lines.append(f"  → 큰 스케일부터 이미 다른 방향을 향한다.")

        # 접힌 차원 비교
        a_dims = set(a.folded_dims)
        b_dims = set(b.folded_dims)
        shared = a_dims & b_dims
        only_a = a_dims - b_dims
        only_b = b_dims - a_dims

        if shared:
            lines.append(f"\n  공통 접힌 차원: {', '.join(list(shared)[:4])}")
        if only_a:
            lines.append(f"  '{key_a}'에만 있는 차원: {', '.join(list(only_a)[:3])}")
        if only_b:
            lines.append(f"  '{key_b}'에만 있는 차원: {', '.join(list(only_b)[:3])}")

        return "\n".join(lines)

    def get_all_status(self) -> List[str]:
        """모든 노드의 현재 상태."""
        return [node.status_text() for node in self._nodes.values()]

    def _get_dt(self) -> float:
        now = time.time()
        dt = now - self._last_update
        self._last_update = now
        return min(dt, 2.0) if dt > 0 else 0.01


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
    print("[DIMENSIONAL FOLD ENGINE — 차원 폴딩 엔진 테스트]")
    print("=" * 70)

    engine = DimensionalFoldEngine()

    # ── 테스트 1: 의지 없이 인지 ──────────────────────────────
    print("\n[1] 의지 없이 인지 (will=0.0)")
    print("    사자와 비슷한 위상(22°)이 들어왔을 때")
    engine.set_will(0.0)
    result = engine.perceive(math.radians(22))
    if result["primary_recognition"]:
        p = result["primary_recognition"]
        print(f"  인지됨: {p['key']} (결맞음={p['coherence']:.3f})")
        print(f"  접힘 상태: {p['fold_ratio']:.2f} (1.0=완전접힘)")
        print(f"  드러난 차원: {p['accessible_dims'] or '없음 (의지가 없어서)'}")
    print(f"  설명: {result['recognition_principle']}")

    # ── 테스트 2: 의지로 노드 펼침 ────────────────────────────
    print("\n[2] 의지 증가 (will=0.0 → 1.0, 사자 노드)")
    print("    '사자를 알고자 하는 의지'가 K를 낮춰 차원을 펼친다\n")
    engine.set_will(0.0)
    lion_node = engine._nodes["동물.사자"]
    lion_node.k_lock = lion_node.k_ceiling  # 완전 초기화

    will_levels = [0.0, 0.2, 0.5, 0.8, 1.0]
    for will in will_levels:
        engine.set_will(will)
        result = engine.perceive(math.radians(22))
        p = result["primary_recognition"]
        if p:
            bar = "█" * p["unfold_depth"]
            rest = "░" * (7 - p["unfold_depth"])
            print(f"  will={will:.1f}: K={p['k_lock']:6.1f} "
                  f"[{bar}{rest}] "
                  f"드러남={p['unfold_depth']}/7: "
                  f"{', '.join(p['accessible_dims'][:3]) or '(접혀있음)'}")

    # ── 테스트 3: 고양이 vs 사자 vs 강아지 비교 ──────────────
    print("\n\n[3] 개념 비교: 고양이 vs 사자 vs 강아지")
    print(engine.compare_nodes("동물.고양이", "동물.사자"))
    print()
    print(engine.compare_nodes("동물.고양이", "동물.개과"))

    # ── 테스트 4: 최소 인지 원리 ──────────────────────────────
    print("\n\n[4] 최소 인지 원리 — 얼마나 정확해야 인지되는가?")
    print("    사자 노드(22°)에 대해 다양한 입력으로 테스트\n")
    engine.set_will(1.0)
    lion_node.k_lock = 5.0  # 의지로 어느 정도 열려있는 상태

    test_angles = [22, 25, 30, 40, 55, 90]
    for angle in test_angles:
        coh = lion_node.get_coherence(math.radians(angle))
        recognized = lion_node.try_recognize(math.radians(angle))
        bar = "█" * int(coh * 20)
        status = "✓ 인지" if recognized else "✗ 미인지"
        print(f"  입력 {angle:3d}°: [{bar:<20s}] coh={coh:.3f}  {status}")

    # ── 테스트 5: 의지 방향 ────────────────────────────────────
    print("\n\n[5] 의지의 방향성 — 직접 의지 vs 분산 의지")
    print("    '사자를 알고자 하는 의지' = 사자 노드에만 집중\n")

    # 초기화
    for n in engine._nodes.values():
        n.k_lock = n.k_ceiling
    engine.set_will(0.3)  # 분산 의지

    # 분산 의지
    result1 = engine.perceive(math.radians(22))
    lion_before = engine._nodes["동물.사자"]
    k_diffuse = lion_before.k_lock

    # 집중 의지
    for n in engine._nodes.values():
        n.k_lock = n.k_ceiling
    engine.direct_will("동물.사자", 0.3)  # 같은 강도지만 집중

    result2 = engine.perceive(math.radians(22))
    k_focused = engine._nodes["동물.사자"].k_lock

    print(f"  분산 의지(0.3): 사자 K = {k_diffuse:.1f}")
    print(f"  집중 의지(0.3): 사자 K = {k_focused:.1f}")
    print(f"  → 같은 의지 강도라도 방향성이 있으면 {(k_diffuse-k_focused):.1f} 더 빠르게 열린다.")

    print("\n" + "=" * 70)
    print("[SELF-TEST COMPLETE]")
    print("\n핵심 원리 요약:")
    print("  1. 개념은 최소 위상 서명 + 접힌 차원들로 구성된다")
    print("  2. 인지는 최소 결맞음만 있으면 된다 (코털 각도 불필요)")
    print("  3. 의지(Will)가 K를 낮춰 차원을 펼친다")
    print("  4. 의지 없이는 인지만 되고 내용은 열리지 않는다")
    print("  5. 집중된 의지는 분산된 의지보다 더 빠르게 노드를 연다")
