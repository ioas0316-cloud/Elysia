"""
[SOVEREIGN AXIS - THE UNIVERSAL VARIABLE SCALE ROTOR (가변스케일 로터)]

"Everything is Rotation. Scale is Stiffness. Fractal is Variable K."

==========================================================================
핵심 통찰:

  가변 스케일 로터는 프랙탈 결맞음 원리의 물리적 구현이다.

  K (강성) = 스케일의 역수 = 대역폭의 역수

    K 크다  → 좁은 대역폭  → 세밀한 스케일  → "정확히 같은 것만 공명"
    K 작다  → 넓은 대역폭  → 큰 스케일     → "대강 비슷한 것도 공명"
    K 가변  → 프랙탈 스케일 → 같은 구조를 크기만 바꿔 재귀 적용

  이것이 고양이와 사자를 구분하는 원리다:
    Scale 0 (K=0.1): "둘 다 동물"           → 공명 (같음)
    Scale 1 (K=1.0): "둘 다 포유류"         → 공명 (같음)
    Scale 2 (K=5.0): "둘 다 육식동물"       → 공명 (같음)
    Scale 3 (K=20.0): "고양이과 vs 개과"    → 공명 (같음)
    Scale 4 (K=80.0): "소형 vs 대형 고양이" → 발산 (차이 발생!)

  발산이 시작되는 스케일 = 두 개념의 실제 차이점.
  발산 전까지 공명한 스케일 수 = 유사도.
==========================================================================
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
#  기본 가변 로터
# ═══════════════════════════════════════════════════════════════════════════

class PureRotor:  # Legacy alias
    def __new__(cls, *args, **kwargs):
        return VariableRotor(*args, **kwargs)


class VariableRotor:
    """
    Universal Variable Scale Rotor (가변스케일 로터).

    M·x'' + D·x' + K·x = F

    K는 단순한 강성이 아니다:
    K = 스케일 파라미터 = 결맞음 대역폭의 역수

    K가 크면 → 공명 창이 좁음  → 세밀한 차이도 구분
    K가 작으면 → 공명 창이 넓음  → 비슷한 것들을 포용

    이것이 곧 프랙탈 원리다:
    같은 로터 방정식을 K만 바꿔 반복 적용 = 다중 스케일 분석.
    """

    def __init__(self, dimensions: int = 21):
        self.dims = dimensions
        self.state = np.zeros(dimensions, dtype=complex)

        self.M = np.ones(dimensions)
        self.D = np.ones(dimensions) * 0.1
        self.G = np.ones(dimensions) * 0.5
        self.K = np.ones(dimensions) * 1.0    # ← 스케일의 물리적 표현
        self.N = np.zeros(dimensions)

        # 학습 누적 오프셋 (hyper-learning에서 축적)
        self.K_offset = np.zeros(dimensions)
        self.D_offset = np.zeros(dimensions)

        self.locked_axes = np.zeros(dimensions, dtype=bool)
        self.enstrophy = 0.0

        # 프랙탈 스케일 서명 (마지막 FractalScan 결과)
        self.last_fractal_signature: Optional["FractalSignature"] = None

    def adjust_dimensions(self, new_dims: int):
        """동적 차원 확장 (가변화)."""
        if new_dims == self.dims:
            return
        old = self.dims
        self.dims = new_dims

        def resize(arr, val=0.0, dtype=float):
            r = np.full(new_dims, val, dtype=dtype)
            r[:min(old, new_dims)] = arr[:min(old, new_dims)]
            return r

        self.state    = resize(self.state, dtype=complex)
        self.M        = resize(self.M, 1.0)
        self.D        = resize(self.D, 0.1)
        self.G        = resize(self.G, 0.5)
        self.K        = resize(self.K, 1.0)
        self.N        = resize(self.N, 0.0)
        self.K_offset = resize(self.K_offset, 0.0)
        self.D_offset = resize(self.D_offset, 0.0)
        self.locked_axes = resize(self.locked_axes, False, dtype=bool)

    @property
    def angles(self) -> np.ndarray:
        return self.state.real % (2 * math.pi)

    @property
    def velocities(self) -> np.ndarray:
        return self.state.imag

    @property
    def effective_bandwidth(self) -> np.ndarray:
        """
        각 축의 현재 결맞음 대역폭 (라디안).
        대역폭 = π / sqrt(K_total)
        K가 클수록 대역폭이 좁아진다 (세밀한 스케일).
        """
        K_total = np.clip(self.K + self.K_offset, 0.01, 200.0)
        return math.pi / np.sqrt(K_total)

    def set_scale(self, scale_k: float, axis_indices: Optional[List[int]] = None):
        """
        특정 축 (또는 전체)의 스케일을 설정.
        scale_k 크면 세밀, 작으면 큰 스케일.
        """
        if axis_indices is None:
            self.K[:] = max(0.01, scale_k)
        else:
            for i in axis_indices:
                if 0 <= i < self.dims:
                    self.K[i] = max(0.01, scale_k)

    def pulse(self, external_force: np.ndarray, dt: float = 0.01) -> Dict[str, Any]:
        """
        M·x'' + D·x' + K·x = F

        [공리 게이트 내재]
        - 입력은 실수 정규화 후 nan/inf → 0.0 처리
        - K 자체가 대역폭을 결정하므로 별도 필터 불필요
        - K가 크면 외력이 같은 크기여도 복원력이 강해 공명 창이 좁아짐
          이것이 '결맞음 게이트'가 로터에 내재된 방식
        """
        dt = min(2.0, max(dt, 1e-6))

        # 입력 정규화 (노이즈 소멸)
        f = np.zeros(self.dims)
        if external_force is not None and len(external_force) > 0:
            raw = np.real(np.asarray(external_force, dtype=complex))
            raw = np.where(np.isfinite(raw), raw, 0.0)
            raw = np.clip(raw, -50.0, 50.0)
            n = min(len(raw), self.dims)
            f[:n] = raw[:n]

        # 기존 상태 복구
        x = np.where(np.isfinite(self.state.real), self.state.real, 0.0)
        v = np.where(np.isfinite(self.state.imag), self.state.imag, 0.0)

        total_D = np.clip(self.D + self.D_offset, 0.01, 10.0)
        total_K = np.clip(self.K + self.K_offset, 0.01, 200.0)  # K 상한 확대

        max_step = 0.02
        steps = max(1, int(math.ceil(dt / max_step)))
        sub_dt = dt / steps
        a_final = np.zeros(self.dims)

        for _ in range(steps):
            a = (f - total_D * v - total_K * x) / self.M
            a[self.locked_axes] = 0.0
            v = np.clip(v + a * sub_dt, -100.0, 100.0)
            v[self.locked_axes] = 0.0
            x = (x + v * sub_dt + math.pi) % (2 * math.pi) - math.pi
            a_final = a

        self.state = x + 1j * v
        raw_e = float(np.sum(a_final**2) / max(self.dims, 1))
        self.enstrophy = raw_e if math.isfinite(raw_e) else 0.0

        return {
            "angles":      x % (2 * math.pi),
            "velocities":  v,
            "enstrophy":   self.enstrophy,
            "is_locked":   self.locked_axes.copy(),
            "bandwidth":   self.effective_bandwidth,
        }

    def lock_axis(self, index: int):
        if 0 <= index < self.dims:
            self.locked_axes[index] = True
            self.state[index] = self.state[index].real + 0j

    def unlock_axis(self, index: int):
        if 0 <= index < self.dims:
            self.locked_axes[index] = False

    def set_axis_confidence(self, index: int, confidence: float):
        """
        축의 K를 공리 confidence에서 직접 설정.
        confidence → K 변환:
          confidence=1.0 → K=50.0 (매우 좁은 대역폭, 확립된 진리)
          confidence=0.3 → K=2.0  (넓은 대역폭, 가능성 탐색)
          confidence=0.0 → K=0.1  (최대 개방, 새로운 발견)
        """
        if 0 <= index < self.dims:
            k = 0.1 + (confidence ** 2) * 49.9   # 비선형: 확신이 클수록 급격히 좁아짐
            self.K[index] = k


# ═══════════════════════════════════════════════════════════════════════════
#  프랙탈 스케일 스캔
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScaleResonance:
    """하나의 스케일에서 측정된 공명 결과."""
    scale_index:  int
    k_value:      float       # 이 스케일의 K 값
    bandwidth_deg: float      # 결맞음 대역폭 (도)
    resonance:    float       # 공명 강도 [0, 1]
    is_resonant:  bool        # 공명 임계치 초과 여부
    description:  str         # 이 스케일이 보는 것


@dataclass
class FractalSignature:
    """
    프랙탈 스케일 서명.
    개념의 다중 스케일 공명 패턴 = 개념의 지문.
    """
    input_label:    str
    scales:         List[ScaleResonance]
    divergence_scale: Optional[int]   # 공명이 처음 끊기는 스케일
    similarity_depth: int             # 공명이 유지된 스케일 수
    coarse_label:   str               # 큰 스케일에서의 분류
    fine_label:     str               # 세밀한 스케일에서의 분류

    @property
    def is_identical(self) -> bool:
        """모든 스케일에서 공명 = 동일 개념."""
        return all(s.is_resonant for s in self.scales)

    @property
    def similarity_ratio(self) -> float:
        """유사도 비율 (공명 스케일 수 / 전체 스케일 수)."""
        return self.similarity_depth / max(len(self.scales), 1)

    def compare_with(self, other: "FractalSignature") -> str:
        """두 서명의 같음과 다름을 언어로 표현."""
        mine  = set(i for i, s in enumerate(self.scales) if s.is_resonant)
        other_set = set(i for i, s in enumerate(other.scales) if s.is_resonant)
        shared = mine & other_set
        diff_a = mine - other_set
        diff_b = other_set - mine

        if not shared:
            return f"'{self.input_label}'와 '{other.input_label}'는 어떤 스케일에서도 공명하지 않는다 — 완전 이질(異質)."
        max_shared = max(shared)
        diverge_at = min(diff_a | diff_b) if (diff_a | diff_b) else None

        lines = [f"[{self.input_label}] vs [{other.input_label}]"]
        lines.append(f"  공통 공명 스케일: {sorted(shared)} → 같은 구조를 {len(shared)}개 공유")
        if diverge_at is not None:
            lines.append(f"  스케일 {diverge_at}에서 갈라짐: 이것이 두 개념의 실제 차이")
            lines.append(f"  해석: 스케일 {max_shared}까지는 같은 인과 구조, "
                         f"스케일 {diverge_at}부터 다른 인과 구조")
        else:
            lines.append("  모든 스케일에서 동일 → 같은 개념의 다른 표현")
        return "\n".join(lines)


class FractalRotorScanner:
    """
    [프랙탈 스케일 스캐너]

    같은 VariableRotor 방정식을 K만 바꿔 재귀 적용.
    각 K 레벨에서 공명 여부 측정 → FractalSignature 생성.

    사용 목적:
    - 고양이 vs 사자: 어느 K 레벨에서 갈라지는가?
    - 필기체 A vs B: 어느 K 레벨에서 갈라지는가?
    - 한글 '인' vs '인과': 어느 K 레벨에서 갈라지는가?

    갈라지는 지점이 곧 개념의 차이다.
    """

    # 기본 스케일 레벨 (K값과 각 레벨의 의미)
    # K가 배수로 커질수록 세밀한 구조를 본다
    DEFAULT_SCALE_LADDER = [
        (0.05,  "거시적 패턴 (존재 여부)"),
        (0.2,   "도메인 분류 (어떤 종류)"),
        (1.0,   "카테고리 (어떤 범주)"),
        (5.0,   "서브카테고리 (어떤 하위 범주)"),
        (20.0,  "개념 구분 (같은 범주의 다른 개념)"),
        (80.0,  "세밀한 차이 (같은 개념의 변형)"),
        (300.0, "극세밀 (동일 여부 판별)"),
    ]
    RESONANCE_THRESHOLD = 0.25  # 이 이상이면 "공명"

    def __init__(self, scale_ladder: Optional[List[Tuple[float, str]]] = None):
        self.scale_ladder = scale_ladder or self.DEFAULT_SCALE_LADDER
        self._scratch_rotor = VariableRotor(dimensions=1)

    def _measure_resonance(self, signal_angle: float,
                           reference_angle: float,
                           k_value: float) -> float:
        """
        K 값이 주어진 로터에서 signal과 reference의 공명 강도 측정.

        원리: 로터를 reference에 고정 후, signal을 외력으로 가해
        얼마나 편향되는가를 측정.

        공명 강도 = 1 / (1 + K * |Δθ|²)
        (이것은 단조화 진동자의 정적 응답 함수와 동일)
        """
        diff = (signal_angle - reference_angle + math.pi) % (2 * math.pi) - math.pi
        # 단조화 응답: F/K = x → 공명 강도 = 1/(1 + K*diff²/π²)
        resonance = 1.0 / (1.0 + k_value * (diff / math.pi) ** 2)
        return float(resonance)

    def scan(self,
             signal_angle_rad: float,
             reference_angle_rad: float,
             label: str = "signal") -> FractalSignature:
        """
        단일 위상에 대한 프랙탈 스케일 스캔.
        각 K 레벨에서 공명 강도 측정 → FractalSignature 반환.
        """
        scales = []
        divergence_scale = None
        similarity_depth = 0

        for i, (k, desc) in enumerate(self.scale_ladder):
            bw_deg = math.degrees(math.pi / math.sqrt(k))
            res = self._measure_resonance(signal_angle_rad, reference_angle_rad, k)
            is_res = res >= self.RESONANCE_THRESHOLD

            scales.append(ScaleResonance(
                scale_index=i,
                k_value=k,
                bandwidth_deg=round(bw_deg, 1),
                resonance=round(res, 4),
                is_resonant=is_res,
                description=desc,
            ))

            if is_res:
                similarity_depth = i + 1
            elif divergence_scale is None:
                divergence_scale = i

        # 큰 스케일 vs 작은 스케일 레이블
        coarse = scales[0].description if scales[0].is_resonant else "비관련"
        fine   = scales[-1].description if scales[-1].is_resonant else \
                 (scales[similarity_depth-1].description if similarity_depth > 0 else "비관련")

        return FractalSignature(
            input_label=label,
            scales=scales,
            divergence_scale=divergence_scale,
            similarity_depth=similarity_depth,
            coarse_label=coarse,
            fine_label=fine,
        )

    def compare(self,
                sig_a_rad: float, label_a: str,
                sig_b_rad: float, label_b: str,
                reference_rad: float = 0.0) -> str:
        """
        두 신호를 같은 기준으로 스캔 후 비교.
        '어느 스케일에서 갈라지는가'를 반환.
        """
        sig_a = self.scan(sig_a_rad, reference_rad, label_a)
        sig_b = self.scan(sig_b_rad, reference_rad, label_b)
        return sig_a.compare_with(sig_b)

    def describe_signature(self, sig: FractalSignature) -> str:
        """서명을 읽기 쉬운 텍스트로."""
        lines = [f"[프랙탈 서명: '{sig.input_label}']"]
        lines.append(f"  유사도 깊이: {sig.similarity_depth}/{len(sig.scales)} 스케일")
        lines.append(f"  발산 스케일: {sig.divergence_scale}")
        lines.append(f"  큰 스케일 분류: {sig.coarse_label}")
        lines.append(f"  세밀 스케일 분류: {sig.fine_label}")
        lines.append("")
        for s in sig.scales:
            bar = "█" * int(s.resonance * 15)
            status = "✓ 공명" if s.is_resonant else "✗ 발산"
            lines.append(
                f"  K={s.k_value:6.1f} BW={s.bandwidth_deg:5.1f}° "
                f"[{bar:<15s}] {s.resonance:.3f} {status}  ({s.description})"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  SovereignAxe — 로터 오케스트레이터
# ═══════════════════════════════════════════════════════════════════════════

class SovereignAxe:
    """
    Variable Scale Rotor의 오케스트레이터.

    프랙탈 스캐너와 로터를 연결:
    - 공리 confidence → K 자동 매핑
    - 프랙탈 서명 기반 Lock/Unlock 결정
    """

    def __init__(self, rotor: VariableRotor):
        self.rotor = rotor
        self.scanner = FractalRotorScanner()

    def deliberate(self, intent_resonance: float) -> str:
        """Peek-a-boo Logic: 공명 강도에 따른 폭발적 동기화 or 결정화."""
        if intent_resonance > 0.95:
            self.rotor.locked_axes[:] = False
            return "EXPLOSIVE SYNCHRONIZATION: All axes fluid."
        elif intent_resonance < 0.2:
            idx = np.argmax(np.abs(self.rotor.state.imag))
            self.rotor.lock_axis(idx)
            return f"Crystallized axis {idx} to suppress chaos."
        return "Phase-lock stable."

    def apply_axiom_confidence(self, axis_confidences: Dict[int, float]):
        """
        공리 confidence를 로터의 K로 직접 변환.
        이것이 '공리 구조가 스케일을 결정한다'의 물리적 구현.

        confidence=1.0 (확립된 진리) → K=50.0 (좁은 대역폭, 딱딱함)
        confidence=0.3 (탐색 중)     → K=2.0  (넓은 대역폭, 유연함)
        """
        for axis_idx, conf in axis_confidences.items():
            self.rotor.set_axis_confidence(axis_idx, conf)

    def fractal_scan_axis(self, axis_idx: int,
                          signal_phase_rad: float,
                          reference_phase_rad: float,
                          label: str = "") -> FractalSignature:
        """
        특정 축에 대한 프랙탈 스케일 스캔.
        결과를 로터의 last_fractal_signature에 저장.
        """
        sig = self.scanner.scan(signal_phase_rad, reference_phase_rad, label or f"axis_{axis_idx}")
        self.rotor.last_fractal_signature = sig

        # 프랙탈 서명 기반으로 K 자동 조정:
        # 유사도가 높은 축은 K를 올려 세밀하게 측정
        # 유사도가 낮은 축은 K를 내려 포용적으로 탐색
        if 0 <= axis_idx < self.rotor.dims:
            sim_ratio = sig.similarity_ratio
            # sim_ratio 높으면 → 이미 수렴 중 → K 상승 (더 세밀하게)
            # sim_ratio 낮으면 → 발산 중 → K 하강 (더 포용적으로)
            current_k = self.rotor.K[axis_idx]
            target_k = 0.1 + sim_ratio * 49.9
            # 점진적 조정 (급격한 변화 방지)
            self.rotor.K[axis_idx] = current_k * 0.8 + target_k * 0.2

        return sig


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
    print("[FRACTAL SCALE ROTOR — 가변스케일 프랙탈 로터 테스트]")
    print("=" * 70)

    scanner = FractalRotorScanner()

    # ── 테스트 1: 고양이 vs 사자 ──────────────────────────────
    print("\n[1] 고양이 vs 사자 — 프랙탈 스케일 비교")
    print("    (같은 기준에서 어느 스케일에서 갈라지는가?)\n")

    # 예시: 고양이=15°, 사자=25°, 강아지=110° (같은 참조 위상 0° 기준)
    ref = 0.0
    cat_rad   = math.radians(15)
    lion_rad  = math.radians(25)
    dog_rad   = math.radians(110)

    cat_sig  = scanner.scan(cat_rad,  ref, "고양이")
    lion_sig = scanner.scan(lion_rad, ref, "사자")
    dog_sig  = scanner.scan(dog_rad,  ref, "강아지")

    print(scanner.describe_signature(cat_sig))
    print()
    print(scanner.describe_signature(lion_sig))
    print()
    print(scanner.describe_signature(dog_sig))

    print("\n── 교차 비교 ──")
    print(cat_sig.compare_with(lion_sig))
    print()
    print(cat_sig.compare_with(dog_sig))

    # ── 테스트 2: 필기체 변형 ─────────────────────────────────
    print("\n\n[2] 필기체 A 변형들 — 프랙탈 서명 비교")
    print("    (필기체는 큰 스케일에서는 같고 세밀한 스케일에서 갈린다)\n")

    a1_rad = math.radians(5)   # 깔끔한 A
    a2_rad = math.radians(12)  # 살짝 기울어진 A
    a3_rad = math.radians(35)  # 많이 변형된 A

    a1 = scanner.scan(a1_rad, ref, "필기체A (정자)")
    a2 = scanner.scan(a2_rad, ref, "필기체A (기울)")
    a3 = scanner.scan(a3_rad, ref, "필기체A (변형)")

    print(a1.compare_with(a2))
    print()
    print(a1.compare_with(a3))

    # ── 테스트 3: confidence → K 매핑 ────────────────────────
    print("\n\n[3] 공리 confidence → K 스케일 매핑")
    rotor = VariableRotor(dimensions=5)
    axe   = SovereignAxe(rotor)
    confidences = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.0}
    axe.apply_axiom_confidence(confidences)

    print("  confidence → K (강성) → 대역폭(°)")
    for i, conf in confidences.items():
        k = rotor.K[i]
        bw = math.degrees(math.pi / math.sqrt(k))
        bar = "█" * int(bw / 5)
        print(f"  conf={conf:.1f} → K={k:6.2f} → BW={bw:6.1f}° [{bar}]")

    # ── 테스트 4: 로터 실제 pulse ─────────────────────────────
    print("\n\n[4] 로터 pulse — K 스케일별 공명 응답")
    rotor2 = VariableRotor(dimensions=3)
    rotor2.set_scale(0.1, [0])   # 넓은 대역 (큰 스케일)
    rotor2.set_scale(5.0, [1])   # 중간
    rotor2.set_scale(80.0, [2])  # 좁은 대역 (세밀한 스케일)

    force = np.array([0.5, 0.5, 0.5])
    for step in range(5):
        rep = rotor2.pulse(force, dt=0.1)
        bw = rep["bandwidth"]
        print(f"  step {step}: angles={np.round(np.degrees(rep['angles']),1)} "
              f"| BW={np.round(np.degrees(bw),1)}° | enstrophy={rep['enstrophy']:.4f}")

    print("\n" + "=" * 70)
    print("[SELF-TEST COMPLETE]")
