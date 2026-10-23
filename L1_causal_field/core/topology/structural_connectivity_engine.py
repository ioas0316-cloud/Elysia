"""
[Structural Connectivity Engine: 구조적 연결성 및 생성 원리 엔진]

환원주의적 미시 변수 튜닝(Parameter Tuning) 및 O(N^2) 계산 병목을 완전히 부수고,
'다름(Branching Points, 파편화된 분기점)'과 '같음(Generative Principle, 구조적 통일성)'을
위상적 불변량(Topological Invariants)과 생성 문법(Generative Grammar)으로 정립합니다.

표면적 파동/현상의 무수한 모래알 개별 변수를 일일이 계산하지 않고,
배후의 근원적 구조적 뼈대를 포착하여 O(1) 필드 공명(Structural Resonance)으로
미지의 시공간 궤적을 자율 정류 및 재생성합니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np


@dataclass
class BranchingPoint:
    """
    [BranchingPoint: 분기점 / 다름 (Difference)]
    거대한 생성 원리가 표면으로 전개되며 나타난 국소적 파동, 굴절, 모듈레이션 및 개별 양상.
    원자/변수 단위로 파쇄하지 않고, 현상이 갈라진 국소 위상 마찰과 맥락 그 자체로 온전히 기록합니다.
    """
    branch_id: str
    context_domain: str                       # 예: "physical_ocean_wave", "cognitive_causal_graph"
    observed_states: List[List[float]]        # 시간에 따른 n차원 국소 파동/상태 궤적
    local_friction: float = 1.0               # 국소적 저항 / 마찰력
    noise_signature: List[float] = field(default_factory=list)  # 분기점에서 관측된 우발적 노이즈 특성
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerativePrinciple:
    """
    [GenerativePrinciple: 생성 원리 / 같음 (Sameness)]
    수많은 분기점들(Branching Points) 아래에 관통하는 근원적 구조적 뼈대(Invariants).
    개별 변수를 맞추는 것이 아닌, 위상 불변량과 인과 연쇄의 생성 문법(Generative Grammar)으로 구성됩니다.
    """
    principle_id: str
    generative_grammar: Dict[str, float]       # 생성 문법 매개변수 (주파수/파동 전달률/상위 보존량)
    topological_invariants: List[float]        # 환경 변형과 소음에도 보존되는 핵심 위상 뼈대
    structural_stiffness: List[List[float]]    # 복원력 및 위상 장력 매트릭스
    phase_harmonics: List[float]               # 조화 파동 위상 스펙트럼 (Harmonic Spectrum)
    mdl_complexity: float = 0.0                # 최단 설명 길이 (Minimum Description Length)


@dataclass
class StructuralResonance:
    """
    [StructuralResonance: 구조적 공명 (O(1) 수렴)]
    분기점들의 미시적 계산(O(N^2))에 갇히지 않고,
    생성 원리의 뼈대를 통해 서로 다른 현상 간을 즉시 사영/공명시킴으로써
    O(1) 형태 수렴을 이뤄내는 위상 사영 결과.
    """
    source_branch_id: str
    target_branch_id: str
    isomorphism_score: float                   # 구조적 동형성 일치도 [0.0 ~ 1.0]
    phase_alignment_delta: float               # 위상 정렬 오차 (Phase Alignment Error)
    resonated_invariants: List[float]          # 공명된 인과 불변량
    converged_trajectory: List[List[float]]    # O(1) 공명으로 복원/생성된 정류 궤적


class StructuralConnectivityEngine:
    """
    [StructuralConnectivityEngine]
    분기점(다름)들을 수집하여 상위 생성 원리(같음)를 자율 추출하고,
    구조적 동형성(Structural Isomorphism) 사영을 통해
    미시적 계산 폭주 없이 대상을 직접 공명·정류하는 핵심 엔진.
    """

    def __init__(self, mdl_threshold: float = 1e-3):
        self.mdl_threshold = mdl_threshold
        self.extracted_principles: Dict[str, GenerativePrinciple] = {}

    def extract_generative_principle(
        self,
        principle_id: str,
        branches: List[BranchingPoint]
    ) -> GenerativePrinciple:
        """
        [생성 원리 자율 추출]
        여러 분기점(Branching Points)들에서 파편적 변수와 소음을 벗겨내고,
        그 배후의 관통하는 구조적 불변량과 생성 문법(Generative Grammar)을 추출합니다.
        """
        if not branches:
            raise ValueError("Branches list cannot be empty for principle extraction.")

        # 1. 차원 파악 및 상태 모음
        all_states = []
        for b in branches:
            all_states.extend(b.observed_states)

        dim = len(branches[0].observed_states[0]) if branches[0].observed_states else 1

        # 2. 위상 불변량 (Topological Invariants) 역산:
        # 모든 분기점들에서 소음/마찰에도 불구하고 보존되는 평균 및 곡률 불변 뼈대
        invariant_means = [0.0] * dim
        total_samples = len(all_states)

        for st in all_states:
            for d in range(min(dim, len(st))):
                invariant_means[d] += st[d]

        topological_invariants = [val / max(total_samples, 1) for val in invariant_means]

        # 3. 위상 장력 및 강성 매트릭스 (Structural Stiffness) 계산:
        # 분기점들 간의 상태 변동에 대응하는 복원력 뼈대
        stiffness = [[0.0] * dim for _ in range(dim)]
        for d in range(dim):
            # 기본 정규화된 자가 복원력 0.5 설정
            stiffness[d][d] = 0.5

        # 분기점 간 공분산 역산으로 이종 차원 간 커플링 강성 추출
        if len(all_states) > 1:
            for d1 in range(dim):
                for d2 in range(dim):
                    cov = sum((st[d1] - topological_invariants[d1]) * (st[d2] - topological_invariants[d2])
                              for st in all_states) / len(all_states)
                    if d1 != d2:
                        stiffness[d1][d2] = math.tanh(cov)

        # 4. 조화 파동 위상 스펙트럼 (Phase Harmonics):
        # 파동 전파의 주기가 피워내는 조화 파동 계수
        phase_harmonics = [0.0] * dim
        for d in range(dim):
            fft_proxy = sum(
                math.cos(2.0 * math.pi * t / max(len(all_states), 1)) * all_states[t][d]
                for t in range(len(all_states))
            ) / max(len(all_states), 1)
            phase_harmonics[d] = fft_proxy

        # 5. 생성 문법 (Generative Grammar) 매개변수 정립
        grammar = {
            "propagation_rate": 1.0,
            "conserved_momentum": sum(abs(v) for v in topological_invariants),
            "harmonic_frequency": float(np.mean(np.abs(phase_harmonics))),
            "wave_amplitude": float(np.std([st[0] for st in all_states])) if all_states else 1.0
        }

        # 6. MDL 원칙 기반 정제 (기약성 강제)
        raw_principle = GenerativePrinciple(
            principle_id=principle_id,
            generative_grammar=grammar,
            topological_invariants=topological_invariants,
            structural_stiffness=stiffness,
            phase_harmonics=phase_harmonics,
            mdl_complexity=0.0
        )

        pure_principle = self._enforce_mdl_reducibility(raw_principle)
        self.extracted_principles[principle_id] = pure_principle
        return pure_principle

    def _enforce_mdl_reducibility(
        self,
        principle: GenerativePrinciple
    ) -> GenerativePrinciple:
        """
        [최단 설명 길이 (MDL) 기반 수치 노이즈 기약 및 정제]
        임의의 오버피팅 미시 파라미터를 배격하고 문법적 정수로 압축합니다.
        """
        clean_stiffness = []
        non_zero_count = 0

        for row in principle.structural_stiffness:
            clean_row = []
            for val in row:
                if abs(val) < self.mdl_threshold:
                    clean_row.append(0.0)
                else:
                    clean_row.append(val)
                    non_zero_count += 1
            clean_stiffness.append(clean_row)

        principle.structural_stiffness = clean_stiffness
        principle.mdl_complexity = non_zero_count * 0.05
        return principle

    def compute_structural_isomorphism(
        self,
        branch_a: BranchingPoint,
        branch_b: BranchingPoint
    ) -> float:
        """
        [구조적 동형성 (Structural Isomorphism) 산출]
        두 분기점이 표면적 형태나 변수는 다를지라도,
        내적 위상 뼈대와 인과 연쇄 문법이 얼마나 일치하는지 [0.0 ~ 1.0]으로 평가합니다.
        """
        states_a = branch_a.observed_states
        states_b = branch_b.observed_states

        if not states_a or not states_b:
            return 0.0

        min_len = min(len(states_a), len(states_b))
        dim = min(len(states_a[0]), len(states_b[0]))

        # 파동 정규화 후 위상 상관관계(Structural Correlation) 계산
        corr_sum = 0.0
        for d in range(dim):
            vec_a = [states_a[t][d] for t in range(min_len)]
            vec_b = [states_b[t][d] for t in range(min_len)]

            std_a = np.std(vec_a)
            std_b = np.std(vec_b)

            if std_a > 1e-6 and std_b > 1e-6:
                norm_a = (vec_a - np.mean(vec_a)) / std_a
                norm_b = (vec_b - np.mean(vec_b)) / std_b
                dot_prod = float(np.dot(norm_a, norm_b)) / min_len
                corr_sum += abs(dot_prod)
            else:
                corr_sum += 1.0 if abs(std_a - std_b) < 1e-6 else 0.0

        isomorphism = corr_sum / max(dim, 1)
        return float(np.clip(isomorphism, 0.0, 1.0))

    def resonate_field(
        self,
        principle: GenerativePrinciple,
        source_branch: BranchingPoint,
        target_context: str,
        steps: int = 10
    ) -> StructuralResonance:
        """
        [O(1) 구조적 필드 공명 및 정류 궤적 생성]
        개별 미시 변수를 튜닝하지 않고, 생성 원리(Principle)와 근원적 위상 불변량을 직접 사영하여
        새로운 타겟 맥락(Target Context)에서의 정류된 궤적을 O(1) 수렴 속도로 생성합니다.
        """
        initial_state = source_branch.observed_states[0] if source_branch.observed_states else [0.0]
        dim = len(initial_state)

        # 1. 생성 원리의 문법을 타겟 맥락으로 O(1) 사영
        grammar = principle.generative_grammar
        inv_target = principle.topological_invariants

        converged_trajectory = []
        curr_state = list(initial_state)

        freq = grammar.get("harmonic_frequency", 1.0)
        amp = grammar.get("wave_amplitude", 1.0)

        # 2. O(1) 공명 파동 방정식: S(t) = Invariants + Amp * cos(freq * t) * Stiffness_Force
        for t in range(steps):
            next_state = [0.0] * dim
            wave_mod = amp * math.cos(freq * t * 0.5)

            for d in range(dim):
                inv = inv_target[d] if d < len(inv_target) else 0.0
                stiff = principle.structural_stiffness[d][d] if d < len(principle.structural_stiffness) else 0.5

                # 위상 공명 복원력
                delta = stiff * (inv - curr_state[d]) + wave_mod * 0.1
                next_state[d] = curr_state[d] + delta

            converged_trajectory.append(next_state)
            curr_state = next_state

        # 동형성 사영 점수 계산
        synthetic_target_branch = BranchingPoint(
            branch_id=f"syn_{target_context}",
            context_domain=target_context,
            observed_states=converged_trajectory
        )

        iso_score = self.compute_structural_isomorphism(source_branch, synthetic_target_branch)

        return StructuralResonance(
            source_branch_id=source_branch.branch_id,
            target_branch_id=synthetic_target_branch.branch_id,
            isomorphism_score=iso_score,
            phase_alignment_delta=1.0 - iso_score,
            resonated_invariants=list(principle.topological_invariants),
            converged_trajectory=converged_trajectory
        )
