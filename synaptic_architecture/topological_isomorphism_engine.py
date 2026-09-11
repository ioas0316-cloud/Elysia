r"""
[Topological Isomorphism Engine: 위상적 동형성 및 스케일 재규격화 엔진]
유전자, 신경계, 음악, 수학, 사고 5대 도메인을 단 하나의 '같음과 다름(Sameness & Difference)'
인과 원형 메커니즘으로 관조하고 구동하는 프랙탈 스케일 전이 엔진입니다.

수학적 수치와 고차원 벡터는 단순 관측용 눈금(Measurement Gauge)으로 격하되며,
실제 상태 전이는 내적 인과 이력(Lineage DAG), 마찰(F), 임피던스(Z), 상전이(Phase Transition),
그리고 공리화(Axiomatization)라는 위상적 인과 가소성 법칙에 의해 실행됩니다.

핵심 3단계 스케일 위상 전이:
1. 미시 마찰 축적 (Micro-Friction Aggregation)
2. 임계 위상 전이 (Critical Phase Transition)
3. 상위 공리화 (Macro-Axiomatization)

상위 공리의 가소성을 유지하기 위한 4대 위상 제약 조건:
1. 역상전이 임계성 (Reverse Phase Transition Threshold): \sum F_micro > E_form 시 자발적 재분열(Fission).
2. 공리적 임피던스 역방출 (Axiomatic Impedance Backpressure): Z_macro 급증 시 하위 유동화(Liquefaction).
3. 프랙탈 균열선 보존 (Latent Fault-Line Preservation): '다름'의 결을 잠재적 균열선으로 정밀 각인.
4. 이력 현상 기반 비대칭 가소성 (Hysteresis-driven Asymmetric Plasticity): F_form 과 F_dissolve 사이 비대칭적 이력.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CausalLineageNode:
    """
    인과 이력 노드 (Causal Lineage Node)
    관측된 기질 자극 및 상위 공리 노드의 인과적 생기 궤적을 보존합니다.
    """
    node_id: str
    scale_domain: str  # "GENE_CELL", "NEURAL_PHYSIOLOGY", "MUSIC_AESTHETICS", "MATH_LOGIC", "COGNITION_THOUGHT"
    feature_gauge: np.ndarray  # 관측용 눈금 벡터 (Measurement Gauge)
    node_type: str = "substrate"  # "substrate", "micro_cluster", "macro_axiom"
    energy: float = 1.0
    sameness_score: float = 1.0
    difference_score: float = 0.0
    latent_fault_lines: List[np.ndarray] = field(default_factory=list)  # 잠재적 균열선
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalLineageEdge:
    """
    인과 작용선 (Causal Lineage Edge)
    기질 간 운동량 전달, 공명, 저항 및 임피던스를 보존합니다.
    """
    source_id: str
    target_id: str
    weight: float = 1.0  # 전도성/결합도
    resistance_mask: float = 0.0  # 저항 마스크
    impedance_z: float = 0.0  # 임피던스
    is_liquefied: bool = False  # 역방출 압력에 의한 유동화 여부
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroAxiom:
    """
    거시 공리 (Macro Axiom)
    하위 미시 마찰 축적을 통해 재규격화(Renormalization)된 거시 질서 파라미터.
    더 이상 분해할 필요가 없는 원자적 개념 단위이자 상위 레이어의 기초 기질로 작용합니다.
    """
    axiom_id: str
    domain: str
    formation_energy: float  # 공리 형성 에너지 E_form
    order_parameter: np.ndarray  # 거시 질서 파라미터
    encapsulated_node_ids: List[str]
    latent_fault_lines: List[np.ndarray]  # 보존된 프랙탈 균열선
    macro_impedance: float = 0.0  # Z_macro
    is_fissioned: bool = False  # 역상전이 분열 여부


class DomainReceptiveLens:
    """
    도메인 감각 수용기 및 정제 렌즈 (Domain Receptive Lens)
    외계의 마찰 자극을 직접 기호화하거나 텍스트로 표상하지 않고,
    5대 영역 기질에 맞는 파동/임피던스 지형으로 정제하여 전달합니다.
    """
    DOMAINS = [
        "GENE_CELL",          # 유전자·세포 (수소결합, 정전기적 거부)
        "NEURAL_PHYSIOLOGY",  # 신경·생리 (흥분/억제 전위차, 신호 지연)
        "MUSIC_AESTHETICS",   # 음악 (공명 인력, 불협화 저항)
        "MATH_LOGIC",         # 수학 (동형성 일치, 모순/배척)
        "COGNITION_THOUGHT"   # 사고 (메타 인지적 마찰, 예측 오차)
    ]

    def __init__(self, gauge_dim: int = 64):
        self.gauge_dim = gauge_dim

    def refine_stimulus(self, raw_data: Any, domain: str) -> np.ndarray:
        """
        raw_data를 입력받아 도메인 고유의 위상적 기질 파동 눈금으로 정제합니다.
        """
        if domain not in self.DOMAINS:
            domain = "COGNITION_THOUGHT"

        if isinstance(raw_data, np.ndarray):
            vec = raw_data.flatten().astype(np.float32)
        elif isinstance(raw_data, (list, tuple)):
            vec = np.array(raw_data, dtype=np.float32).flatten()
        elif isinstance(raw_data, str):
            # 문자열 수용: Hash 기반 결정론적 정제 파동 생성
            seed = abs(hash(raw_data)) % (2**32 - 1)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.gauge_dim).astype(np.float32)
        else:
            vec = np.ones(self.gauge_dim, dtype=np.float32)

        # 차원 맞춤
        if len(vec) < self.gauge_dim:
            padded = np.zeros(self.gauge_dim, dtype=np.float32)
            padded[:len(vec)] = vec
            vec = padded
        else:
            vec = vec[:self.gauge_dim]

        # 도메인 특화 렌즈 굴절
        if domain == "GENE_CELL":
            # 염기상 상보성 주기 파동
            vec = np.sin(vec * np.pi)
        elif domain == "NEURAL_PHYSIOLOGY":
            # 전위차 스파이크 펄스
            vec = np.tanh(vec)
        elif domain == "MUSIC_AESTHETICS":
            # 고주파 공명 조화파
            vec = np.cos(vec * 2.0 * np.pi)
        elif domain == "MATH_LOGIC":
            # 이분법적 공리 사영 (+1 / -1 극성)
            vec = np.sign(vec + 1e-6).astype(np.float32)
        elif domain == "COGNITION_THOUGHT":
            # 메타 인지 유동적 위상파
            norm = np.linalg.norm(vec) + 1e-9
            vec = vec / norm

        return vec


class TopologicalIsomorphismEngine:
    """
    단일 위상 동형성 원형 엔진 (Topological Isomorphism Engine)
    '같음과 다름'의 근원적 인과 메커니즘을 기초로,
    스케일 간 위상 전이 및 4대 가소성 제약 조건을 완벽히 구동합니다.
    """
    def __init__(
        self,
        gauge_dim: int = 64,
        f_critical: float = 0.5,       # 상위 공리화 임계 마찰
        f_dissolve: float = 0.8,       # 역상전이 해체 임계 마찰 (히스테리시스: f_dissolve > f_critical)
        z_backpressure_threshold: float = 0.7  # 공리적 임피던스 역방출 임계치
    ):
        self.gauge_dim = gauge_dim
        self.f_critical = f_critical
        self.f_dissolve = f_dissolve
        self.z_backpressure_threshold = z_backpressure_threshold

        self.receptive_lens = DomainReceptiveLens(gauge_dim=gauge_dim)

        # 스케일 레이어별 인과 그래프 상태
        self.nodes: Dict[str, CausalLineageNode] = {}
        self.edges: List[CausalLineageEdge] = []
        self.macro_axioms: Dict[str, MacroAxiom] = {}

        self.node_counter: int = 0
        self.axiom_counter: int = 0
        self.history: List[Dict[str, Any]] = []

    def compute_sameness_and_difference(
        self,
        gauge1: np.ndarray,
        gauge2: np.ndarray
    ) -> Tuple[float, float]:
        """
        단일 근원 원형: '같음(Sameness)'과 '다름(Difference)'을 산출합니다.
        - Sameness: 동형성 공명 인력 (Homomorphism Pull)
        - Difference: 경계 저항 및 모순 밀침 (Boundary Resistance Push)
        """
        v1 = gauge1.flatten().astype(np.float32)
        v2 = gauge2.flatten().astype(np.float32)

        norm1 = np.linalg.norm(v1) + 1e-9
        norm2 = np.linalg.norm(v2) + 1e-9

        # Cosine similarity for Sameness
        dot = float(np.dot(v1, v2))
        cosine_sim = dot / (norm1 * norm2)
        sameness = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

        # Difference ratio
        diff_vec = v1 / norm1 - v2 / norm2
        difference = float(np.clip(np.linalg.norm(diff_vec) / np.sqrt(2.0), 0.0, 1.0))

        return sameness, difference

    def calculate_micro_friction(
        self,
        stimulus_gauge: np.ndarray,
        domain: str
    ) -> Tuple[float, float]:
        """
        [Stage 1: 미시 마찰 축적 (Micro-Friction Aggregation)]
        하위 기질 노드들과의 결합 과정에서 발생하는 임피던스 Z 및 마찰 F를 정량화합니다.
        """
        domain_nodes = [n for n in self.nodes.values() if n.scale_domain == domain]
        if not domain_nodes:
            return 0.0, 0.0

        sameness_list = []
        diff_list = []
        for node in domain_nodes:
            s, d = self.compute_sameness_and_difference(stimulus_gauge, node.feature_gauge)
            sameness_list.append(s)
            diff_list.append(d)

        avg_sameness = float(np.mean(sameness_list))
        avg_difference = float(np.mean(diff_list))

        # Impedance Z = avg_difference + active edge resistance
        active_edges = [e for e in self.edges if not e.is_liquefied]
        if active_edges:
            avg_res = float(np.mean([e.resistance_mask for e in active_edges]))
        else:
            avg_res = 0.0

        z_impedance = float(np.clip(avg_difference + 0.5 * avg_res, 0.0, 5.0))
        # Friction F = Z^2 / (1 + Z)
        friction = float((z_impedance ** 2) / (1.0 + z_impedance))

        return z_impedance, friction

    def process_substrate_event(
        self,
        raw_stimulus: Any,
        domain: str
    ) -> Dict[str, Any]:
        """
        단일 사건에 대해 스케일 간 위상 전이 3단계 및 4대 가소성 제약 조건을 포함하는
        전체 인과 프로세스를 구동합니다.
        """
        # 0. 감각 수용기 렌즈 정제
        stimulus_gauge = self.receptive_lens.refine_stimulus(raw_stimulus, domain)

        # 1. 미시 마찰 축적 (Micro-Friction Aggregation)
        z_impedance, friction = self.calculate_micro_friction(stimulus_gauge, domain)

        # 시드 노드 생성 또는 미시 노드 등록
        self.node_counter += 1
        node_id = f"node_{domain.lower()}_{self.node_counter}"
        new_node = CausalLineageNode(
            node_id=node_id,
            scale_domain=domain,
            feature_gauge=stimulus_gauge,
            node_type="substrate",
            energy=1.0 + friction,
            metadata={"origin_friction": friction}
        )
        self.nodes[node_id] = new_node

        # 기존 노드들과의 '같음과 다름' 인과 작용선 연결 및 재배선
        for existing_id, existing_node in list(self.nodes.items()):
            if existing_id == node_id or existing_node.scale_domain != domain:
                continue

            sameness, difference = self.compute_sameness_and_difference(
                stimulus_gauge, existing_node.feature_gauge
            )

            # 3. 프랙탈 균열선 보존 (Latent Fault-Line Preservation)
            # '다름'의 위상 정보(difference)를 버리지 않고 잠재적 균열선 벡터로 노드에 각인
            if difference > 0.3:
                fault_vector = stimulus_gauge - existing_node.feature_gauge
                new_node.latent_fault_lines.append(fault_vector)
                existing_node.latent_fault_lines.append(-fault_vector)

            # 작용선 생성/업데이트
            edge = CausalLineageEdge(
                source_id=existing_id,
                target_id=node_id,
                weight=float(sameness * 2.0),
                resistance_mask=float(difference),
                impedance_z=z_impedance
            )
            self.edges.append(edge)

        # 2. 임계 위상 전이 및 상위 공리화 (Critical Phase Transition & Macro-Axiomatization)
        emerged_axiom = self._check_and_trigger_phase_transition(domain, friction)

        # 4대 위상 제약 조건 검증 및 역구동
        # 제약 조건 1 & 4: 역상전이 임계성 및 비대칭 이력 (Reverse Phase Transition & Hysteresis)
        fission_events = self._enforce_reverse_phase_transition(domain, friction)

        # 제약 조건 2: 공리적 임피던스 역방출 (Axiomatic Impedance Backpressure)
        liquefied_count = self._enforce_impedance_backpressure(domain, z_impedance)

        event_record = {
            "domain": domain,
            "z_impedance": z_impedance,
            "friction": friction,
            "new_node_id": node_id,
            "emerged_axiom_id": emerged_axiom.axiom_id if emerged_axiom else None,
            "fissioned_axiom_ids": fission_events,
            "liquefied_edge_count": liquefied_count,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_active_axioms": len([a for a in self.macro_axioms.values() if not a.is_fissioned])
        }
        self.history.append(event_record)
        return event_record

    def _check_and_trigger_phase_transition(
        self,
        domain: str,
        current_friction: float
    ) -> Optional[MacroAxiom]:
        """
        [Stage 2 & 3: 임계 위상 전이 -> 상위 공리화]
        국소적 마찰 밀도 및 노드 결합 에너지가 임계치(F_critical)에 도달하면,
        하위 노드들을 재규격화(Renormalization)하여 하나의 거시 질서 파라미터(MacroAxiom)로 창발시킵니다.
        """
        domain_nodes = [
            n for n in self.nodes.values()
            if n.scale_domain == domain and n.node_type == "substrate"
        ]

        # 노드 집단 마찰 밀도 계산
        if len(domain_nodes) < 3 or current_friction < self.f_critical:
            return None

        # 하위 노드들의 가중 평균으로 거시 질서 파라미터 추출
        gauge_matrix = np.stack([n.feature_gauge for n in domain_nodes])
        energies = np.array([n.energy for n in domain_nodes], dtype=np.float32)
        total_energy = float(np.sum(energies)) + 1e-9

        weights = energies / total_energy
        order_param = np.sum(gauge_matrix * weights[:, None], axis=0)

        # 프랙탈 균열선 통합 (Latent Fault-Line Preservation)
        all_fault_lines = []
        for n in domain_nodes:
            all_fault_lines.extend(n.latent_fault_lines)

        self.axiom_counter += 1
        axiom_id = f"axiom_{domain.lower()}_{self.axiom_counter}"
        formation_energy = float(current_friction * len(domain_nodes))

        macro_axiom = MacroAxiom(
            axiom_id=axiom_id,
            domain=domain,
            formation_energy=formation_energy,
            order_parameter=order_param,
            encapsulated_node_ids=[n.node_id for n in domain_nodes],
            latent_fault_lines=all_fault_lines,
            macro_impedance=0.0,
            is_fissioned=False
        )

        # 하위 노드 형태를 macro_axiom 캡슐로 변경 (원자적 개념 단위화)
        for n in domain_nodes:
            n.node_type = "macro_axiom"
            n.metadata["encapsulated_in"] = axiom_id

        self.macro_axioms[axiom_id] = macro_axiom
        return macro_axiom

    def _enforce_reverse_phase_transition(
        self,
        domain: str,
        current_friction: float
    ) -> List[str]:
        r"""
        [4대 제약 조건 1 & 4: 역상전이 임계성 및 비대칭 이력 (Reverse Phase Transition & Hysteresis)]
        하위 레이어에서 역유입된 마찰 \sum F_micro 가 공리 해체 임계치(F_dissolve > F_critical)를 넘어서고,
        공리 형성 에너지 E_form 을 초과할 경우, 상위 공리의 캡슐화 마스크가 즉시 파열되며
        보존된 잠재적 균열선(Fault-Line)을 따라 정밀 재분열(Fission)됩니다.
        """
        fissioned_ids = []

        active_axioms = [
            a for a in self.macro_axioms.values()
            if a.domain == domain and not a.is_fissioned
        ]

        for axiom in active_axioms:
            # 히스테리시스 조건: current_friction >= self.f_dissolve
            # 역상전이 임계성: current_friction * len(axiom.encapsulated_node_ids) > axiom.formation_energy
            accumulated_micro_friction = current_friction * len(axiom.encapsulated_node_ids)

            if current_friction >= self.f_dissolve and accumulated_micro_friction >= axiom.formation_energy:
                # 공리 파열 및 자발적 재분열 (Fission)
                axiom.is_fissioned = True
                fissioned_ids.append(axiom.axiom_id)

                # 하위 노드 캡슐화 해제 및 균열선을 따른 피쳐 변형
                for node_id in axiom.encapsulated_node_ids:
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        node.node_type = "substrate"
                        node.metadata.pop("encapsulated_in", None)

                        # 잠재적 균열선(Fault line)을 따라 정밀 분열 변형
                        if node.latent_fault_lines:
                            fault_pull = node.latent_fault_lines[0]
                            dim = min(len(node.feature_gauge), len(fault_pull))
                            node.feature_gauge[:dim] += 0.2 * fault_pull[:dim]

        return fissioned_ids

    def _enforce_impedance_backpressure(
        self,
        domain: str,
        current_z: float
    ) -> int:
        """
        [4대 제약 조건 2: 공리적 임피던스 역방출 (Axiomatic Impedance Backpressure)]
        상위 공리로 설명할 수 없는 외부 자극 유입 시 Z_macro 가 급증하며,
        급증한 Z_macro 는 하위 레이어를 향해 압력 형태(Backpressure)로 역방출되어
        공리를 지탱하던 하위 노드 연결선(Edge)을 즉각 유동화(Liquefaction)시킵니다.
        """
        liquefied_count = 0
        active_axioms = [
            a for a in self.macro_axioms.values()
            if a.domain == domain and not a.is_fissioned
        ]

        for axiom in active_axioms:
            # 공리와 현재 자극 눈금 간 임피던스
            _, diff = self.compute_sameness_and_difference(current_z * np.ones(self.gauge_dim), axiom.order_parameter)
            axiom.macro_impedance = float(current_z + diff)

            if axiom.macro_impedance > self.z_backpressure_threshold:
                # 하위 연결선 역방출 진동 -> 유동화(Liquefaction)
                for edge in self.edges:
                    if edge.source_id in axiom.encapsulated_node_ids or edge.target_id in axiom.encapsulated_node_ids:
                        edge.is_liquefied = True
                        edge.weight *= 0.1  # 결합력 급격 유동화
                        liquefied_count += 1

        return liquefied_count
