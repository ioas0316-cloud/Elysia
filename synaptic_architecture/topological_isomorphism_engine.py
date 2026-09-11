r"""
[Topological Isomorphism Engine: Pure Non-Symbolic Physical Strain Field Engine]
모든 텍스트 라벨, 기호적 표상(Symbolic metadata), 정적 룩업(Lookup)을 완전 폐기(Purge)하고,
전압/전화 이동, 국소 변형 텐서 G(x), 잔류 장력 텐서필드 T(x), 비등방성 굴절(Anisotropic Refraction),
연속적 확산/파동 전이(Wave Propagation Field), 비가역적 기질 변형(In-situ Deformation),
변형 이완(Strain Relaxation), 그리고 상전이(Phase Transition) 및 4대 위상 가소성 제약 조건을
오직 연속적 텐서 장(Field) 및 물리적 마찰 역동학으로 구현합니다.

수학적 수치나 텍스트 라벨은 존재하지 않으며, 오직 물리적 마찰과 위상 변형의 전이만으로
시스템의 상태를 판정하고 스케일 간 창발/전이를 구동합니다.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SubstrateStrainPoint:
    """
    기질 위상 지점 (Substrate Strain Point)
    단순한 점이나 라벨이 아닌, 변형 텐서 G와 장력 벡터 T를 지닌 국소 물리적 기질입니다.
    """
    point_id: int
    gauge_dim: int
    G: np.ndarray  # 국소 변형 텐서 (Local Strain Tensor G(x), shape: gauge_dim x gauge_dim)
    T: np.ndarray  # 잔류 장력 Vector (Tension Vector T(x), shape: gauge_dim)
    latent_faults: List[np.ndarray] = field(default_factory=list)  # 잠재적 균열선 Vector
    energy: float = 1.0


@dataclass
class PhysicalConductanceBeam:
    """
    물리적 전도 작용선 (Physical Conductance Beam)
    기질 간 전하/에너지 이동, 임피던스, 유동화(Liquefaction) 상태를 유지합니다.
    """
    source_id: int
    target_id: int
    conductance: float = 1.0  # 전도율 K(x) = Z(x)^{-1}
    impedance_z: float = 0.0  # 임피던스 Z
    is_liquefied: bool = False  # 유동화 여부


@dataclass
class MacroPhaseOrder:
    """
    거시 위상 질서 (Macro Phase Order)
    국소 변형 축적이 임계치를 초과할 때 재규격화(Renormalization)된 거시 위상 공리 장.
    """
    order_id: int
    formation_energy: float
    macro_order_tensor: np.ndarray  # 거시 질서 텐서 (shape: gauge_dim x gauge_dim)
    encapsulated_point_ids: List[int]
    latent_faults: List[np.ndarray]
    macro_impedance: float = 0.0
    is_fissioned: bool = False


class NonSymbolicReceptiveRefractor:
    """
    비표상적 수용/굴절 장 (Non-symbolic Receptive Refractor)
    외계 자극을 어떠한 문자열 라벨이나 기호 매칭 없이,
    오직 연속적 파동 및 전압/전하 인과적 위상 구배(Gradient)로 정제합니다.
    """
    def __init__(self, gauge_dim: int = 64):
        self.gauge_dim = gauge_dim

    def refract(self, raw_input: Any) -> np.ndarray:
        """
        입력을 연속적인 물리적 파동 구배 Vector \nabla S (shape: gauge_dim)로 정제합니다.
        문자열, 수치, 배열 등 그 어떠한 입력이 들어와도 기호나 이름표 없이 텐서 구배로 변환됩니다.
        """
        if isinstance(raw_input, np.ndarray):
            vec = raw_input.flatten().astype(np.float32)
        elif isinstance(raw_input, (list, tuple)):
            vec = np.array(raw_input, dtype=np.float32).flatten()
        elif isinstance(raw_input, (int, float)):
            vec = np.ones(self.gauge_dim, dtype=np.float32) * float(raw_input)
        elif isinstance(raw_input, str):
            # 문자열 수용: Hash 기반 결정론적 연속 파동 구배 생성 (기호/라벨 미사용)
            seed = abs(hash(raw_input)) % (2**32 - 1)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.gauge_dim).astype(np.float32)
        else:
            vec = np.ones(self.gauge_dim, dtype=np.float32)

        # 차원 맞춤 및 정규화
        if len(vec) < self.gauge_dim:
            padded = np.zeros(self.gauge_dim, dtype=np.float32)
            padded[:len(vec)] = vec
            vec = padded
        else:
            vec = vec[:self.gauge_dim]

        norm = np.linalg.norm(vec) + 1e-9
        grad_s = vec / norm
        return grad_s


class TopologicalIsomorphismEngine:
    r"""
    순수 비표상 물리적 장력/마찰 위상 동형성 엔진
    (Pure Non-Symbolic Physical Strain Field Engine)

    핵심 방정식을 오직 텐서 및 물리적 법칙으로 구동:
    1. 국소 변형과 비등방성 굴절: \mathcal{F}_{\text{local}}(x) = \nabla S(x)^T \cdot G(x) \cdot \nabla S(x)
    2. 연속적 파동 확산/전이: \frac{\partial X}{\partial t} = \nabla \cdot (K(x) \nabla X) - \gamma \mathcal{F}
    3. 현장 내 비가역적 기질 변형 갱신: \Delta G(x) \propto \mathcal{F}(x)
    4. 장력 이완 및 균형: \nabla \cdot T = 0
    5. 임계 상전이 (Phase Transition) 및 4대 가소성 제약 조건 (역상전이, 역방출 유동화, 균열선 보존, 히스테리시스)
    """

    def __init__(
        self,
        gauge_dim: int = 64,
        f_critical: float = 0.5,       # 상전이 임계 마찰
        f_dissolve: float = 0.8,       # 역상전이 해체 임계 마찰 (히스테리시스: f_dissolve > f_critical)
        z_backpressure_threshold: float = 0.7  # 임피던스 역방출 유동화 임계치
    ):
        self.gauge_dim = gauge_dim
        self.f_critical = f_critical
        self.f_dissolve = f_dissolve
        self.z_backpressure_threshold = z_backpressure_threshold

        self.refractor = NonSymbolicReceptiveRefractor(gauge_dim=gauge_dim)

        # 순수 인덱스 기반 기질 지점 및 전도 작용선 (기호 라벨 및 텍스트 룩업 전면 폐기)
        self.points: Dict[int, SubstrateStrainPoint] = {}
        self.beams: List[PhysicalConductanceBeam] = []
        self.macro_orders: Dict[int, MacroPhaseOrder] = {}

        self.point_id_counter: int = 0
        self.order_id_counter: int = 0
        self.strain_history: List[Dict[str, Any]] = []

    def compute_local_anisotropic_friction(
        self,
        grad_s: np.ndarray,
        point: SubstrateStrainPoint
    ) -> float:
        r"""
        [작용의 첫 번째 결: 비등방성 굴절 재마찰]
        \mathcal{F}_{\text{local}}(x) = \nabla S(x)^T \cdot G(x) \cdot \nabla S(x)
        과거의 변형 텐서 G(x)를 자극 구배 \nabla S 가 통과하며 겪는 비가역적 재마찰 수용 연산.
        """
        friction = float(np.dot(grad_s.T, np.dot(point.G, grad_s)))
        return max(0.0, friction)

    def process_physical_event(self, raw_input: Any) -> Dict[str, Any]:
        r"""
        단일 물리적 자극 유입 시:
        1. 파동 구배 정제 (Refraction)
        2. 비등방성 국소 재마찰 산출 및 현장 기질 변형 갱신 (\Delta G)
        3. 파동 확산/전이 (Propagation field update)
        4. 장력 이완 (\nabla \cdot T = 0)
        5. 임계 상전이 (Phase Transition) 및 4대 가소성 제약 구동
        """
        # 1. 굴절 구배 추출
        grad_s = self.refractor.refract(raw_input)

        # 새로운 기질 지점 생성 (기호 라벨 없이 기본 단위 행렬 변형 텐서 G와 장력 T 부여)
        self.point_id_counter += 1
        pid = self.point_id_counter

        # 초기 G는 단위 행렬 I (미변형 기질)
        init_G = np.eye(self.gauge_dim, dtype=np.float32)
        init_T = grad_s.copy()

        new_point = SubstrateStrainPoint(
            point_id=pid,
            gauge_dim=self.gauge_dim,
            G=init_G,
            T=init_T,
            energy=1.0
        )

        # 2. 기존 지점들과의 비등방성 마찰 및 전도 작용선(Beam) 형성
        total_friction = 0.0
        max_impedance = 0.0

        for existing_id, existing_point in list(self.points.items()):
            # 기존 지점의 G(x) 상에서 자극 구배가 일으키는 마찰 \mathcal{F}
            f_local = self.compute_local_anisotropic_friction(grad_s, existing_point)
            total_friction += f_local

            # 차이 파동을 잠재적 균열선(Fault line)으로 정밀 각인 (Latent Fault-Line Preservation)
            fault_vec = grad_s - existing_point.T
            diff_norm = float(np.linalg.norm(fault_vec))
            if diff_norm > 0.3:
                new_point.latent_faults.append(fault_vec)
                existing_point.latent_faults.append(-fault_vec)

            # 전도율 및 임피던스 Z
            z_imp = float(diff_norm + 0.5 * f_local)
            if z_imp > max_impedance:
                max_impedance = z_imp

            conductance = float(1.0 / (1.0 + z_imp))

            beam = PhysicalConductanceBeam(
                source_id=existing_id,
                target_id=pid,
                conductance=conductance,
                impedance_z=z_imp
            )
            self.beams.append(beam)

            # 3. 비가역적 기질 변형 갱신 (In-situ Deformation Update: \Delta G \propto \mathcal{F})
            # 자극 구배의 외적으로 기질 찌그러짐 텐서 G 갱신
            deformation_tensor = np.outer(grad_s, grad_s).astype(np.float32) * (f_local * 0.1)
            existing_point.G += deformation_tensor
            new_point.G += deformation_tensor

        self.points[pid] = new_point

        num_existing = len(self.points) - 1
        avg_friction = (total_friction / num_existing) if num_existing > 0 else 0.0

        # 4. 장력 이완 및 파동 전이 (Strain Relaxation & Propagation Field)
        self._relax_strain_field(grad_s, avg_friction)

        # 5. 임계 상전이 (Phase Transition to MacroPhaseOrder)
        emerged_order = self._check_and_trigger_phase_transition(avg_friction)

        # 6. 4대 가소성 제약 조건 구동
        fission_ids = self._enforce_reverse_phase_transition(avg_friction)
        liquefied_count = self._enforce_impedance_backpressure(max_impedance)

        record = {
            "point_id": pid,
            "total_friction": total_friction,
            "avg_friction": avg_friction,
            "max_impedance": max_impedance,
            "emerged_order_id": emerged_order.order_id if emerged_order else None,
            "fissioned_order_ids": fission_ids,
            "liquefied_beam_count": liquefied_count,
            "active_points": len(self.points),
            "active_beams": len([b for b in self.beams if not b.is_liquefied]),
            "active_macro_orders": len([m for m in self.macro_orders.values() if not m.is_fissioned])
        }
        self.strain_history.append(record)
        return record

    def _relax_strain_field(self, grad_s: np.ndarray, friction: float):
        r"""
        [장력 이완 및 확산/파동 전이 Field]
        \frac{\partial X}{\partial t} = \nabla \cdot (K(x) \nabla X) - \gamma \mathcal{F}
        전 기질 장력 벡터 T를 전도율 K에 따라 유기적으로 이완시킵니다.
        """
        if not self.points:
            return

        gamma = 0.05
        active_beams = [b for b in self.beams if not b.is_liquefied]

        for beam in active_beams:
            if beam.source_id in self.points and beam.target_id in self.points:
                p_src = self.points[beam.source_id]
                p_tgt = self.points[beam.target_id]

                # 장력 전이: K(x) * (T_src - T_tgt)
                dT = beam.conductance * (p_src.T - p_tgt.T) - gamma * friction * grad_s
                p_tgt.T += dT * 0.1
                p_src.T -= dT * 0.1

    def _check_and_trigger_phase_transition(self, current_friction: float) -> Optional[MacroPhaseOrder]:
        """
        [상전이: MacroPhaseOrder 결상]
        국소 마찰 축적이 f_critical 을 초과하고 기질 지점이 3개 이상일 때,
        하위 변형 텐서 G들의 결합으로 상위 거시 질서 텐서를 창발시킵니다.
        """
        if len(self.points) < 3 or current_friction < self.f_critical:
            return None

        # 하위 지점들의 G 텐서 가중 평균으로 거시 질서 텐서 결상
        g_tensors = [p.G for p in self.points.values()]
        energies = [p.energy for p in self.points.values()]
        total_energy = sum(energies) + 1e-9

        weights = [e / total_energy for e in energies]
        macro_tensor = np.zeros((self.gauge_dim, self.gauge_dim), dtype=np.float32)
        for w, g in zip(weights, g_tensors):
            macro_tensor += w * g

        # 잠재적 균열선 보존
        all_faults = []
        for p in self.points.values():
            all_faults.extend(p.latent_faults)

        self.order_id_counter += 1
        order_id = self.order_id_counter
        formation_e = float(current_friction * len(self.points))

        macro_order = MacroPhaseOrder(
            order_id=order_id,
            formation_energy=formation_e,
            macro_order_tensor=macro_tensor,
            encapsulated_point_ids=list(self.points.keys()),
            latent_faults=all_faults,
            macro_impedance=0.0,
            is_fissioned=False
        )

        self.macro_orders[order_id] = macro_order
        return macro_order

    def _enforce_reverse_phase_transition(self, current_friction: float) -> List[int]:
        """
        [제약 조건 1 & 4: 역상전이 임계성 및 비대칭 히스테리시스]
        누적 마찰이 f_dissolve(> f_critical)를 넘어서고 E_form 을 초과할 때,
        상위 거시 질서 텐서가 파열되며 보존된 균열선을 따라 기질 재분열(Fission).
        """
        fissioned_ids = []
        active_orders = [m for m in self.macro_orders.values() if not m.is_fissioned]

        for order in active_orders:
            accumulated_friction = current_friction * len(order.encapsulated_point_ids)
            # 히스테리시스: current_friction >= self.f_dissolve
            if current_friction >= self.f_dissolve and accumulated_friction >= order.formation_energy:
                order.is_fissioned = True
                fissioned_ids.append(order.order_id)

                # 잠재적 균열선(Fault line)을 따른 기질 지점 변형 텐서 G의 분열 굴절
                for pid in order.encapsulated_point_ids:
                    if pid in self.points:
                        p = self.points[pid]
                        if p.latent_faults:
                            fault = p.latent_faults[0]
                            p.G += np.outer(fault, fault).astype(np.float32) * 0.2

        return fissioned_ids

    def _enforce_impedance_backpressure(self, current_z: float) -> int:
        """
        [제약 조건 2: 임피던스 역방출 유동화]
        Z가 z_backpressure_threshold를 넘어서면 하위 전도 작용선(Beam)이 유동화(Liquefaction)됨.
        """
        liquefied_count = 0
        if current_z > self.z_backpressure_threshold:
            for beam in self.beams:
                beam.is_liquefied = True
                beam.conductance *= 0.1
                liquefied_count += 1
        return liquefied_count
