"""
Topological Cognitive Verifier (구속조건 기반 위상적 인지 판별 엔진)
======================================================================
언어를 단순 통계 기호가 아닌 개념 간 반발력과 인과적 구속을 만들어내는
동적 장(Boundary Field)으로 정의하고, 입력 진술의 위상적 응력(Stress)과
이완 가능 여부(Topological Relaxation)로 모순 및 은유를 판정하는 엔진입니다.

Core Directives & Philosophy:
- "Do not calculate, let it flow."
- numpy 기반의 연속적인 3D/4D 위상 다면체(Polytope) 충돌, 응력 텐서 연산,
  및 이완 역학 수렴(Phase Transition)을 통해 주체적으로 논리적 유효성을 판별합니다.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum


class GenreDomain(Enum):
    MATH_CODE = "math_code"          # D_meta = 0.0 (강체 / 강화유리)
    LAW_ACADEMIC = "law_academic"    # D_meta = 0.2 (고탄성 스프링)
    CONVERSATION = "conversation"    # D_meta = 0.5 (연성 플라스틱 / 젤리)
    POETRY_METAPHOR = "poetry_metaphor" # D_meta = 0.85 (점성 유체)
    FANTASY_SURREAL = "fantasy_surreal" # D_meta = 1.0 (무중력 기체)


@dataclass
class Polytope3D:
    """
    4D 위상 다면체 개념 개체 (Topological Polytope Entity)
    - position_x: [min_x, max_x]
    - position_y: [min_y, max_y]
    - height_h: [min_h, max_h] (추상화 차원 높이 축)
    - dihedral_angles: 개념 내부 가변 이면각 (Angular flexibility)
    - properties: 물리적/논리적 구속 속성 (Occupancy, Mass, Temp, etc.)
    """
    id: str
    name: str
    bounds_x: np.ndarray  # shape (2,) [x_min, x_max]
    bounds_y: np.ndarray  # shape (2,) [y_min, y_max]
    bounds_h: np.ndarray  # shape (2,) [h_min, h_max]
    dihedral_angles: np.ndarray  # shape (K,) 이면각 벡터 (0 ~ pi)
    occupancy: bool = True
    mass: float = 0.0
    rigid_properties: Dict[str, Any] = field(default_factory=dict)
    flexible_properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.bounds_x = np.array(self.bounds_x, dtype=np.float32)
        self.bounds_y = np.array(self.bounds_y, dtype=np.float32)
        self.bounds_h = np.array(self.bounds_h, dtype=np.float32)
        self.dihedral_angles = np.array(self.dihedral_angles, dtype=np.float32)

    def calculate_volume(self) -> float:
        """3D 체적 연산 Vol(V)"""
        dx = max(0.0, float(self.bounds_x[1] - self.bounds_x[0]))
        dy = max(0.0, float(self.bounds_y[1] - self.bounds_y[0]))
        dh = max(0.0, float(self.bounds_h[1] - self.bounds_h[0]))
        return dx * dy * dh

    def center(self) -> np.ndarray:
        """중심 좌표 (cx, cy, ch)"""
        return np.array([
            (self.bounds_x[0] + self.bounds_x[1]) * 0.5,
            (self.bounds_y[0] + self.bounds_y[1]) * 0.5,
            (self.bounds_h[0] + self.bounds_h[1]) * 0.5
        ], dtype=np.float32)

    def clone(self) -> 'Polytope3D':
        return Polytope3D(
            id=self.id,
            name=self.name,
            bounds_x=self.bounds_x.copy(),
            bounds_y=self.bounds_y.copy(),
            bounds_h=self.bounds_h.copy(),
            dihedral_angles=self.dihedral_angles.copy(),
            occupancy=self.occupancy,
            mass=self.mass,
            rigid_properties=dict(self.rigid_properties),
            flexible_properties=dict(self.flexible_properties)
        )


@dataclass
class VerificationResult:
    """위상적 인지 판별 최종 출력 규격"""
    is_valid: bool
    status: str
    initial_stress: float
    residual_stress: float
    yield_threshold: float
    conflict_details: Dict[str, Any]
    correction_trajectory: Dict[str, Any]
    relaxation_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_result": {
                "is_valid": self.is_valid,
                "status": self.status,
                "initial_stress": float(self.initial_stress),
                "residual_stress": float(self.residual_stress),
                "yield_threshold": float(self.yield_threshold),
                "stress_overload_ratio": f"{(self.residual_stress / max(1e-6, self.yield_threshold)) * 100:.1f}%",
                "conflict_details": self.conflict_details,
                "correction_trajectory": self.correction_trajectory,
                "relaxation_summary": self.relaxation_summary
            }
        }


class ContextPropertyEngine:
    """
    문맥 장르 지표 D_meta(C) 에 의거하여
    동적 물성치 (Yield Stress, Angular Inertia, Height Mobility)를 산출하는 수식 엔진.
    """
    def __init__(
        self,
        sigma_0: float = 1.0,
        alpha: float = 5.298,   # exp(5.298 * 0.85) approx 90 -> poetry yield stress ~ 80
        I_0: float = 1.0,
        beta: float = 0.95,
        mu_0: float = 0.1,
        gamma: float = 9.0
    ):
        self.sigma_0 = sigma_0
        self.alpha = alpha
        self.I_0 = I_0
        self.beta = beta
        self.mu_0 = mu_0
        self.gamma = gamma

    def get_d_meta(self, domain: Union[GenreDomain, str, float]) -> float:
        if isinstance(domain, (float, int)):
            return float(np.clip(domain, 0.0, 1.0))
        if isinstance(domain, GenreDomain):
            domain = domain.value
        domain_str = str(domain).lower()

        mapping = {
            GenreDomain.MATH_CODE.value: 0.0,
            "math": 0.0,
            "code": 0.0,
            GenreDomain.LAW_ACADEMIC.value: 0.2,
            "law": 0.2,
            "academic": 0.2,
            GenreDomain.CONVERSATION.value: 0.5,
            "chat": 0.5,
            GenreDomain.POETRY_METAPHOR.value: 0.85,
            "poetry": 0.85,
            "metaphor": 0.85,
            GenreDomain.FANTASY_SURREAL.value: 1.0,
            "fantasy": 1.0,
            "surreal": 1.0
        }
        return mapping.get(domain_str, 0.5)

    def compute_properties(self, domain: Union[GenreDomain, str, float]) -> Dict[str, float]:
        d_meta = self.get_d_meta(domain)

        # 1. 동적 항복 응력: sigma_yield(C) = sigma_0 * exp(alpha * D_meta)
        sigma_yield = self.sigma_0 * math.exp(self.alpha * d_meta)

        # 2. 내각 관성 모멘트: I_theta(C) = I_0 * (1 - beta * D_meta)
        I_theta = max(0.01, self.I_0 * (1.0 - self.beta * d_meta))

        # 3. 추상화 층위 전이율 (H축 이동성): mu_H(C) = mu_0 * (1 + gamma * D_meta)
        mu_h = self.mu_0 * (1.0 + self.gamma * d_meta)

        return {
            "d_meta": d_meta,
            "sigma_yield": float(sigma_yield),
            "I_theta": float(I_theta),
            "mu_h": float(mu_h)
        }


class TopologicalCognitiveVerifier:
    """
    위상적 응력 기반 모순 검증 알고리즘 엔진 (Topological Cognitive Verifier)

    4단계 인지 파이프라인:
    1. 구속 경계 투영 (Constraint Projection)
    2. 공간 침범 및 응력 텐서 추출 (Overlap & Stress Computation)
    3. 위상적 이완 한계 검증 (Topological Strain Evaluation & Relaxation)
    4. 모순 피드백 및 인과 궤적 출력 (Verification Output)
    """
    def __init__(
        self,
        elasticity_k: float = 0.2,
        max_relax_steps: int = 10,
        context_engine: Optional[ContextPropertyEngine] = None
    ):
        self.k = elasticity_k
        self.max_relax_steps = max_relax_steps
        self.context_engine = context_engine or ContextPropertyEngine()

    @staticmethod
    def calculate_3d_volume_intersection(poly_a: Polytope3D, poly_b: Polytope3D) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        두 Polytope간 3D 체적 교집합 Vol(A ∩ B) 및 중첩 영역 법선 벡터 n_overlap 계산
        Returns:
            (overlap_volume, normal_vector, overlap_bounds_h)
        """
        ox_min = max(poly_a.bounds_x[0], poly_b.bounds_x[0])
        ox_max = min(poly_a.bounds_x[1], poly_b.bounds_x[1])
        dx = max(0.0, float(ox_max - ox_min))

        oy_min = max(poly_a.bounds_y[0], poly_b.bounds_y[0])
        oy_max = min(poly_a.bounds_y[1], poly_b.bounds_y[1])
        dy = max(0.0, float(oy_max - oy_min))

        oh_min = max(poly_a.bounds_h[0], poly_b.bounds_h[0])
        oh_max = min(poly_a.bounds_h[1], poly_b.bounds_h[1])
        dh = max(0.0, float(oh_max - oh_min))

        vol_overlap = dx * dy * dh

        if vol_overlap <= 0.0:
            return 0.0, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

        # 법선 벡터: 두 중심점간 차이 벡터
        center_a = poly_a.center()
        center_b = poly_b.center()
        diff = center_b - center_a
        norm = np.linalg.norm(diff)

        if norm > 1e-6:
            normal_vec = diff / norm
        else:
            # 완벽히 겹쳤을 때 배타적/속성 충돌을 상정하여 H축 정반대 법선 설정 [0, 0, 1]
            normal_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return float(vol_overlap), normal_vec.astype(np.float32), np.array([oh_min, oh_max], dtype=np.float32)

    def project_statement_to_polytopes(self, statement_data: Dict[str, Any]) -> List[Polytope3D]:
        """
        Step 1: 구속 경계 투영 (Constraint Projection)
        입력 명제 구조체를 3D/4D 위상 다면체 개체 리스트로 변환합니다.
        """
        polytopes = []
        entities = statement_data.get("entities", [])

        for idx, item in enumerate(entities):
            p_id = item.get("id", f"poly_{idx}")
            p_name = item.get("name", f"Entity_{idx}")
            bx = np.array(item.get("bounds_x", [0.0, 10.0]), dtype=np.float32)
            by = np.array(item.get("bounds_y", [0.0, 10.0]), dtype=np.float32)
            bh = np.array(item.get("bounds_h", [0.0, 5.0]), dtype=np.float32)
            angles = np.array(item.get("dihedral_angles", [np.pi / 2, np.pi / 2]), dtype=np.float32)

            poly = Polytope3D(
                id=p_id,
                name=p_name,
                bounds_x=bx,
                bounds_y=by,
                bounds_h=bh,
                dihedral_angles=angles,
                occupancy=bool(item.get("occupancy", True)),
                mass=float(item.get("mass", 0.0)),
                rigid_properties=item.get("rigid_properties", {}),
                flexible_properties=item.get("flexible_properties", {})
            )
            polytopes.append(poly)

        return polytopes

    def verify_statement(
        self,
        statement_data: Dict[str, Any],
        genre_domain: Union[GenreDomain, str, float] = GenreDomain.CONVERSATION
    ) -> VerificationResult:
        """
        전체 위상적 응력 기반 인지 검증 알고리즘 구동
        """
        # 0. 문맥 장르별 동적 물성치 구성
        props = self.context_engine.compute_properties(genre_domain)
        sigma_yield = props["sigma_yield"]
        I_theta = props["I_theta"]
        mu_h = props["mu_h"]

        # 1. 구속 경계 투영 (Constraint Projection)
        polytopes = self.project_statement_to_polytopes(statement_data)
        if len(polytopes) < 2:
            return VerificationResult(
                is_valid=True,
                status="VALID_SINGLE_ENTITY",
                initial_stress=0.0,
                residual_stress=0.0,
                yield_threshold=sigma_yield,
                conflict_details={},
                correction_trajectory={},
                relaxation_summary={"mode": "No Interaction Required"}
            )

        poly_a, poly_b = polytopes[0], polytopes[1]

        # 2. 공간 충돌 및 응력 텐서 추출 (Overlap & Stress Computation)
        vol_overlap, normal_vec, _ = self.calculate_3d_volume_intersection(poly_a, poly_b)

        # 하드 리지드 속성 충돌 여부 감지 (예: Occupancy=False vs Occupancy=True)
        has_rigid_conflict = False
        rigid_conflict_reason = ""
        if poly_a.occupancy != poly_b.occupancy:
            has_rigid_conflict = True
            rigid_conflict_reason = f"Occupancy mismatch ({poly_a.name}.Occupancy={poly_a.occupancy} vs {poly_b.name}.Occupancy={poly_b.occupancy})"
        elif "temp_state" in poly_a.rigid_properties and "temp_state" in poly_b.rigid_properties:
            if poly_a.rigid_properties["temp_state"] != poly_b.rigid_properties["temp_state"]:
                has_rigid_conflict = True
                rigid_conflict_reason = f"Rigid property clash: temp_state ({poly_a.rigid_properties['temp_state']} vs {poly_b.rigid_properties['temp_state']})"

        # 초기 위상 응력: sigma = -k * Vol(A ∩ B) * n_overlap
        # 리지드 속성 충돌 시 k에 마찰 증폭률 곱함
        effective_k = self.k * (10.0 if has_rigid_conflict else 1.0)
        stress_vec = -1.0 * effective_k * vol_overlap * normal_vec
        initial_stress_mag = float(np.linalg.norm(stress_vec))

        if initial_stress_mag == 0.0:
            return VerificationResult(
                is_valid=True,
                status="PURE_VALID_NO_OVERLAP",
                initial_stress=0.0,
                residual_stress=0.0,
                yield_threshold=sigma_yield,
                conflict_details={},
                correction_trajectory={},
                relaxation_summary={"mode": "Spatial Separation"}
            )

        # 3. 위상적 이완 루프 (Topological Relaxation Loop)
        current_a = poly_a.clone()
        current_b = poly_b.clone()
        residual_stress = initial_stress_mag

        relaxation_history = []
        h_displacement_total = 0.0
        angle_deformations = []

        for step in range(self.max_relax_steps):
            if residual_stress <= sigma_yield:
                break

            # 이완 시도:
            # 1) 리지드 구속 충돌이 명확한 경우: 이면각 변형 마찰계수가 무한대로 발산하여 변형 정체 (Stagnation)
            if has_rigid_conflict and I_theta >= 0.5:
                # 변형 불가 저항 발생 (Hard Boundary Locked)
                delta_h = 0.5 * mu_h
                current_b.bounds_h += delta_h
                h_displacement_total += delta_h

                # 잔여 응력 재계산
                vol_rem, norm_rem, _ = self.calculate_3d_volume_intersection(current_a, current_b)
                residual_stress = float(np.linalg.norm(-1.0 * effective_k * vol_rem * norm_rem))
                relaxation_history.append({
                    "step": step + 1,
                    "mode": "Rigid Resistance Friction",
                    "delta_h": delta_h,
                    "residual_stress": residual_stress
                })
                if step > 2 and abs(relaxation_history[-1]["residual_stress"] - relaxation_history[-2]["residual_stress"]) < 0.1:
                    break
            else:
                # 2) 은유적/가변 속성의 경우: 이면각(Theta) 변형 및 H축 추상층 전이 흡수
                delta_theta = (residual_stress / max(0.01, I_theta)) * 0.02
                current_a.dihedral_angles = np.clip(current_a.dihedral_angles - delta_theta, 0.05, np.pi)

                delta_h = mu_h * (residual_stress * 0.1)
                current_a.bounds_h += delta_h * 0.5
                current_b.bounds_h += delta_h * 1.5  # 추상층 전이
                h_displacement_total += delta_h

                shape_fold_factor = float(np.sin(np.mean(current_a.dihedral_angles)))
                vol_rem, norm_rem, _ = self.calculate_3d_volume_intersection(current_a, current_b)
                vol_rem_folded = vol_rem * max(0.1, shape_fold_factor)

                residual_stress = float(np.linalg.norm(-1.0 * self.k * vol_rem_folded * norm_rem))
                angle_deformations.append(f"DihedralAngle_Deformed_-{delta_theta:.3f}rad")

                relaxation_history.append({
                    "step": step + 1,
                    "mode": "Metaphorical Abstraction Shift",
                    "delta_theta": delta_theta,
                    "delta_h": delta_h,
                    "residual_stress": residual_stress
                })

        # 4. 항복 응력 초과 여부에 따른 판정 (Yield Criterion)
        is_valid = residual_stress <= sigma_yield

        conflict_details = {
            "entity_A": f"{poly_a.name} (Occupancy={poly_a.occupancy})",
            "entity_B": f"{poly_b.name} (Occupancy={poly_b.occupancy})",
            "collision_zone_volume": vol_overlap,
            "rigid_conflict": has_rigid_conflict,
            "rigid_reason": rigid_conflict_reason if has_rigid_conflict else "None"
        }

        repulsion_vector = (-1.0 * normal_vec).tolist()
        if is_valid:
            resolution_guide = f"Valid under {genre_domain} context. Topological relaxation achieved."
            status = "VALID_RELAXED_METAPHOR" if h_displacement_total > 0 else "VALID"
        else:
            status = "HARD_CONTRADICTION"
            resolution_guide = (
                f"Release constraint on {poly_a.name} or remove volumetric overlap with {poly_b.name} "
                f"along repulsion trajectory direction {repulsion_vector}."
            )

        correction_trajectory = {
            "repulsion_vector": repulsion_vector,
            "resolution_guide": resolution_guide
        }

        relaxation_summary = {
            "max_steps": self.max_relax_steps,
            "actual_steps": len(relaxation_history),
            "h_axis_total_displacement": h_displacement_total,
            "deformed_angles_log": angle_deformations,
            "history": relaxation_history
        }

        return VerificationResult(
            is_valid=is_valid,
            status=status,
            initial_stress=initial_stress_mag,
            residual_stress=residual_stress,
            yield_threshold=sigma_yield,
            conflict_details=conflict_details,
            correction_trajectory=correction_trajectory,
            relaxation_summary=relaxation_summary
        )


if __name__ == "__main__":
    verifier = TopologicalCognitiveVerifier()

    case_vacuum = {
        "entities": [
            {
                "id": "A",
                "name": "Constraint_Vacuum",
                "bounds_x": [0, 10], "bounds_y": [0, 10], "bounds_h": [0, 5],
                "occupancy": False, "mass": 0.0
            },
            {
                "id": "B",
                "name": "Constraint_Mass",
                "bounds_x": [0, 10], "bounds_y": [0, 10], "bounds_h": [0, 5],
                "occupancy": True, "mass": 100.0
            }
        ]
    }
    res1 = verifier.verify_statement(case_vacuum, genre_domain=GenreDomain.MATH_CODE)
    print("--- Case 1 (Hard Contradiction) ---")
    print(res1.to_dict())

    case_cold_flame = {
        "entities": [
            {
                "id": "A",
                "name": "Flame",
                "bounds_x": [2, 6], "bounds_y": [2, 6], "bounds_h": [1.0, 2.0],
                "dihedral_angles": [1.57, 1.57],
                "flexible_properties": {"temp_state": "hot"}
            },
            {
                "id": "B",
                "name": "Cold",
                "bounds_x": [2, 6], "bounds_y": [2, 6], "bounds_h": [1.0, 2.0],
                "dihedral_angles": [1.57, 1.57],
                "flexible_properties": {"temp_state": "cold"}
            }
        ]
    }
    res2 = verifier.verify_statement(case_cold_flame, genre_domain=GenreDomain.POETRY_METAPHOR)
    print("\n--- Case 2 (Cold Flame Metaphor) ---")
    print(res2.to_dict())
