"""
human_agency_engine.py
======================
Elysia Causal Engine - Human Agency & Landau-Ginzburg Phase Collapse Mechanics

Implements:
1. Constellation prediction distribution P_pred vs. Hero choice distribution P_act.
2. Defiance weight S_defiance, Causal Entropy H_causal (KL divergence), and Wonder Index A_wonder.
3. Landau-Ginzburg potential field V(H), gradient, Hessian matrix, and phase collapse condition det(Hessian) <= 0.
4. Stochastic Differential Equation (SDE) trajectory simulation using Euler-Maruyama.
5. Dynamic Bifurcation Path determination:
   - Path A: Angelic Ascension
   - Path B: Demonic Inversion
   - Path C: Human Boundary Expansion
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class TransitionPath(Enum):
    PATH_A_ASCENSION = "Path A: Angelic Transmutation / Ascension (천사적 승격)"
    PATH_B_INVERSION = "Path B: Demonic Decadence / Inversion (악마적 타락)"
    PATH_C_EXPANSION = "Path C: Human Resilience / Boundary Expansion (순수 주체성 유지)"


@dataclass
class AscensionEvent:
    hero_id: str
    target_tier: int = 2
    description: str = "인간 자아의 한계를 깨고 Tier 2 성좌로 승격하였습니다."


@dataclass
class InversionEvent:
    hero_id: str
    archetype: str = "Dark Attractor"
    description: str = "유혹의 외력에 인과가 굴복하여 광기의 아키타입으로 반전되었습니다."


@dataclass
class BoundaryExpansionEvent:
    hero_id: str
    expansion_factor: float = 1.5
    description: str = "성좌의 관측 한계를 넘어서는 불확실성의 신화로서 구조경계를 확장하였습니다."


@dataclass
class ChoiceOption:
    option_id: str
    utility: float
    alignment_vector: np.ndarray  # H_y_i


class HumanAgencyEvaluator:
    """
    성좌의 예측 확률 분포 대 인간의 실제 선택 및 인과 엔트로피 / 경이로움 지수 산출 엔진
    """

    def __init__(self, gamma_agency: float = 1.0, alpha_wonder: float = 2.0):
        self.gamma_agency = gamma_agency
        self.alpha_wonder = alpha_wonder

    def calculate_choice_probabilities(
        self,
        options: List[ChoiceOption],
        W_constellation: np.ndarray,
        T_predict: float = 1.0,
        T_agency: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        P_pred, P_act, S_defiance 산출

        Returns:
            P_pred: 상위 성좌 예측 확률 분포
            P_act: 인간의 실제 선택 확률 분포
            S_defiance: 반발 가중치 벡터
        """
        num_options = len(options)
        dot_products = np.array([np.dot(W_constellation, opt.alignment_vector) for opt in options], dtype=np.float64)

        # Softmax for P_pred
        exp_pred = np.exp(dot_products / max(T_predict, 1e-6))
        P_pred = exp_pred / np.sum(exp_pred)

        # Defiance Weight S_defiance(y_i) = 1.0 - P_pred(y_i)
        S_defiance = 1.0 - P_pred

        # Utilities and Softmax for P_act
        utilities = np.array([opt.utility for opt in options], dtype=np.float64)
        act_logits = (utilities + self.gamma_agency * S_defiance) / max(T_agency, 1e-6)
        exp_act = np.exp(act_logits - np.max(act_logits))  # Numerical stability
        P_act = exp_act / np.sum(exp_act)

        return P_pred, P_act, S_defiance

    def calculate_causal_entropy(self, P_act: np.ndarray, P_pred: np.ndarray) -> float:
        """
        인과 엔트로피 H_causal = D_KL(P_act || P_pred)
        """
        eps = 1e-12
        P_act_clamped = np.clip(P_act, eps, 1.0)
        P_pred_clamped = np.clip(P_pred, eps, 1.0)

        kl_div = np.sum(P_act_clamped * np.log(P_act_clamped / P_pred_clamped))
        return float(kl_div)

    def calculate_wonder_index(self, H_causal: float, xi_ordeal: float) -> float:
        """
        경이로움 지수 A_wonder = Sigmoid(alpha * H_causal * xi_ordeal)
        """
        val = self.alpha_wonder * H_causal * xi_ordeal
        wonder = 1.0 / (1.0 + math.exp(-val))
        return float(wonder)


class LandauGinzburgPotentialField:
    """
    란다우-긴즈부르크(Landau-Ginzburg) 상전이 포텐셜 연산 엔진
    V(H) = (a/2) * ||H||^2 + (b/4) * ||H||^4 - <E_trial, H>
    """

    def __init__(self, a: float = -2.0, b: float = 1.0):
        self.a = a
        self.b = b

    def potential(self, H: np.ndarray, E_trial: np.ndarray) -> float:
        norm_sq = float(np.sum(H ** 2))
        val = (self.a / 2.0) * norm_sq + (self.b / 4.0) * (norm_sq ** 2) - float(np.dot(E_trial, H))
        return val

    def gradient(self, H: np.ndarray, E_trial: np.ndarray) -> np.ndarray:
        """
        grad V(H) = (a + b * ||H||^2) * H - E_trial
        """
        norm_sq = float(np.sum(H ** 2))
        return (self.a + self.b * norm_sq) * H - E_trial

    def hessian(self, H: np.ndarray) -> np.ndarray:
        """
        Hessian matrix Nabla^2 V(H):
        d^2 V / dH_i dH_j = (a + b * ||H||^2) * delta_ij + 2 * b * H_i * H_j
        """
        d = len(H)
        norm_sq = float(np.sum(H ** 2))
        diag_term = (self.a + self.b * norm_sq) * np.eye(d)
        outer_term = 2.0 * self.b * np.outer(H, H)
        return diag_term + outer_term

    def is_phase_collapsed(self, H: np.ndarray) -> Tuple[bool, float]:
        """
        위상 붕괴 조건: det(Nabla^2 V(H)) <= 0
        """
        hess = self.hessian(H)
        det_val = float(np.linalg.det(hess))
        return (det_val <= 0.0), det_val


class HumanAgencyEngine:
    """
    통합 인간 노드 자율 선택 및 위상 붕괴 / SDE 수치 적분 엔진
    """

    def __init__(
        self,
        potential_field: Optional[LandauGinzburgPotentialField] = None,
        evaluator: Optional[HumanAgencyEvaluator] = None,
        sigma_fluctuation: float = 0.05,
        dt: float = 0.01,
        num_steps: int = 100
    ):
        self.potential_field = potential_field or LandauGinzburgPotentialField(a=-2.0, b=1.0)
        self.evaluator = evaluator or HumanAgencyEvaluator()
        self.sigma_fluctuation = sigma_fluctuation
        self.dt = dt
        self.num_steps = num_steps

    def simulate_sde_trajectory(
        self,
        H_init: np.ndarray,
        E_trial: np.ndarray,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Euler-Maruyama SDE 수치 적분
        dH_t = -grad V(H_t) dt + sigma * sqrt(dt) * eta_t
        """
        if seed is not None:
            np.random.seed(seed)

        H_current = H_init.copy().astype(np.float64)
        trajectory = [H_current.copy()]

        sqrt_dt = math.sqrt(self.dt)
        for _ in range(self.num_steps):
            grad_v = self.potential_field.gradient(H_current, E_trial)
            eta = np.random.normal(0.0, 1.0, size=H_current.shape)
            dH = -grad_v * self.dt + self.sigma_fluctuation * sqrt_dt * eta
            H_current += dH
            trajectory.append(H_current.copy())

        return H_current, trajectory

    def determine_bifurcation_path(
        self,
        H_final: np.ndarray,
        trajectory: List[np.ndarray],
        H_angel: np.ndarray,
        H_devil: np.ndarray,
        T_agency: float,
        hero_id: str
    ) -> Tuple[TransitionPath, Union[AscensionEvent, InversionEvent, BoundaryExpansionEvent]]:
        """
        SDE 적분 후 최종 가치관 좌표 및 궤적 분산을 기반으로 Path A/B/C 판정
        """
        norm_final = np.linalg.norm(H_final)
        norm_angel = np.linalg.norm(H_angel)
        norm_devil = np.linalg.norm(H_devil)

        eps = 1e-12
        cos_angel = float(np.dot(H_final, H_angel) / ((norm_final * norm_angel) + eps))
        cos_devil = float(np.dot(H_final, H_devil) / ((norm_final * norm_devil) + eps))

        traj_arr = np.array(trajectory)  # [Steps+1, d]
        traj_variance = float(np.mean(np.var(traj_arr, axis=0)))

        # Path A: Angelic Ascension
        if cos_angel >= 0.75 and T_agency >= 0.5:
            return TransitionPath.PATH_A_ASCENSION, AscensionEvent(hero_id=hero_id)

        # Path B: Demonic Inversion
        if cos_devil >= 0.75 and T_agency < 0.5:
            return TransitionPath.PATH_B_INVERSION, InversionEvent(hero_id=hero_id)

        # Path C: Human Boundary Expansion
        return TransitionPath.PATH_C_EXPANSION, BoundaryExpansionEvent(
            hero_id=hero_id,
            expansion_factor=1.0 + traj_variance
        )

    def process_trial_event(
        self,
        hero_id: str,
        H_current: np.ndarray,
        E_trial: np.ndarray,
        H_angel: np.ndarray,
        H_devil: np.ndarray,
        T_agency: float = 1.0,
        xi_ordeal: float = 1.0,
        options: Optional[List[ChoiceOption]] = None,
        W_constellation: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        시련/유혹 주입 -> 위상 붕괴 검사 -> SDE 적분 -> 전이 경로 판정 및 인과 엔트로피 산출
        """
        is_collapsed, det_val = self.potential_field.is_phase_collapsed(H_current)

        H_final, trajectory = self.simulate_sde_trajectory(H_current, E_trial, seed=seed)

        path, event = self.determine_bifurcation_path(
            H_final=H_final,
            trajectory=trajectory,
            H_angel=H_angel,
            H_devil=H_devil,
            T_agency=T_agency,
            hero_id=hero_id
        )

        result: Dict[str, Any] = {
            "hero_id": hero_id,
            "is_phase_collapsed": is_collapsed,
            "hessian_det": det_val,
            "H_init": H_current.tolist(),
            "H_final": H_final.tolist(),
            "bifurcation_path": path.value,
            "event": event,
            "trajectory_steps": len(trajectory)
        }

        # 선택지가 주어진 경우 엔트로피 및 경이로움 지수 연산
        if options and W_constellation is not None:
            P_pred, P_act, S_defiance = self.evaluator.calculate_choice_probabilities(
                options=options,
                W_constellation=W_constellation,
                T_agency=T_agency
            )
            H_causal = self.evaluator.calculate_causal_entropy(P_act, P_pred)
            A_wonder = self.evaluator.calculate_wonder_index(H_causal, xi_ordeal)

            result.update({
                "P_pred": P_pred.tolist(),
                "P_act": P_act.tolist(),
                "S_defiance": S_defiance.tolist(),
                "H_causal": H_causal,
                "A_wonder": A_wonder
            })

        return result
