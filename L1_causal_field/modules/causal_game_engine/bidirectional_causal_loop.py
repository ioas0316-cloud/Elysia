"""
bidirectional_causal_loop.py
============================
Elysia Causal Engine - Integrated Bidirectional Causal Loop Manager
Connects Multi-scale Cosmological Engine, Pearl's Do-Calculus SCM,
PyTorch Differentiable Neural SCM, and Causal Prompt Decoder into a unified cycle.
"""

from dataclasses import dataclass, field
import torch
from typing import Dict, List, Optional, Tuple, Any

from modules.causal_game_engine.alignment_field import (
    AlignmentVector,
    AlignmentType,
    HeroAlignmentState,
    AlignmentTensorField
)
from modules.causal_game_engine.multiscale_constellation import (
    MultiscaleCosmologicalEngine,
    MultiscaleConstellationNode,
    ConstellationTier,
    Tier3AffiliationType
)
from modules.causal_game_engine.do_calculus_engine import StructuralCausalModel
from modules.causal_game_engine.causal_scm_nn import DifferentiableSCM, CausalLossCalculator
from modules.causal_game_engine.causal_prompt_decoder import CausalPromptDecoder, DecodedPromptConstraints


class IntegratedBidirectionalCausalLoop:
    """
    외부 세계(유저 입력/외부 에이전트 행동)와 내부 텐서 필드 간의
    양방향 인과 순환 및 신경망 역전파 통합 루프 관리자.
    """

    def __init__(self):
        self.alignment_field = AlignmentTensorField()
        self.cosmology_engine = MultiscaleCosmologicalEngine(self.alignment_field)
        self.scm_graph = StructuralCausalModel()
        self.neural_scm = DifferentiableSCM(num_nodes=5)  # [Hero_X, Hero_Y, Const_Gravity, Event_Shock, SPI]
        self.loss_calculator = CausalLossCalculator()
        self.optimizer = torch.optim.Adam(self.neural_scm.parameters(), lr=0.01)
        self.prompt_decoder = CausalPromptDecoder()

        self._initialize_scm_graph()

    def _initialize_scm_graph(self):
        """기본 인과 그래프 엣지 구성"""
        # Event_Shock (Node 3) -> Hero_X (Node 0), Hero_Y (Node 1)
        self.scm_graph.add_causal_edge("event_shock", "hero_x", 0.4)
        self.scm_graph.add_causal_edge("event_shock", "hero_y", -0.3)
        # Constellation Gravity (Node 2) -> Hero_X (Node 0), Hero_Y (Node 1)
        self.scm_graph.add_causal_edge("const_gravity", "hero_x", 0.5)
        self.scm_graph.add_causal_edge("const_gravity", "hero_y", 0.5)
        # SPI (Node 4) -> Hero_X (Node 0)
        self.scm_graph.add_causal_edge("spi_stat", "hero_x", -0.2)

    def execute_bidirectional_step(
        self,
        hero_id: str,
        external_action_log: str,
        intervention_vector: Tuple[float, float],
        real_world_observed_outcome: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        1 full cycle of Bidirectional Causal Projection:
        1. Forward: External Action -> do(X) Graph Surgery -> Hero Alignment Drift
        2. Neural Backprop: Counterfactual Loss computation -> Neural SCM W_adj optimization
        3. Backward: Updated internal state -> Decoded Prompt Constraints for LLM Agent
        """
        hero = self.alignment_field.heroes.get(hero_id)
        if not hero:
            return {"success": False, "reason": f"Hero {hero_id} not found"}

        # 1. Forward Path: do(X) Graph Surgery Execution
        event_shock = AlignmentVector(x=intervention_vector[0], y=intervention_vector[1])
        surgered_scm = self.scm_graph.apply_do_intervention("event_shock", intervention_vector[0])

        # Internal Field Update: drift Hero Alignment
        drift_result = self.alignment_field.update_hero_alignment_drift(
            hero_id=hero_id,
            event_shock_vector=event_shock
        )

        # 2. Neural SCM Optimization (Loss Calculation & Backprop)
        dummy_x = torch.tensor([[
            hero.current_alignment.x,
            hero.current_alignment.y,
            1.5,  # Constellation gravity
            intervention_vector[0],
            hero.spi_stat / 100.0
        ]], dtype=torch.float32)

        do_mask = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        do_values = torch.tensor([0.0, 0.0, 0.0, intervention_vector[0], 0.0])

        pred_cf = self.neural_scm(dummy_x, do_mask=do_mask, do_values=do_values)

        if real_world_observed_outcome:
            target_real = torch.tensor([real_world_observed_outcome], dtype=torch.float32)
        else:
            target_real = dummy_x.clone()
            target_real[0, 0] += 0.1 * intervention_vector[0]

        self.optimizer.zero_grad()
        loss = self.loss_calculator(pred_cf, target_real, self.neural_scm.get_masked_adj())
        loss.backward()
        self.optimizer.step()

        # 3. Backward Path: Prompt Constraint Decoding
        player_const = next((c for c in self.alignment_field.constellations.values() if c.is_player), None)
        affinity_score = 0.0
        relation_state = "Friction"
        if player_const:
            affinity_score, rel_enum = self.alignment_field.calculate_constellation_affinity(
                player_const.constellation_id,
                hero.bound_constellation_id or "t2_pantheon_lg"
            )
            relation_state = rel_enum.value

        field_summary = {
            "hero_id": hero.hero_id,
            "alignment": {"x": hero.current_alignment.x, "y": hero.current_alignment.y},
            "relation_state": relation_state,
            "affinity_score": affinity_score,
            "is_deicide": hero.is_deicide,
            "spi_stat": hero.spi_stat
        }

        decoded_payload = self.prompt_decoder.build_agent_payload(field_summary, user_prompt=external_action_log)

        return {
            "success": True,
            "hero_id": hero_id,
            "drift_result": drift_result,
            "causal_loss": loss.item(),
            "surgered_scm_state": surgered_scm.get_current_state(),
            "agent_prompt_payload": decoded_payload
        }
