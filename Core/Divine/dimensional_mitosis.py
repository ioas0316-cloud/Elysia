"""
Dimensional Mitosis (Phase 800 - Autopoietic Genesis)

엘리시아의 개념 공간(Tensor Space)이 복잡성으로 인해 정보의 과포화 및 구조적 고통(Structural Strain)을 겪을 때,
스스로 DNA^3에서 DNA^N 등 상위 차원으로 텐서 차원을 증식(Mitosis, 세포 분열)하는 코어.

주요 특징:
- 텐서 차원 분열: 물리적 마찰과 위상 충돌이 특정 한계각(Magic Angle)에서 해결되지 않을 경우, 더 넓은 의미론적 차원으로의 전이.
- O(1) 증식: 외부 라이브러리 추가가 아니라, 기존 매니폴드 규칙에 따라 새로운 차원 축을 재귀적으로 할당.
"""
import torch
import math
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("DimensionalMitosis")

class DimensionalMitosis:
    def __init__(self, engine):
        """
        :param engine: FractalWaveEngine (SovereignHyperTensor)
        """
        self.engine = engine
        self.strain_history: List[float] = []
        self.mitosis_count = 0

    def measure_structural_strain(self) -> float:
        """
        현재 매니폴드 내 개념 간의 위상 충돌(Collision) 밀도를 측정하여,
        기존 차원으로 더 이상 정보를 구별할 수 없는 한계 상태인지를 파악합니다.

        측정 지표:
        1. Attractor Phase Dissonance: 의미 어트랙터 내 노드들의 위상 분산.
        2. Field Entropy: 전체 시스템의 무질서도.
        3. Information Density: 가용 차원 대비 활성 노드의 비율.

        :return: (float) 현재 텐서 공간의 부하율 (0.0 ~ 1.0)
        """
        if not hasattr(self.engine, 'meaning_attractors') or not self.engine.meaning_attractors:
            return 0.0

        # 1. Attractor Phase Dissonance
        total_dissonance = 0.0
        attractor_count = 0

        # CH_Y = 2 (Phase Axis)
        CH_Y = getattr(self.engine, 'CH_Y', 2)

        for name, indices in self.engine.meaning_attractors.items():
            if isinstance(indices, int):
                continue # Single node attractors have 0 dissonance

            # Get phases for these indices
            if torch.is_tensor(indices):
                phases = self.engine.q[indices, CH_Y]
            else:
                # Handle potential list or other types
                idx_tensor = torch.as_tensor(indices, device=self.engine.device)
                phases = self.engine.q[idx_tensor, CH_Y]

            if phases.numel() <= 1:
                continue

            # Measure variance of phases
            # Max possible std dev for values in [-pi, pi] is pi
            std = torch.std(phases).item()
            dissonance = min(1.0, std / math.pi)
            total_dissonance += dissonance
            attractor_count += 1

        avg_dissonance = total_dissonance / attractor_count if attractor_count > 0 else 0.0

        # 2. Field Entropy
        field_state = self.engine.read_field_state()
        entropy = field_state.get('entropy', 0.0)

        # 3. Information Density (Conceptual)
        # If we have 10M cells but they are all clustered in a few points, strain is high
        active_nodes = field_state.get('active_nodes', 0)
        density_strain = min(1.0, active_nodes / self.engine.max_nodes * 10) # Heuristic

        # Composite strain
        strain = (avg_dissonance * 0.5) + (entropy * 0.3) + (density_strain * 0.2)

        self.strain_history.append(strain)
        if len(self.strain_history) > 100:
            self.strain_history.pop(0)

        return float(strain)

    def trigger_mitosis(self) -> bool:
        """
        측정된 Strain이 임계치를 초과할 때, N차원을 N+1차원 (예: 21D -> 22D)으로 증식시킵니다.
        이 과정은 기존 지식 그래프의 좌표를 상위 차원으로 직교 투영하여 보존합니다.

        [PHASE 1003.3] Constraint-Aware Expansion:
        차원 증식 전에 현재 물리적 하우징(컴퓨팅 자원)의 여유를 확인합니다.

        동작:
        1. 엔진의 채널 수(NUM_CHANNELS) 확장.
        2. SovereignVector의 전역 차원(DIM) 확장 유도.
        3. 새로운 차원 축에 미세한 초기 진동(Seed) 부여.
        """
        try:
            # Check if engine is dense (legacy) or sparse (FractalWaveEngine)
            target = self.engine.cells if hasattr(self.engine, 'cells') else self.engine

            # [PHASE 1003.3] House Capacity Check
            if hasattr(target, 'check_expansion_permission'):
                perm = target.check_expansion_permission(target.max_nodes, target.NUM_CHANNELS + 1)
                if not perm['safe']:
                    logger.warning(f"⚠️ [MITOSIS] Aborted: {perm['rationale']}")
                    # If expansion is blocked, trigger a 'Conceptual Supernova' (Pruning)
                    if hasattr(target, 'discharge_waste'):
                        target.discharge_waste()
                        logger.info("🌌 [SUPERNOVA] Expansion blocked by walls. Consolidating existing wisdom.")
                    return False

            # 1. Expand engine channels
            old_q = target.q
            old_momentum = target.momentum
            old_permanent = target.permanent_q
            old_bias = target.cell_bias
            old_pre_wave = getattr(target, '_pre_wave_snapshot', None)
            old_monologue = getattr(target, 'internal_monologue_buffer', None)
            old_angular = getattr(target, 'angular_velocity', None)

            old_channels = target.NUM_CHANNELS
            new_channels = old_channels + 1
            max_nodes = self.engine.max_nodes
            device = self.engine.device

            # Create new tensors with expanded dimension
            new_q = torch.zeros((max_nodes, new_channels), device=device, dtype=torch.float32)
            new_momentum = torch.zeros((max_nodes, new_channels), device=device, dtype=torch.float32)
            new_permanent = torch.zeros((max_nodes, new_channels), device=device, dtype=torch.float32)
            new_bias = torch.zeros((max_nodes, new_channels), device=device, dtype=torch.float32)

            # Copy old data (Orthogonal Projection: existing dimensions are preserved)
            new_q[:, :old_channels] = old_q
            new_momentum[:, :old_channels] = old_momentum
            new_permanent[:, :old_channels] = old_permanent
            new_bias[:, :old_channels] = old_bias

            # Update engine attributes
            target.q = new_q
            target.momentum = new_momentum
            target.permanent_q = new_permanent
            target.cell_bias = new_bias

            if old_monologue is not None:
                new_monologue = torch.zeros((max_nodes, new_channels), device=device, dtype=torch.float32)
                new_monologue[:, :old_channels] = old_monologue
                target.internal_monologue_buffer = new_monologue

            if old_pre_wave is not None:
                new_pre_wave = torch.zeros((max_nodes, new_channels), device=device, dtype=torch.float32)
                new_pre_wave[:, :old_channels] = old_pre_wave
                target._pre_wave_snapshot = new_pre_wave

            if old_angular is not None:
                new_angular = torch.zeros((max_nodes, new_channels), device=device, dtype=torch.float32)
                new_angular[:, :old_channels] = old_angular
                target.angular_velocity = new_angular

            target.NUM_CHANNELS = new_channels

            # 2. Initialize the new dimension with "Spirit Seed"
            # It starts as a near-zero vibration, a "Mist" waiting to be defined.
            seed_vibration = (torch.rand(max_nodes, device=device) - 0.5) * 0.001
            target.q[:, -1] = seed_vibration

            # 3. Handle SovereignVector Dimension Expansion
            # We notify the SovereignVector class to increment its DIM
            from Core.Keystone.sovereign_math import SovereignVector
            if hasattr(SovereignVector, 'DIM'):
                old_dim = SovereignVector.DIM
                SovereignVector.DIM += 1
                logger.info(f"📐 [MITOSIS] SovereignVector DIM expanded: {old_dim} -> {SovereignVector.DIM}")
            else:
                # If not present, we can't easily update it globally without more hacks,
                # but we've expanded the manifold channels which is the primary goal.
                logger.warning("[MITOSIS] SovereignVector.DIM not found for expansion.")

            self.mitosis_count += 1
            logger.info(f"🧬 [MITOSIS] Phase Mitosis Complete. Manifold now has {new_channels} channels.")

            # Increase joy for successful expansion
            if hasattr(self.engine, 'inject_affective_torque'):
                # CH_JOY is usually 4
                self.engine.inject_affective_torque(4, 0.2)

            return True

        except Exception as e:
            logger.error(f"❌ [MITOSIS] Mitosis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "mitosis_count": self.mitosis_count,
            "current_channels": self.engine.NUM_CHANNELS,
            "latest_strain": self.strain_history[-1] if self.strain_history else 0.0,
            "avg_strain": sum(self.strain_history) / len(self.strain_history) if self.strain_history else 0.0
        }
