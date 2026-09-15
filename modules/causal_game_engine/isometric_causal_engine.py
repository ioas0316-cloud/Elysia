r"""
Isometric Causal Engine module for Elysia Causal Game Engine with Scale Lens ($\mathcal{L}_{scale}$)
and 4-Constraint Elysian Intervention Framework.

Integrates the 2.5D Isometric Tilemap Adapter with Elysia's Causal Field, Engram dynamics,
Multi-Scale Lens ($\mathcal{L}_{scale} \in [0.0, 1.0]$), and Elysian Intervention Dynamics:
- Entropic Cost: Energy cost paid for materialization/destruction.
- Causal Rebound: Simulation system's restorative friction & mass panic/heresy side effects.
- Synchronization Rate: Anchor NPC / Holy Node alignment reducing intervention penalty.
- Causality Latency: Indirect catalyst intervention vs Direct miracle intervention.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel
from modules.causal_game_engine.isometric_adapter import IsometricTilemap, IsoTileConfig, IsoTileCell


@dataclass
class CognitiveLensState:
    """Represents the current cognitive lens focus of an NPC, Town, or Empire ecosystem."""
    current_lens: str = "Survival"  # "Survival", "Gathering", "Construction", "Thriving", "Empire_Expansion"
    tension_level: float = 0.0
    active_attractors: List[str] = field(default_factory=list)
    resource_reserves: Dict[str, float] = field(default_factory=lambda: {
        "food": 100.0,
        "wood": 100.0,
        "stone": 50.0,
        "gold": 20.0
    })


@dataclass
class MicroSimsState:
    """Individual NPC micro psychological state."""
    npc_id: str
    hunger: float = 0.0          # 0 (full) to 100 (starving)
    energy: float = 100.0        # 0 (exhausted) to 100 (well-rested)
    social_stress: float = 0.0   # 0 to 100
    happiness: float = 80.0        # 0 to 100
    sync_rate: float = 0.1       # Anchor synchronization rate with Elysia (0.0 to 1.0)
    is_anchor: bool = False      # Whether NPC acts as an Anchor / Constellation Agent


@dataclass
class InterventionResult:
    """Result of an Elysian Intervention attempt."""
    success: bool
    action_type: str
    cost_spent: float
    causal_rebound_friction: float
    side_effects: List[str]
    message: str


class ElysianInterventionEngine:
    """
    4-Constraint Intervention Engine for Elysia.

    Constraints:
    1. Entropic Cost: Energy required for intervention.
    2. Causal Rebound: Physical & psychological backlash.
    3. Synchronization Rate: Discount gained via Anchor NPCs / Holy Nodes.
    4. Causality Latency: Direct miracles vs Indirect catalysts.
    """

    def __init__(self, initial_energy: float = 500.0):
        self.causal_energy: float = float(initial_energy)

    def calculate_cost_and_rebound(
        self,
        action_type: str,
        target_sync_rate: float,
        scale_lens: float,
        is_indirect: bool = False
    ) -> Tuple[float, float]:
        """Calculates entropic cost and causal rebound friction for an intervention."""
        base_costs = {
            "SPAWN_BUILDING": 100.0,
            "DESTROY_TILE": 150.0,
            "SPAWN_RESOURCE": 50.0,
            "MODIFY_CLIMATE": 30.0,
            "INJECT_IDEA": 20.0,
            "TRIGGER_MIRACLE": 200.0
        }
        base_cost = base_costs.get(action_type, 50.0)

        if is_indirect:
            base_cost *= 0.1

        final_cost = float(base_cost * (1.0 - (target_sync_rate * 0.5)))

        direct_severity = 2.0 if not is_indirect else 0.4
        rebound_friction = float(direct_severity * (1.0 + scale_lens) * (1.0 - target_sync_rate * 0.3))

        return final_cost, rebound_friction

    def request_intervention(
        self,
        action_type: str,
        target_sync_rate: float = 0.1,
        scale_lens: float = 0.5,
        is_indirect: bool = False
    ) -> InterventionResult:
        """Processes an intervention request against Elysia's 4 constraints."""
        cost, rebound = self.calculate_cost_and_rebound(action_type, target_sync_rate, scale_lens, is_indirect)

        if self.causal_energy < cost:
            return InterventionResult(
                success=False,
                action_type=action_type,
                cost_spent=0.0,
                causal_rebound_friction=0.0,
                side_effects=[],
                message=f"Insufficient Causal Energy (Required: {cost:.1f}, Current: {self.causal_energy:.1f})"
            )

        self.causal_energy -= cost

        side_effects = []
        if rebound > 1.5:
            side_effects.append("NPC_MASS_PANIC")
            side_effects.append("HERESY_RELIGION_RISE")
        if not is_indirect and "DESTROY" in action_type:
            side_effects.append("EXISTENTIAL_DREAD")

        return InterventionResult(
            success=True,
            action_type=action_type,
            cost_spent=cost,
            causal_rebound_friction=rebound,
            side_effects=side_effects,
            message=f"Intervention executed successfully. Energy spent: {cost:.1f}, Rebound: {rebound:.2f}"
        )


class IsometricCausalEngine:
    """
    Continuous Causal Simulation Engine with Multi-Scale Lens ($\mathcal{L}_{scale}$)
    and 4-Constraint Elysian Intervention Engine.
    """

    def __init__(self, width: int = 20, height: int = 20, projection_config: Optional[IsoTileConfig] = None):
        self.width = width
        self.height = height
        self.tilemap = IsometricTilemap(width, height, projection_config)
        self.causal_field = CausalField(dimensions=2)
        self.lens_state = CognitiveLensState()
        self.intervention_engine = ElysianInterventionEngine(initial_energy=1000.0)

        self.scale_lens: float = 0.5

        self.micro_states: Dict[str, MicroSimsState] = {}

        self.macro_geopolitical_tension: float = 0.0
        self.macro_empire_node_potential: float = 0.0

        self.survival_engram = EngramAttractor(
            id="engram_survival",
            name="Survival & Hazard Avoidance Intent",
            position=np.array([width / 2.0, height / 2.0], dtype=np.float32),
            intensity=30.0,
            sigma=8.0,
            active=True,
            tier="macro",
            symbolic_intent="Survive environmental friction and restore baseline tension"
        )

        self.gathering_engram = EngramAttractor(
            id="engram_gathering",
            name="Resource Gathering Intent",
            position=np.array([width / 2.0, height / 2.0], dtype=np.float32),
            intensity=20.0,
            sigma=10.0,
            active=False,
            tier="meso",
            symbolic_intent="Harvest wood, food, and stone to satisfy growth threshold"
        )

        self.construction_engram = EngramAttractor(
            id="engram_construction",
            name="Town Expansion & Construction Intent",
            position=np.array([width / 2.0, height / 2.0], dtype=np.float32),
            intensity=25.0,
            sigma=12.0,
            active=False,
            tier="meso",
            symbolic_intent="Construct housing, granaries, and town infrastructure"
        )

        self.empire_engram = EngramAttractor(
            id="engram_empire",
            name="Macro Imperial Hegemony Intent",
            position=np.array([width / 2.0, height / 2.0], dtype=np.float32),
            intensity=40.0,
            sigma=15.0,
            active=False,
            tier="macro",
            symbolic_intent="Expand empire hegemony and geopolitics across global region"
        )

        self.causal_field.register_engram(self.survival_engram)
        self.causal_field.register_engram(self.gathering_engram)
        self.causal_field.register_engram(self.construction_engram)
        self.causal_field.register_engram(self.empire_engram)

        self.npc_voxels: Dict[str, InformationVoxel] = {}

        self.potential_matrix = np.zeros((width, height), dtype=np.float32)
        self.friction_matrix = np.ones((width, height), dtype=np.float32)

    def set_scale_lens(self, scale: float):
        """Sets the scale lens L_scale in [0.0, 1.0]."""
        self.scale_lens = max(0.0, min(1.0, float(scale)))

    def get_scale_category(self) -> str:
        """Returns the active scale category: 'MICRO_SIMS', 'MID_CITY', or 'MACRO_CIV'."""
        if self.scale_lens < 0.3:
            return "MICRO_SIMS"
        elif self.scale_lens > 0.7:
            return "MACRO_CIV"
        else:
            return "MID_CITY"

    def spawn_npc(self, npc_id: str, x: float, y: float, role: str = "villager", is_anchor: bool = False, sync_rate: float = 0.1):
        """Spawns an NPC as an InformationVoxel bound to the causal field and micro state."""
        voxel = InformationVoxel(
            id=npc_id,
            content=f"NPC_{role}",
            tensor=np.array([1.0, 0.0], dtype=np.float32),
            position=np.array([x, y], dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            mass=1.0
        )
        self.npc_voxels[npc_id] = voxel
        self.causal_field.add_voxel(voxel)
        self.micro_states[npc_id] = MicroSimsState(npc_id=npc_id, is_anchor=is_anchor, sync_rate=float(sync_rate))

    def apply_elysian_intervention(
        self,
        action_type: str,
        target_x: int,
        target_y: int,
        anchor_npc_id: Optional[str] = None,
        is_indirect: bool = False
    ) -> InterventionResult:
        """Performs an Elysian intervention subject to the 4 constraint system."""
        sync_rate = 0.1
        if anchor_npc_id and anchor_npc_id in self.micro_states:
            sync_rate = self.micro_states[anchor_npc_id].sync_rate

        res = self.intervention_engine.request_intervention(
            action_type=action_type,
            target_sync_rate=sync_rate,
            scale_lens=self.scale_lens,
            is_indirect=is_indirect
        )

        if res.success:
            if action_type == "SPAWN_BUILDING":
                self.tilemap.set_building(target_x, target_y, "town_hall")
            elif action_type == "DESTROY_TILE":
                self.tilemap.set_elevation(target_x, target_y, 0.0)
                self.tilemap.set_building(target_x, target_y, None)

            self.apply_environmental_friction(target_x, target_y, res.causal_rebound_friction)
            for ms in self.micro_states.values():
                ms.social_stress = min(100.0, ms.social_stress + res.causal_rebound_friction * 10.0)

        return res

    def apply_environmental_friction(self, x: int, y: int, friction_delta: float, elevation_delta: float = 0.0):
        """
        Applies environmental friction ($\Omega$) or physical strain onto a grid tile.
        Updates causal potential, elevation, and tile friction dynamically.
        """
        cell = self.tilemap.get_cell(x, y)
        if cell:
            cell.custom_friction += friction_delta
            cell.elevation += elevation_delta
            cell.causal_potential += abs(friction_delta) * 10.0
            self._update_tile_matrices()

    def update_resource(self, resource_type: str, delta: float):
        """Updates town resource reserves and triggers dynamic lens switching evaluation."""
        self.lens_state.resource_reserves[resource_type] = max(
            0.0, float(self.lens_state.resource_reserves.get(resource_type, 0.0) + delta)
        )
        self._evaluate_cognitive_lens_switch()

    def _evaluate_cognitive_lens_switch(self):
        """
        Evaluates cognitive lens switching based on environmental tension ($\Omega$),
        resource levels, and Macro geopolitical state.
        """
        food = self.lens_state.resource_reserves.get("food", 0.0)
        wood = self.lens_state.resource_reserves.get("wood", 0.0)
        stone = self.lens_state.resource_reserves.get("stone", 0.0)

        avg_tension = float(np.mean(self.potential_matrix))
        self.lens_state.tension_level = avg_tension

        previous_lens = self.lens_state.current_lens

        if food < 30.0 or avg_tension > 50.0:
            self.lens_state.current_lens = "Survival"
        elif wood < 50.0 or stone < 30.0:
            self.lens_state.current_lens = "Gathering"
        elif self.macro_geopolitical_tension > 40.0:
            self.lens_state.current_lens = "Empire_Expansion"
        else:
            self.lens_state.current_lens = "Construction"

        if previous_lens != self.lens_state.current_lens:
            self.causal_field.set_engram_active("engram_survival", self.lens_state.current_lens == "Survival")
            self.causal_field.set_engram_active("engram_gathering", self.lens_state.current_lens == "Gathering")
            self.causal_field.set_engram_active("engram_construction", self.lens_state.current_lens == "Construction")
            self.causal_field.set_engram_active("engram_empire", self.lens_state.current_lens == "Empire_Expansion")

    def _update_tile_matrices(self):
        """Recomputes potential and friction matrices across the isometric grid."""
        for x in range(self.width):
            for y in range(self.height):
                cell = self.tilemap.get_cell(x, y)
                if cell:
                    pos = np.array([float(x), float(y)], dtype=np.float32)
                    engram_grad = self.causal_field.calculate_engram_gradient(pos)
                    cell.causal_potential = float(np.linalg.norm(engram_grad)) + (cell.elevation * 2.0)
                    self.potential_matrix[x, y] = cell.causal_potential
                    self.friction_matrix[x, y] = self.tilemap.compute_cell_friction(x, y)

    def _aggregate_micro_to_macro(self):
        """Aggregation (Zoom-Out: Micro Sims -> Macro Civ): Aggregates NPC stress to macro geopolitical tension."""
        if not self.micro_states:
            return
        total_stress = sum(ms.social_stress + ms.hunger for ms in self.micro_states.values())
        avg_stress = total_stress / float(len(self.micro_states))
        self.macro_empire_node_potential = float(avg_stress * 2.0 + float(np.sum(self.potential_matrix)) * 0.05)

    def _decompose_macro_to_micro(self):
        """Decomposition (Zoom-In: Macro Civ -> Micro Sims): Decomposes macro tension into micro NPC stress."""
        if self.macro_geopolitical_tension > 0:
            stress_delta = self.macro_geopolitical_tension * 0.1
            for ms in self.micro_states.values():
                ms.social_stress = float(min(100.0, ms.social_stress + stress_delta))

    def step(self, delta_time: float = 0.1) -> Dict[str, Any]:
        """
        Advances the Multi-Scale Causal Engine by one time step.
        """
        self._update_tile_matrices()
        self._aggregate_micro_to_macro()
        self._decompose_macro_to_micro()
        self._evaluate_cognitive_lens_switch()

        npc_positions = {}
        for npc_id, voxel in self.npc_voxels.items():
            current_pos = voxel.position.copy()

            engram_force = self.causal_field.calculate_engram_gradient(current_pos)

            gx = int(np.clip(current_pos[0], 0, self.width - 1))
            gy = int(np.clip(current_pos[1], 0, self.height - 1))
            friction = self.friction_matrix[gx, gy]

            if np.isinf(friction):
                speed_scale = 0.0
            else:
                speed_scale = 1.0 / max(0.5, friction)

            velocity = engram_force * speed_scale * 0.5
            new_pos = current_pos + velocity * delta_time

            new_pos[0] = np.clip(new_pos[0], 0.0, float(self.width - 1))
            new_pos[1] = np.clip(new_pos[1], 0.0, float(self.height - 1))

            voxel.position = new_pos
            voxel.velocity = velocity

            ms = self.micro_states.get(npc_id)
            if ms:
                ms.hunger = float(min(100.0, ms.hunger + 0.2 * delta_time))
                ms.energy = float(max(0.0, ms.energy - 0.1 * delta_time))

            iso_p = self.tilemap.projection.grid_to_iso(new_pos[0], new_pos[1], elevation=0.0)
            npc_positions[npc_id] = {
                "grid_pos": [float(new_pos[0]), float(new_pos[1])],
                "iso_pos": [float(iso_p[0]), float(iso_p[1])],
                "micro_state": {
                    "hunger": float(ms.hunger) if ms else 0.0,
                    "energy": float(ms.energy) if ms else 100.0,
                    "social_stress": float(ms.social_stress) if ms else 0.0,
                    "sync_rate": float(ms.sync_rate) if ms else 0.1,
                    "is_anchor": bool(ms.is_anchor) if ms else False
                } if ms else {}
            }

        self.causal_field.step(delta_time)

        if self.lens_state.current_lens == "Survival":
            self.update_resource("food", -0.5 * delta_time)
        elif self.lens_state.current_lens == "Gathering":
            self.update_resource("food", 1.0 * delta_time)
            self.update_resource("wood", 1.5 * delta_time)
            self.update_resource("stone", 0.8 * delta_time)
        elif self.lens_state.current_lens == "Construction":
            self.update_resource("food", -0.2 * delta_time)
            self.update_resource("wood", -0.5 * delta_time)
            self.update_resource("stone", -0.3 * delta_time)
        elif self.lens_state.current_lens == "Empire_Expansion":
            self.update_resource("gold", 2.0 * delta_time)

        return {
            "scale_lens": float(self.scale_lens),
            "scale_category": str(self.get_scale_category()),
            "cognitive_lens": str(self.lens_state.current_lens),
            "tension_level": float(self.lens_state.tension_level),
            "macro_geopolitical_tension": float(self.macro_geopolitical_tension),
            "macro_empire_node_potential": float(self.macro_empire_node_potential),
            "elysian_causal_energy": float(self.intervention_engine.causal_energy),
            "resource_reserves": {k: float(v) for k, v in self.lens_state.resource_reserves.items()},
            "npc_positions": npc_positions
        }
