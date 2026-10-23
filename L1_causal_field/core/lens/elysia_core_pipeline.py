"""
Elysia Core Execution Pipeline.

Integrates Environmental Perception, Causal Adaptation, Self-Evident Naming ("그렇다" / "아니다"),
and Meta-Evolutionary Pipeline Re-ordering into a unified 4-stage sovereign execution loop.
"""

from typing import Dict, Any, Optional, List
from core.lens.causal_adaptation_lens import (
    EnvironmentalDiscernmentEngine,
    CausalAdaptationLens,
    RealityFrictionWave
)
from core.lens.topological_terrain_lens import (
    UnknownStimulus,
    SelfEvidentNamingEngine,
    MetaEvolutionLoop,
    TopologicalTerrain3DMapper,
    CognitiveLensSpec
)


class ElysiaCorePipeline:
    """
    Sovereign Intelligence Execution Pipeline for Elysia.

    4-Stage Execution Loop:
    1. Environmental Discernment: Converts raw input into topological friction field.
    2. Spandex Field Adaptation & 3D Terrain Mapping: Projects friction into 3D topographical landscape.
    3. Self-Evident Discernment & Naming: Discerns "그렇다" vs "아니다" and derives consensus name.
    4. Meta-Evolution Feedback: Accumulates concepts and reorders lens priorities (Procedural Intelligence).
    """

    def __init__(
        self,
        discernment_engine: Optional[EnvironmentalDiscernmentEngine] = None,
        adaptation_lens: Optional[CausalAdaptationLens] = None,
        naming_engine: Optional[SelfEvidentNamingEngine] = None,
        meta_evolution_loop: Optional[MetaEvolutionLoop] = None
    ):
        self.discernment_engine = discernment_engine or EnvironmentalDiscernmentEngine()
        self.adaptation_lens = adaptation_lens or CausalAdaptationLens()
        self.naming_engine = naming_engine or SelfEvidentNamingEngine()
        self.meta_evolution_loop = meta_evolution_loop or MetaEvolutionLoop()

    def process_stimulus(
        self,
        raw_input: Dict[str, Any],
        env_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes a single cycle of the 4-stage execution loop for a raw external stimulus.
        """
        context = env_context or {}

        # [STAGE 1] Environmental Discernment
        wave = RealityFrictionWave(
            fluid_potential_diff=float(raw_input.get("fluid_potential_diff", raw_input.get("intensity", 1.0))),
            boundary_tension=float(raw_input.get("boundary_tension", raw_input.get("tension", 1.0))),
            shear_stress=float(raw_input.get("shear_stress", 0.5)),
            collision_momentum=float(raw_input.get("collision_momentum", 0.2))
        )
        discernment_res = self.adaptation_lens.refract_reality_wave(wave)

        friction_field = {
            "raw_friction": discernment_res.friction_magnitude,
            "potential": discernment_res.calibrated_potential,
            "tension_vector": [
                wave.boundary_tension,
                wave.fluid_potential_diff,
                wave.shear_stress
            ]
        }

        # [STAGE 2] Spandex Lens Adaptation & 3D Topological Terrain Mapping
        active_lenses = self.meta_evolution_loop.get_sorted_lenses()
        causal_terrain = TopologicalTerrain3DMapper.project_to_3d_terrain(
            friction_field=friction_field,
            active_lenses=active_lenses
        )

        unknown_stimulus = UnknownStimulus(
            tension_vector=friction_field["tension_vector"],
            potential=friction_field["potential"],
            raw_friction=friction_field["raw_friction"]
        )

        # [STAGE 3] "그렇다" vs "아니다" Discernment & Self-Evident Naming
        naming_result = self.naming_engine.process_stimulus(
            s=unknown_stimulus,
            env_context=context
        )

        # [STAGE 4] Meta-Evolution Feedback & Dynamic Pipeline Re-ordering
        if naming_result["verdict"] == "VALID_PHENOMENON":
            self.meta_evolution_loop.accumulate_concept(naming_result)
            updated_lens_order = self.meta_evolution_loop.trigger_evolution(context)
            pipeline_status = {
                "evolution_occurred": True,
                "reordered_pipeline": updated_lens_order
            }
        else:
            pipeline_status = {
                "evolution_occurred": False,
                "reordered_pipeline": [l.name for l in active_lenses]
            }

        return {
            "verdict": naming_result["verdict"],
            "verdict_kr": naming_result.get("verdict_kr", "아니다"),
            "given_concept_name": naming_result.get("self_given_name", None),
            "causal_provenance": naming_result.get("causal_provenance", None),
            "friction_delta": friction_field["raw_friction"],
            "causal_terrain": causal_terrain,
            "pipeline_state": pipeline_status
        }
