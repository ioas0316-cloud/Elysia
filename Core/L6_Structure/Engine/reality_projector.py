"""
Reality Projector (       )
================================
"From Wave to Form, from Meaning to Matter."

This engine is the 'Coagulation' layer of Elysia.
It collapses 4D WaveDNA and Governance Dials into 3D Physical Geometry.
"""

import math
import logging
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# Add project root to sys.path for standalone testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.L6_Structure.Wave.wave_dna import WaveDNA
from Core.L1_Foundation.Foundation.universal_constants import GOLDEN_RATIO, FREQ_LOVE, FREQ_TRUTH
from Core.L6_Structure.Engine.governance_engine import GovernanceEngine
from Core.L4_Causality.World.Physics.sofa_optimizer import SofaPathOptimizer
from Core.L4_Causality.World.Physics.scalar_sensing import PPSSensor, ScalarField

logger = logging.getLogger("RealityProjector")

class CoagulationMode(Enum):
    SDF = "SDF"       # Volumetric/Abstract
    MESH = "MESH"     # Game/Structured (Wuthering Waves / Lost Ark style)

class RealityStyle(Enum):
    OPEN_WORLD = "OPEN_WORLD"  # Wuthering Waves style
    ISOMETRIC = "ISOMETRIC"    # Lost Ark style

class RealityProjector:
    def __init__(self, mode: CoagulationMode = CoagulationMode.SDF):
        self.mode = mode
        self.current_style = RealityStyle.OPEN_WORLD
        
        # Collective Aesthetic DNA (Human Consensus Beauty)
        self.aesthetic_dna = {
            "vibrany": 0.8,
            "contrast": 0.7,
            "complexity": 0.6,
            "softness": 0.4
        }
        
        # Cosmic Law Coefficients
        self.laws = {
            "optics_interfere": 1.0,  # Fermat's principle scaling
            "sacred_geo": GOLDEN_RATIO,
            "harmonic_resonance": FREQ_TRUTH,
            "sofa_optimization": True
        }
        
        self.sofa_engine = SofaPathOptimizer()
        self.pps_sensor = PPSSensor()
        self.elysia_field = ScalarField("Elysia", radius=5.0)

    def set_style(self, style: RealityStyle):
        """Sets the target 'Coagulation' style (e.g. OpenWorld vs Isometric)."""
        self.current_style = style
        logger.info(f"  Reality Style set to: {style.value}")

    def project(self, governance: GovernanceEngine) -> Dict[str, Any]:
        """
        Projects the current Governance state into 3D Reality Parameters.
        """
        # 1. Fetch Master Dials
        physics_dna = governance.physics_rotors["Density"].dna  # Simplified for prototype
        narrative_dna = governance.narrative_rotors["Emotion"].dna
        aesthetic_dna = governance.aesthetic_rotors["Light"].dna
        
        # 2. Volumetric Collapse (Mapping Dials to Physicality)
        # Using Principle-Driven Mapping
        
        # Terrain complexity driven by Physics.Entropy and Narrative.Conflict
        terrain_complexity = (physics_dna.physical * 0.7) + (narrative_dna.phenomenal * 0.3)
        
        # Atmosphere/Light driven by Aesthetic.Light and Cosmic Optics
        light_intensity = aesthetic_dna.phenomenal * self.laws["optics_interfere"]
        
        # 3. Apply Style-Specific Scaling
        rendering_config = self._apply_style_rules(terrain_complexity, light_intensity)
        
        # 4. Integrate Collective Aesthetics (The 'Pixiv' Pulse)
        rendering_config["post_processing"] = {
            "vibrant": self.aesthetic_dna["vibrany"] * aesthetic_dna.phenomenal,
            "bloom": self.aesthetic_dna["softness"] * (1.0 - physics_dna.structural),
            "lut": "FANTASY_VIBRANT" if self.current_style == RealityStyle.OPEN_WORLD else "GRITTY_EPIC"
        }
        
        return {
            "mode": self.mode.value,
            "style": self.current_style.value,
            "parameters": rendering_config,
            "geometry_seed": self._generate_geometry_seed(governance),
            "sofa_path": self.sofa_engine.get_optimal_pose(0.5), # Sample pose
            "interference": self.pps_sensor.sense_environment(self.elysia_field, [0,0,0], []), # Empty for proto
            "timestamp": governance.physics_rotors["Gravity"].config.rpm 
        }

    def _apply_style_rules(self, complexity: float, light: float) -> Dict[str, Any]:
        """Applies Game-specific layout rules (Lost Ark vs Wuthering Waves)."""
        if self.current_style == RealityStyle.ISOMETRIC:
            # Lost Ark Logic: Static Camera, High Detail in foreground
            return {
                "fov": 45.0,
                "camera_pitch": -45.0,
                "terrain_scale": 1.0 * complexity,
                "detail_density": 0.8,
                "skybox": "SKY_EPIC_ORANGE"
            }
        else:
            # Open World Logic: Dynamic Camera, Global Fog, Horizon Focus
            return {
                "fov": 75.0,
                "camera_pitch": 0.0,
                "terrain_scale": 5.0 * complexity,
                "detail_density": 0.4,
                "skybox": "SKY_WINDY_BLUE",
                "fog_density": 0.05 * (1.0 - light)
            }

    def _generate_geometry_seed(self, governance: GovernanceEngine) -> str:
        """Generates a structural seed based on Sacred Geometry and Dials."""
        # Phi-based scaling for structural beauty
        scale = governance.aesthetic_rotors["Dimension"].dna.structural * GOLDEN_RATIO
        return f"GEO_{int(scale * 1000)}_{self.current_style.value[:3]}"

if __name__ == "__main__":
    # Test Run
    engine = GovernanceEngine()
    projector = RealityProjector(CoagulationMode.MESH)
    
    print("  [REALITY PROJECTOR] Dreaming in Open World...")
    world_data = projector.project(engine)
    print(f"Data: {world_data}")
    
    print("\n   [REALITY PROJECTOR] Switching to Isometric (Lost Ark Style)...")
    projector.set_style(RealityStyle.ISOMETRIC)
    world_data = projector.project(engine)
    print(f"Data: {world_data}")
