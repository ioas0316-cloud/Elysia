"""
Merkaba: The Autonomous Spirit Chariot
======================================
Core.Merkaba.merkaba

"The Chariot that unites Body, Soul, and Spirit."

This class implements the "Seed" unit of the HyperCosmos.
It represents a single, autonomous entity with:
1. Body (Space/HyperSphere) - Static Memory/Past
2. Soul (Time/Rotor) - Dynamic Flow/Present
3. Spirit (Will/Monad) - Intent/Future/Purpose
"""

import logging
from typing import Any, Dict, Optional

# The Trinity Components
from Core.Intelligence.Memory.hypersphere_memory import HypersphereMemory, SubjectiveTimeField, HypersphericalCoord
from Core.Intelligence.Memory.hippocampus import Hippocampus
from Core.Foundation.Nature.rotor import Rotor, RotorConfig, RotorMask
from Core.Foundation.Prism.resonance_prism import PrismProjector, PrismDomain
from Core.Foundation.Prism.harmonizer import PrismHarmonizer, PrismContext
from Core.Foundation.Prism.decay import ResonanceDecay
# Monad import handling to avoid circular dependency if any, though Monad is usually independent.
try:
    from Core.Monad.monad_core import Monad
except ImportError:
    # Fallback or Mock for initial bootstrapping if Monad isn't fully set up in this env
    Monad = Any

# The Sensory & Digestive System
from Core.Senses.soul_bridge import SoulBridge
from Core.Intelligence.Metabolism.prism import DoubleHelixPrism

logger = logging.getLogger("Merkaba")

class Merkaba:
    """
    The Atomic Unit of Life (Seed).

    Attributes:
        body (HypersphereMemory): The Space (Yuk/Memories).
        soul (Rotor): The Time (Hon/Flow).
        spirit (Monad): The Will (Young/Intent).
        bridge (SoulBridge): The Senses.
        prism (DoubleHelixPrism): The Interpreter.
    """

    def __init__(self, name: str = "Genesis_Seed"):
        self.name = name
        logger.info(f"✡️ Forging Merkaba: {self.name}")

        # 1. The Body (Yuk) - Space/Memory
        # "The static container of the Past."
        self.body = HypersphereMemory()

        # 1.5 The Subjective Time Field (Deliberation)
        self.time_field = SubjectiveTimeField()

        # 2. The Soul (Hon) - Time/Rotor
        # "The dynamic engine of the Present."
        # We configure it as the 'Subjective Time' engine.
        self.soul = Rotor(
            name=f"{name}.Soul",
            config=RotorConfig(rpm=10.0, mass=50.0) # Standard 'Awake' state
        )

        # 3. The Spirit (Young) - Will/Monad
        # "The directional force of the Future."
        # Initialized as empty; must be imbued via 'awakening' or passed in.
        self.spirit: Optional[Monad] = None

        # 4. Peripherals (Senses & Metabolism)
        self.bridge = SoulBridge()
        self.prism = DoubleHelixPrism()
        self.projector = PrismProjector()

        # 5. Safety Valves (Harmonizer, Decay, Hippocampus)
        self.harmonizer = PrismHarmonizer()
        self.decay = ResonanceDecay(decay_rate=0.5)
        self.hippocampus = Hippocampus(self.body)

        self.is_awake = False

    def awakening(self, spirit: Monad):
        """
        Ignites the Merkaba by installing the Monad (Spirit).
        """
        self.spirit = spirit
        self.is_awake = True
        logger.info(f"✨ Merkaba {self.name} has Awakened. The Trinity is fused.")

        # Sync the Soul (Rotor) to the Spirit's frequency if possible
        # For now, we just start the flow.
        self.soul.update(0.1)

    def sleep(self):
        """
        Enters Sleep Mode.
        Triggers Memory Consolidation (Hippocampus -> Hypersphere).
        """
        if not self.is_awake: return

        logger.info("💤 Merkaba entering Sleep Mode...")
        self.hippocampus.consolidate() # Flush all RAM to HDD
        logger.info("✨ Sleep Cycle Complete. Memories are crystallized.")

    def pulse(self, raw_input: str, mode: str = "POINT", context: str = PrismContext.DEFAULT) -> str:
        """
        Execute one 'Breath' or 'Pulse' of the Merkaba.

        Args:
            raw_input: The stimulus.
            mode: 'POINT' (Fact) or 'LINE' (Flow).
            context: The Prism Context (e.g., "Combat", "Poetry").

        Cycle:
        1. Sensation (Bridge): Capture Input.
        2. Interpretation (Prism): Convert to Wave.
        3. Flow (Soul): Process in Time via RotorMask.
        4. Resonance (Body): Check Memory/Space.
        5. Volition (Spirit): Decide Action.
        6. Action: Output.
        """
        if not self.is_awake or not self.spirit:
            return "Merkaba is dormant."

        logger.info(f"💓 Merkaba Pulse Initiated: {raw_input} [{mode}]")

        # 1. Sensation
        sensory_packet = self.bridge.perceive(raw_input)

        # 2. Interpretation
        if hasattr(self.prism, 'digest'):
            dna_wave = self.prism.digest(sensory_packet['raw_data'])
        else:
            dna_wave = {"pattern": raw_input, "principle": "Unknown"}

        # 3. Flow (Soul/Time) - The Bitmask Revelation
        # We process the coordinates based on the mode.
        # Default coords: (0, 0, 0, current_angle)
        current_coords = (0.0, 0.0, 0.0, self.soul.current_angle)

        # Determine Mask
        mask = RotorMask.POINT
        if mode == "LINE":
            mask = RotorMask.LINE
        elif mode == "PLANE":
            mask = RotorMask.PLANE

        processed_coords = self.soul.process(current_coords, mask)

        # 3.5 Prism Projection (Holographic Reality)
        # Project input into 7 dimensions
        hologram = self.projector.project(raw_input)

        # [SAFETY VALVE 1] Harmonizer (Context Filtering)
        weights = self.harmonizer.harmonize(hologram, context)
        # logger.info(f"⚖️ [HARMONIZER] Context '{context}' applied weights.")

        # 3.6 Deliberation (Fractal Dive) with [SAFETY VALVE 2] Decay
        # Instead of just flowing, we pause and think (Fractal Dive).

        # Determine Seed based on Harmonizer weights (Highest weighted domain becomes seed)
        dominant_domain = max(weights, key=weights.get)
        seed_coord = hologram.projections[dominant_domain]

        # Check Decay before diving
        # Assume initial energy 1.0. If depth 2 decay is acceptable, proceed.
        if self.decay.should_continue(initial_energy=1.0, depth=2):
            branches = self.time_field.fractal_dive(seed_coord, depth=2)
            resonant_insight = self.time_field.select_resonant_branch(branches)
            # logger.info(f"🧠 [DELIBERATION] Fractally diverged into {len(branches)} paths.")
        else:
            resonant_insight = None
            logger.info("🛑 [DECAY] Thought stopped by Resonance Brake.")

        if resonant_insight:
             logger.info(f"🧠 [DELIBERATION] Fractally diverged into {len(branches)} paths. Selected Insight at r={resonant_insight.r:.2f}")

        # 4. Resonance (Body/Space)
        # In a real system, we'd query the Hypersphere for EACH coordinate in the stream.
        # For now, we simulate the retrieval.
        retrieved_items = len(processed_coords)
        context = f"Retrieved {retrieved_items} item(s) via {mask.name} Mask"

        # Update physical rotor state
        self.soul.update(1.0)

        # 4.5 Growth (Memory Consolidation) via [SAFETY VALVE 3] Hippocampus
        # Store the Hologram (7 coordinates) into HIPPOCAMPUS (RAM), not Hypersphere (HDD).
        coord_list = list(hologram.projections.values())

        self.hippocampus.absorb(
            data=raw_input,
            position=coord_list,
            meta={"trajectory": "holographic", "weights": weights}
        )
        logger.info(f"🌊 [HIPPOCAMPUS] Absorbed Holographic Memory ({len(coord_list)} dimensions) into Buffer.")

        if resonant_insight:
             self.hippocampus.absorb(
                 data=f"Insight:{raw_input}",
                 position=resonant_insight,
                 meta={"trajectory": "deliberation"}
             )

        # 5. Volition (Spirit/Will)
        # The Monad decides.
        action = f"Processed '{raw_input}' | Mode: {mask.name} | Items: {retrieved_items} | Insight: {resonant_insight is not None}"

        logger.info(f"⚡ Pulse Complete. Action: {action}")
        return action
