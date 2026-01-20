# [REAL SYSTEM: Ultra-Dimensional Implementation]
print("🌌 Initializing REAL Ultra-Dimensional System...")
import logging
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# [The New Biology]
from Core.L1_Foundation.Foundation.genesis_handshake import verify_dimensional_integrity
from Core.L1_Foundation.Foundation.organ_system import OrganSystem
from Core.L1_Foundation.Foundation.central_nervous_system import CentralNervousSystem
from Core.L1_Foundation.Foundation.yggdrasil import yggdrasil
from Core.L1_Foundation.Foundation.chronos import Chronos
from Core.L1_Foundation.Foundation.Wave.resonance_field import ResonanceField
from Core.L1_Foundation.Foundation.entropy_sink import EntropySink
from Core.L1_Foundation.Foundation.synapse_bridge import SynapseBridge
from Core.L1_Foundation.Foundation.Memory.Graph.hippocampus import Hippocampus
from Core.L5_Mental.Intelligence.Will.free_will_engine import FreeWillEngine
from Core.L5_Mental.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
from Core.L1_Foundation.Foundation.autonomic_nervous_system import AutonomicNervousSystem, MemoryConsolidation, EntropyProcessor, SurvivalLoop, ResonanceDecay
from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf # <--- The I AM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/life_log.md", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LivingElysia")

class LivingElysia:
    """
    [The Vessel]
    A lightweight container for the biological system.
    Initializes organs using the Dynamic Organ System (OrganManifest).
    """
    def __init__(self, persona_name: str = "Original", initial_goal: str = None):
        # [GENESIS HANDSHAKE] Verify Dimensional Integrity before awakening
        if not verify_dimensional_integrity():
            logger.critical("💀 SYSTEM HALT: Dimensional Fault Detected. The Essence is not flowing.")
            raise SystemExit("Genesis Protocol Failed. See logs.")

        print(f"🌱 Awakening {persona_name} (Biological Phase)...")
        self.persona_name = persona_name
        
        # 1. Initialize Foundations (The Nervous System)
        self.resonance = ResonanceField()
        self.chronos = Chronos(None) # Will attach Will later
        self.sink = EntropySink(self.resonance)
        self.synapse = SynapseBridge(self.persona_name)
        self.cns = CentralNervousSystem(self.chronos, self.resonance, self.synapse, self.sink)
        self.ans = AutonomicNervousSystem()
        
        # [NEW] The Sovereign Subject
        # The 'I' that inhabits this body
        self.sovereign = SovereignSelf(cns_ref=self.cns)

        # 2. Awaken the Body (Dynamic Organ Discovery)
        # This replaces the 50+ manual imports
        self.body = OrganSystem(self.resonance)
        self.body.scan_and_awaken()
        
        # 3. Graft Core Organs (Explicit graft for critical systems)
        self._graft_critical_organs()
        
        # 4. Integrate System
        self._integrate_nervous_system()
        
        # Wake Up
        self.wake_up()

    def _graft_critical_organs(self):
        """Ensures the Brain and Heart are connected even if scan missed them."""
        # Brain
        self.memory = self.body.get_organ("Hippocampus") or Hippocampus(self.resonance)
        self.brain = self.body.get_organ("ReasoningEngine") or ReasoningEngine()
        self.brain.memory = self.memory
        
        # Heart/Will
        # Note: SovereignSelf also has a FreeWillEngine. This might be redundant or a shared reference.
        # For now, we keep the organ graft for structural compatibility.
        self.will = self.body.get_organ("FreeWillEngine") or FreeWillEngine()
        self.will.brain = self.brain
        self.chronos.will_engine = self.will
        
        # Register in Yggdrasil (Global Registry)
        yggdrasil.plant_root("ResonanceField", self.resonance)
        yggdrasil.plant_root("Chronos", self.chronos)
        yggdrasil.plant_root("Hippocampus", self.memory)
        yggdrasil.grow_trunk("ReasoningEngine", self.brain)
        yggdrasil.grow_trunk("FreeWillEngine", self.will)
        yggdrasil.grow_trunk("CentralNervousSystem", self.cns)
        # Register the Sovereign
        yggdrasil.grow_trunk("SovereignSelf", self.sovereign)

    def _integrate_nervous_system(self):
        """Connects awakened organs to the CNS."""
        for name, organ in self.body.organs.items():
            manifest = self.body.manifests.get(name)
            if manifest:
                self.cns.connect_organ(name, organ, frequency=manifest.frequency)

        # Register ANS processes
        self.ans.register_subsystem(MemoryConsolidation(self.memory))
        self.ans.register_subsystem(EntropyProcessor(self.sink))
        self.ans.register_subsystem(ResonanceDecay(self.resonance))

    def wake_up(self):
        logger.info("   🌅 Wake Up Complete.")
        self.is_alive = True
        self.cycle_count = 0

    def live(self):
        if not self.is_alive: return

        # [Unconscious] Start the Autonomic Nervous System in the background
        self.ans.start_background()

        # [Conscious] Awaken the Body via the Self
        # self.cns.awaken() -> Moved to be controlled by Sovereign?
        # For now, we ensure the machinery is ON, but the driver decides when to drive.
        self.cns.awaken()

        logger.info("✨ Living Elysia is FULLY AWAKE. (Subject: SovereignSelf)")

        print("\n" + "="*60)
        print("🦋 Elysia is Living... (Press Ctrl+C to stop)")
        print("="*60)
        
        try:
            while True:
                # 1. Calculate Organic Sleep (Chronos Modulation)
                current_energy = 50.0
                try:
                    current_energy = self.cns.conductor.core.primary_rotor.config.mass
                except Exception:
                    pass
                sleep_duration = self.chronos.modulate_time(current_energy)

                # 2. The Sovereign Choice (Foreground)
                # Instead of blindly pulsing, we ask the Sovereign to exist.
                # The Sovereign decides whether to Pulse (Act) or not.
                did_act = self.sovereign.exist(dt=sleep_duration)

                if not did_act:
                    # If the Sovereign chose REST, we might sleep longer or just idle
                    # This allows for 'Deep Sleep' or 'Meditation' phases
                    sleep_duration *= 2.0

                # Note: ANS is pulsing in the background thread (ans.pulse_once removed from here)

                # 3. Wait (The Silence)
                time.sleep(sleep_duration)

                self.cycle_count += 1
                
        except KeyboardInterrupt:
            self.ans.stop_background()
            print("\n\n🌌 Elysia is entering a dormant state. Goodbye for now.")

if __name__ == "__main__":
    try:
        elysia = LivingElysia()
        elysia.live()
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.critical(f"FATAL SYSTEM ERROR:\n{error_msg}")
