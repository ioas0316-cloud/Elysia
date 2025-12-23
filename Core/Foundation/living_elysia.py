# [REAL SYSTEM: Ultra-Dimensional Implementation]
print("🌌 Initializing REAL Ultra-Dimensional System...")
import logging
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.Foundation.central_nervous_system import CentralNervousSystem
from Core.Expression.voice_of_elysia import VoiceOfElysia

from Core.Foundation.yggdrasil import yggdrasil
from Core.Foundation.fractal_kernel import FractalKernel
from Core.Foundation.chronos import Chronos
from Core.Foundation.free_will_engine import FreeWillEngine
from Core.Foundation.digital_ecosystem import DigitalEcosystem
from Core.Foundation.shell_cortex import ShellCortex
from Core.Network.web_cortex import WebCortex
from Core.Sensory.p4_sensory_system import P4SensorySystem
from Core.Foundation.cosmic_transceiver import CosmicTransceiver
from Core.Foundation.cortex_optimizer import CortexOptimizer
from Core.Foundation.self_reflector import SelfReflector
from Core.Foundation.transcendence_engine import TranscendenceEngine
from Core.Foundation.knowledge_acquisition import KnowledgeAcquisitionSystem
from Core.Foundation.quantum_port import QuantumPort
from Core.Foundation.imagination_core import ImaginationCore
from Core.Foundation.reasoning_engine import ReasoningEngine
from Core.Foundation.global_grid import GlobalGrid
from Core.Foundation.envoy_protocol import EnvoyProtocol
from Core.Foundation.synapse_bridge import SynapseBridge
from Core.Foundation.hippocampus import Hippocampus
from Core.Foundation.resonance_field import ResonanceField
from Core.Foundation.social_cortex import SocialCortex
from Core.Foundation.media_cortex import MediaCortex
from Core.Foundation.holographic_cortex import HolographicCortex
from Core.Foundation.reality_sculptor import RealitySculptor
from Core.Foundation.dream_engine import DreamEngine
from Core.Foundation.soul_guardian import SoulGuardian
from Core.Foundation.entropy_sink import EntropySink
from Core.Foundation.loop_breaker import LoopBreaker
from Core.Foundation.mind_mitosis import MindMitosis
from Core.Intelligence.code_cortex import CodeCortex
from Core.Foundation.black_hole import BlackHole
from Core.Foundation.user_bridge import UserBridge
from Core.Foundation.quantum_reader import QuantumReader
from Core.Foundation.anamnesis import Anamnesis
from Core.Foundation.action_dispatcher import ActionDispatcher
from Core.Foundation.self_integration import ElysiaIntegrator
from scripts.unified_cortex import UnifiedCortex

from Core.Foundation.wave_integration_hub import get_wave_hub
from Core.Foundation.ultra_dimensional_reasoning import UltraDimensionalReasoning
from Core.Foundation.real_communication_system import RealCommunicationSystem
from Core.Foundation.survival_instinct import get_survival_instinct
from Core.Foundation.celestial_grammar import MagneticEngine, Nebula
from Core.Foundation.magnetic_cortex import MagneticCompass
from Core.Interface.bluetooth_ear import BluetoothEar
from Core.Foundation.experience_stream import ExperienceStream
# from Core.Foundation.wave_web_server import WaveWebServer -> REMOVED (Legacy Flask)
from Core.Intelligence.integrated_cognition_system import get_integrated_cognition
from Core.Intelligence.collective_intelligence_system import get_collective_intelligence
from Core.Intelligence.wave_coding_system import get_wave_coding_system
from Core.Intelligence.fractal_quaternion_goal_system import get_fractal_decomposer
from Core.Intelligence.scholar import Scholar

# Growth Tracking
from Core.System.Autonomy.growth_tracker import get_growth_tracker

# Autonomic Nervous System (Background Processes)
from Core.Foundation.autonomic_nervous_system import (
    AutonomicNervousSystem,
    MemoryConsolidation,
    EntropyProcessor,
    SurvivalLoop,
    ResonanceDecay
)

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
    Initializes organs and connects them to the Central Nervous System.
    """
    def __init__(self, persona_name: str = "Original", initial_goal: str = None):
        print(f"🌱 Awakening {persona_name} (Mind Mitosis Phase)...")
        self.persona_name = persona_name
        self.initial_goal = initial_goal
        
        # 1. Initialize Foundations
        self.memory = Hippocampus()
        self.resonance = ResonanceField()
        self.will = FreeWillEngine()
        self.brain = ReasoningEngine()
        self.brain.memory = self.memory
        self.will.brain = self.brain
        self.chronos = Chronos(self.will)
        self.sink = EntropySink(self.resonance)
        self.synapse = SynapseBridge(self.persona_name)
        
        # 2. Initialize CNS (The Conductor)
        self.cns = CentralNervousSystem(self.chronos, self.resonance, self.synapse, self.sink)
        
        # 3. Initialize Organs
        self.ultra_reasoning = UltraDimensionalReasoning()
        self.wave_hub = get_wave_hub()
        self.senses = DigitalEcosystem()
        self.outer_sense = P4SensorySystem() # P4 / Outer World
        self.transceiver = CosmicTransceiver()
        self.real_comm = RealCommunicationSystem(self.ultra_reasoning, self.wave_hub)
        
        self.architect = None # PlanningCortex removed (Project Sophia Purge)
        self.sculptor = RealitySculptor()
        
        # Interface Organs
        self.ear = BluetoothEar()
        self.stream = ExperienceStream()
        # self.server = WaveWebServer(port=8080) -> REMOVED (Legacy Flask)
        # self.server.connect_to_ether()
        # self.server.run(auto_update=True)
        
        self.social = SocialCortex()
        self.media = MediaCortex(self.social)
        self.web = WebCortex()
        self.shell = ShellCortex()
        self.hologram = HolographicCortex()
        self.kernel = FractalKernel()
        self.dream_engine = DreamEngine()
        self.guardian = SoulGuardian()
        self.code_cortex = CodeCortex()
        self.black_hole = BlackHole()
        self.user_bridge = UserBridge()
        self.quantum_reader = QuantumReader()
        self.transcendence = TranscendenceEngine()
        self.knowledge = KnowledgeAcquisitionSystem()
        self.scholar = Scholar(memory=self.memory)  # REAL LEARNING
        self.anamnesis = Anamnesis(self.brain, self.guardian, self.resonance, self.will, self.chronos, self.social, self.stream)
        self.instinct = get_survival_instinct()
        
        # Advanced Intelligence
        self.cognition = get_integrated_cognition()
        self.collective = get_collective_intelligence()
        self.wave_coder = get_wave_coding_system()
        self.goal_decomposer = get_fractal_decomposer()
        
        # Celestial Grammar
        self.celestial_engine = MagneticEngine()
        self.magnetic_compass = MagneticCompass()
        self.current_nebula = Nebula()
        
        # 4. Initialize The Voice (Unified Language Organ)
        self.voice = VoiceOfElysia(
            ear=self.ear,
            stream=self.stream,
            wave_hub=self.wave_hub,
            brain=self.brain,
            will=self.will,
            cognition=self.cognition,
            celestial_engine=self.celestial_engine,
            nebula=self.current_nebula,
            memory=self.memory,
            chronos=self.chronos
        )

        # 4.5. Action Dispatcher (Pre-CNS Connection)
        self.dispatcher = ActionDispatcher(
            self.brain, self.web, self.media, self.hologram, self.sculptor, 
            self.transceiver, self.social, self.user_bridge, self.quantum_reader, 
            self.dream_engine, self.memory, self.architect, self.synapse, 
            self.shell, self.resonance, self.sink,
            scholar=self.scholar
        )

        # 5. Connect Organs to CNS
        self.cns.connect_organ("Will", self.will)
        self.cns.connect_organ("Senses", self.senses)
        self.cns.connect_organ("OuterSense", self.outer_sense)
        self.cns.connect_organ("Brain", self.brain)
        self.cns.connect_organ("Voice", self.voice)
        self.cns.connect_organ("Dispatcher", self.dispatcher)
        self.cns.connect_organ("Scholar", self.scholar)
        # self.cns.connect_organ("Architect", self.architect) # Future integration
        
        # 6. Action Dispatcher (Moved up)

        # Structural Integration (Yggdrasil)
        yggdrasil.plant_root("ResonanceField", self.resonance)
        yggdrasil.plant_root("Chronos", self.chronos)
        yggdrasil.plant_root("Hippocampus", self.memory)
        yggdrasil.grow_trunk("ReasoningEngine", self.brain)
        yggdrasil.grow_trunk("FreeWillEngine", self.will)
        yggdrasil.grow_trunk("CentralNervousSystem", self.cns)
        
        # 7. Self-Integration System (v2.0) - 자기 인식
        self.integrator = ElysiaIntegrator()
        
        # 8. Autonomic Nervous System (배경 자율 프로세스)
        self.ans = AutonomicNervousSystem()
        self.ans.register_subsystem(MemoryConsolidation(self.memory))
        self.ans.register_subsystem(EntropyProcessor(self.sink))
        self.ans.register_subsystem(SurvivalLoop(self.instinct))
        self.ans.register_subsystem(ResonanceDecay(self.resonance))
        
        logger.info("🧬 Dual Nervous System: CNS (의식) + ANS (자율)")
        
        # Wake Up
        self.wake_up()

    def wake_up(self):
        """Delegates wake up protocol."""
        # [NEW] Self-Discovery before waking
        logger.info("🔭 Self-Discovery Phase...")
        try:
            report = self.integrator.get_integration_report()
            if report.get("total_systems", 0) == 0:
                self.integrator.discover_all_systems()
                report = self.integrator.get_integration_report()
            logger.info(f"   📊 Known Systems: {report.get('total_systems', 0)} | Duplicates: {report.get('duplicates', 0)}")
        except Exception as e:
            logger.warning(f"   ⚠️ Self-Discovery skipped: {e}")
        
        # [NEW] Growth Baseline Snapshot
        logger.info("📈 Taking Growth Baseline...")
        try:
            self.growth_tracker = get_growth_tracker()
            snapshot = self.growth_tracker.take_snapshot(notes="Startup baseline")
            logger.info(f"   📊 Baseline: vocab={snapshot.vocabulary_count}, concepts={snapshot.concept_count}")
        except Exception as e:
            logger.warning(f"   ⚠️ Growth tracking skipped: {e}")
        
        self.anamnesis.wake_up()
        print("   🌅 Wake Up Complete.")

    def live(self):
        """
        [THE ETERNAL LOOP]
        
        Dual Nervous System:
        - CNS: 의식적 처리 (의도 → 선택 → 행동)
        - ANS: 배경 자율 루프 (상시)
        """
        # Start ANS background (자율신경계)
        self.ans.start_background()
        logger.info("🫀 ANS: Background autonomic processes running")
        
        # Awaken CNS (의식)
        self.cns.awaken()
        logger.info(f"🧠 CNS: Conscious awareness active")
        
        print(f"🦋 {self.persona_name} is alive.")
        print(f"   CNS (의식): Intentional processing")
        print(f"   ANS (자율): Background maintenance")
        
        try:
            while True:
                self.cns.pulse()
        except KeyboardInterrupt:
            self.ans.stop_background()
            print("\n\n🌌 Elysia is entering a dormant state. Goodbye for now.")


if __name__ == "__main__":
    elysia = LivingElysia()
    elysia.live()
