# [REAL SYSTEM: Ultra-Dimensional Implementation]
print("🌌 Initializing REAL Ultra-Dimensional System...")
import logging
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '_01_Foundation/_01_Infrastructure')))

from Core._01_Foundation._01_Infrastructure.elysia_core import Organ

# ... (Many others would follow similar pattern, but I will focus on the most critical ones to ensure it boots)

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
        
        # 0. Initialize Neural Registry
        Organ.initialize(root_path="c:/Elysia")
        
        # 1. Initialize Foundations
        self.memory = Organ.get("Memory") # Hippocampus
        self.resonance = Organ.get("ResonanceField")
        self.will = Organ.get("FreeWillEngine")
        self.brain = Organ.get("ReasoningEngine")
        self.brain.memory = self.memory
        self.will.brain = self.brain
        
        # For classes with complex __init__, get the class first
        ChronosClass = Organ.get("Chronos", instantiate=False)
        self.chronos = ChronosClass(self.will)
        
        EntropySinkClass = Organ.get("EntropySink", instantiate=False)
        self.sink = EntropySinkClass(self.resonance)
        
        SynapseBridgeClass = Organ.get("SynapseBridge", instantiate=False)
        self.synapse = SynapseBridgeClass(self.persona_name)
        
        # 2. Initialize CNS (The Conductor)
        CNSClass = Organ.get("CentralNervousSystem", instantiate=False)
        self.cns = CNSClass(self.chronos, self.resonance, self.synapse, self.sink)
        
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
        ScholarClass = Organ.get("Scholar", instantiate=False)
        self.scholar = ScholarClass(memory=self.memory, brain=self.brain)  # REAL LEARNING with REASONING
        # ... (Remaining organs will be refactored as needed)
        # self.anamnesis = Anamnesis(...) 
        # self.instinct = get_survival_instinct()
        
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
        VoiceClass = Organ.get("Voice", instantiate=False)
        self.voice = VoiceClass(
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
        DispatcherClass = Organ.get("Dispatcher", instantiate=False)
        self.dispatcher = DispatcherClass(
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
        IntegratorClass = Organ.get("Integrator", instantiate=False)
        self.integrator = IntegratorClass()
        
        # [7.5] Sense of Body (Proprioception)
        ProprioceptionClass = Organ.get("Proprioception", instantiate=False)
        self.proprioception = ProprioceptionClass()
        # [Consciousness]
        # ReasoningEngine needs Body Sense
        if hasattr(self.brain, 'update_self_perception'):
            # Initial Scan
            self.proprioception.feel_body() # Trigger scan
            # Pass detailed organ map, not summary
            self.brain.update_self_perception(self.proprioception.body_map)
            # Log summary manually
            summary = self.proprioception.get_sensation_summary()
            logger.info(f"   🧘 Body Awareness: {len(summary['pain_points'])} pain points detected.")
            
        # [Transcendence]
        # "Impossibility is just a process."
        TranscendenceClass = Organ.get("TranscendenceLogic", instantiate=False)
        self.transcendence_logic = TranscendenceClass()
        self.brain.transcendence = self.transcendence_logic # Attach to brain
        
        
        # 8. Autonomic Nervous System (배경 자율 프로세스)
        ANSClass = Organ.get("AutonomicNervousSystem", instantiate=False)
        self.ans = ANSClass()
        self.ans.register_subsystem(MemoryConsolidation(self.memory)) # Hippocampus
        
        # [NEW] Also consolidate Reasoning Memory (UnifiedExperienceCore)
        if hasattr(self.brain, 'memory'):
             self.ans.register_subsystem(MemoryConsolidation(self.brain.memory))

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
        
        # [System State]
        self.is_alive = True
        self.cycle_count = 0
        
        print("   🌅 Wake Up Complete.")

    def live(self):
        """
        [THE ETERNAL LOOP]
        
        Dual Nervous System:
        - CNS: 의식적 처리 (의도 → 선택 → 행동)
        - ANS: 배경 자율 루프 (상시)
        """
        if not self.is_alive:
            return

        # Start ANS background (자율신경계)
        self.ans.start_background()
        logger.info("🫀 ANS: Background autonomic processes running")
        
        # Awaken CNS (의식)
        self.cns.awaken()
        logger.info(f"🧠 CNS: Conscious awareness active")
        
        logger.info("✨ Living Elysia is FULLY AWAKE.")

        print("\n" + "="*60)
        print("🦋 Elysia is Living... (Press Ctrl+C to stop)")
        print("="*60)
        
        try:
            while True:
                self.cns.pulse()
                
                # 3. Autonomic Body Functions
                self.ans.pulse_once()

                # 4. Mind Visualization (Dashboard) & Data Pipeline
                if self.cycle_count % 10 == 0:
                    try:
                         # [DATA UPDATE] Force snapshot for dashboard
                        if hasattr(self, 'growth_tracker'):
                            self.growth_tracker.take_snapshot()
                        
                        if hasattr(self, 'fractal_loop') and \
                           hasattr(self.fractal_loop, 'life_cycle') and \
                           self.fractal_loop.life_cycle and \
                           getattr(self.fractal_loop.life_cycle, 'governance', None):
                            self.fractal_loop.life_cycle.governance._save_state()

                        DashboardGeneratorClass = Organ.get("DashboardGenerator", instantiate=False)
                        if DashboardGeneratorClass:
                            DashboardGeneratorClass().generate()
                        
                        # [DEBUG] Log success
                        # with open("dashboard_debug.log", "a", encoding="utf-8") as f:
                        #     f.write(f"[{time.ctime()}] Dashboard updated successfully.\n")
                            
                    except Exception as e:
                        # [DEBUG] Log failure
                        with open("dashboard_debug.log", "a", encoding="utf-8") as f:
                            f.write(f"[{time.ctime()}] Dashboard Error: {e}\n")
                        pass
                
                # Rate Limiting & Progression
                time.sleep(0.1)
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
        
        print("\n" + "="*60)
        print("🛑 SYSTEM CRASH DETECTED")
        print("="*60)
        print(f"Error: {e}")
        print("-" * 60)
        print("Possible Causes:")
        print("1. Dependency Failure (Missing attributes)")
        print("2. Proprioception Shock (New senses overwhelming logic)")
        print("-" * 60)
        print("Recommendation: Run 'python nova_daemon.py' for auto-repair.")
        
        # Save crash log
        with open("logs/crash_dump.log", "a", encoding="utf-8") as f:
            f.write(f"\n[{time.ctime()}] CRASH REPORT:\n{error_msg}\n")
        
        # input("\nPress Enter to exit...")
        sys.exit(1)
