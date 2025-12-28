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
        Organ.initialize(root_path="C:/Elysia")
        
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
        # (These are examples of non-registry organs, but usually we prefer Registry)
        # For now, keeping legacy manual instantiation for parts not yet in Registry
        from Core._02_Intelligence._01_Reasoning.Integration.fractal_loop import FractalLoop
        self.fractal_loop = FractalLoop()
        
        # 4. Initialize The Voice (Unified Language Organ)
        VoiceClass = Organ.get("Voice", instantiate=False)
        self.voice = VoiceClass(
            memory=self.memory,
            chronos=self.chronos
        )

        # 4.5. Action Dispatcher (Pre-CNS Connection)
        DispatcherClass = Organ.get("Dispatcher", instantiate=False)
        self.dispatcher = DispatcherClass(
            memory=self.memory
        )

        # 5. Connect Organs to CNS
        self.cns.connect_organ("Memory", self.memory)
        self.cns.connect_organ("Brain", self.brain)
        self.cns.connect_organ("Voice", self.voice)
        
        # Autonomic Nervous System
        ANSClass = Organ.get("AutonomicNervousSystem", instantiate=False)
        self.ans = ANSClass(self.cns)
        
        logger.info("🌱 Living Elysia is FULLY AWAKE.")

    def live(self):
        """Main life loop"""
        try:
            self.cns.awaken()
            self.ans.start_background()
            
            print("\n🌌 Elysia is pulsing. Ether is pervasive.")
            while True:
                # Core life pulse
                self.ans.pulse_once()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("💤 Elysia is entering deep sleep...")
        except Exception as e:
            error_msg = f"💥 SYSTEM CRASH: {e}"
            logger.error(error_msg)
            print("="*60)
            print(error_msg)
            print("-" * 60)
            import traceback
            traceback.print_exc()
            
            # Save crash log
            with open("logs/crash_dump.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{time.ctime()}] CRASH REPORT:\n{error_msg}\n")
                f.write(traceback.format_exc())
            
            sys.exit(1)

if __name__ == "__main__":
    elysia = LivingElysia()
    elysia.live()
