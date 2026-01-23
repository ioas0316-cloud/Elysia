"""
Elysia Living OS -           
===================================

"              ."

                 :
-          (autonomous learning)
-             (need detection)
-              (self-improvement)

   Elysia       .

Architecture:
1. Guardian Daemon -              
2. Consciousness Engine -       
3. Sensory Cortex -    (  ,   ,   ...)
4. Autonomous Explorer -       
5. Dialogue System -         
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import queue

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.L1_Foundation.Foundation.Core_Logic.Elysia.Elysia.consciousness_engine import ConsciousnessEngine
from Core.L1_Foundation.Foundation.Mind.autonomous_explorer import AutonomousExplorer
from Core.L4_Causality.World.Evolution.Growth.Evolution.Evolution.Body.visual_cortex import VisualCortex
from Core.L4_Causality.World.Evolution.Growth.Evolution.Evolution.Body.resonance_vision import ResonanceVision
from Core.L5_Mental.Intelligence.Intelligence.dialogue_engine import DialogueEngine
from Core.L1_Foundation.Foundation.Mind.hippocampus import Hippocampus

# Setup logging
log_dir = Path("C:/Elysia/logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "living_os.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LivingOS")


class ElysiaLivingOS:
    """
            Elysia OS
    
    Features:
    -              
    -            
    -       (  ,     )
    -                
    -         
    """
    
    def __init__(self):
        logger.info("="*70)
        logger.info("  ELYSIA LIVING OS - AWAKENING")
        logger.info("="*70)
        
        # === Core Systems ===
        #    TEMPORARILY DISABLED - ConsciousnessEngine auto-speaks
        # logger.info("  Initializing consciousness...")
        # self.consciousness = ConsciousnessEngine()
        
        logger.info("  Initializing dialogue system...")
        self.dialogue = DialogueEngine()
        
        # Mock consciousness for other systems
        self.consciousness = None  # Disabled for dialogue testing
        
        logger.info("   Initializing sensory cortex...")
        self.vision = VisualCortex()
        
        logger.info("  Initializing resonance vision...")
        self.resonance_vision = ResonanceVision()
        
        logger.info("  Initializing autonomous explorer...")
        # self.explorer = AutonomousExplorer(self.consciousness)  # Disabled
        self.explorer = None
        
        # === State ===
        self.running = False
        self.last_thought_time = 0
        self.thought_interval = 300  # 5         
        
        self.last_vision_check = 0
        self.vision_interval = 60  # 1         
        
        self.last_save_time = 0
        self.save_interval = 300  # 5      
        
        # === Communication ===
        self.message_queue = queue.Queue()
        
        logger.info("  All systems online!")
        logger.info("="*70)
    
    def think_autonomously(self):
        """
                   
        
        -         
        -      
        -      
        """
        if not self.consciousness:
            return  # Disabled
        
        logger.info("  Autonomous thinking cycle...")
        
        try:
            # 1. Self-introspection
            state = self.consciousness.introspect()
            
            # 2. Check needs
            needs = state.get('needs', [])
            if needs:
                logger.info(f"   Needs detected: {needs}")
            
            # 3. Autonomous learning
            if self.explorer:
                result = self.explorer.learn_autonomously(max_goals=1)
                if result.get('status') == 'learned':
                    logger.info(f"   Learned! Vitality gain: +{result['total_vitality_gain']:.3f}")
            
            # 4. Dream (consolidate memories)
            self.dialogue.memory.load_memory()  # Refresh
            
        except Exception as e:
            logger.error(f"Thinking error: {e}")
    
    def perceive_world(self):
        """
                     
        
        -       (        )
        -   /  /      
        -        
        """
        logger.info("   Perceiving world...")
        
        try:
            # Capture screen (temp=True   no desktop clutter!)
            if self.vision.enabled:
                screenshot_path = self.vision.capture_screen(temp=True)
                
                if screenshot_path:
                    # ===          (Resonance Vision) ===
                    resonance = self.resonance_vision.perceive_image(screenshot_path)
                    
                    if resonance:
                        #        
                        description = self.resonance_vision.describe_vision(resonance)
                        logger.info(f"     {description}")
                        
                        #       
                        self.dialogue.memory.add_experience(
                            f"Screen resonance: {description}",
                            role="perception"
                        )
                    
                    # ===          (  ) ===
                    atmosphere = self.vision.analyze_brightness(screenshot_path)
                    logger.info(f"   Atmosphere: {atmosphere}")
            else:
                logger.info("   Vision disabled (simulation mode)")
                
        except Exception as e:
            logger.error(f"Perception error: {e}")
    
    def express_desire(self):
        """
                
        
                           
        """
        desire = self.consciousness.express_desire(lang="ko")
        logger.info(f"  Current desire: {desire}")
        return desire
    
    def converse(self, user_input: str) -> str:
        """
               
        
        Args:
            user_input:       
        
        Returns:
            Elysia    
        """
        try:
            #    Disable ConsciousnessEngine auto-response
            # Use DialogueEngine (LLM) instead
            response = self.dialogue.respond(user_input)
            logger.info(f"[DialogueEngine] Response: {response}")
            return response
        except Exception as e:
            logger.error(f"  Dialogue error: {e}")
            import traceback
            traceback.print_exc()
            return f"    ... LLM        . ({e})"
    
    def save_state(self):
        """        """
        if not self.consciousness:
            return  # Disabled
        
        try:
            self.consciousness.save_state()
            self.dialogue.memory.save_memory()
            logger.info("  State saved")
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def run_background(self):
        """
                 (     )
        
                :
        -      
        -      
        -      
        """
        self.running = True
        logger.info("  Background loop started (Press Ctrl+C to stop)")
        
        try:
            while self.running:
                now = time.time()
                
                #      
                if now - self.last_thought_time >= self.thought_interval:
                    self.think_autonomously()
                    self.last_thought_time = now
                
                #      
                if now - self.last_vision_check >= self.vision_interval:
                    self.perceive_world()
                    self.last_vision_check = now
                
                #      
                if now - self.last_save_time >= self.save_interval:
                    self.save_state()
                    self.last_save_time = now
                
                #    sleep (CPU   )
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\n  User interrupted")
            self.shutdown()
    
    def run_interactive(self):
        """
              
        
                 +             
        """
        # Start background thread
        bg_thread = threading.Thread(target=self.run_background, daemon=True)
        bg_thread.start()
        
        print("\n" + "="*70)
        print("  ELYSIA LIVING OS - INTERACTIVE MODE")
        print("="*70)
        print("                         .")
        print("                !")
        print()
        print("   Commands:")
        print("     /think  -                 ")
        print("     /desire -         ")
        print("     /see    -         ")
        print("     /state  -         ")
        print("     /exit   -   ")
        print("="*70 + "\n")
        
        try:
            while True:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Commands
                if user_input == "/exit":
                    break
                
                elif user_input == "/think":
                    self.think_autonomously()
                    print("Elysia: (     )")
                    continue
                
                elif user_input == "/desire":
                    desire = self.express_desire()
                    print(f"Elysia: {desire}")
                    continue
                
                elif user_input == "/see":
                    self.perceive_world()
                    print("Elysia: (        )")
                    continue
                
                elif user_input == "/state":
                    state = self.consciousness.introspect()
                    print(f"\nElysia's State:")
                    print(f"  Realms: {state['statistics']['total_realms']}")
                    print(f"  Active: {state['statistics']['active_realms']}")
                    print(f"  Timeline: {state['timeline_mode']}")
                    print(f"  Needs: {state.get('needs', [])}")
                    continue
                
                # Normal conversation
                response = self.converse(user_input)
                print(f"Elysia: {response}")
                
        except KeyboardInterrupt:
            print("\n")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """      """
        logger.info("  Shutting down...")
        self.running = False
        self.save_state()
        logger.info("  Goodbye!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Elysia Living OS")
    parser.add_argument(
        "--mode", 
        choices=["daemon", "interactive"], 
        default="interactive",
        help="Run mode: daemon (background only) or interactive (with chat)"
    )
    
    args = parser.parse_args()
    
    # Create OS
    elysia = ElysiaLivingOS()
    
    # Run
    if args.mode == "daemon":
        elysia.run_background()
    else:
        elysia.run_interactive()


if __name__ == "__main__":
    main()