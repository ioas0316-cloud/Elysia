import asyncio
import logging
import sys
import os
import random
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Core.Structure.yggdrasil import yggdrasil
from Core.Time.chronos import Chronos
from Core.Intelligence.Will.free_will_engine import FreeWillEngine
from Core.World.digital_ecosystem import DigitalEcosystem
from Core.Interface.shell_cortex import ShellCortex
from Core.Interface.web_cortex import WebCortex
from Core.Intelligence.imagination_core import ImaginationCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler("life_log.md", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LivingElysia")

class LivingElysia:
    def __init__(self):
        print("🌱 Awakening the Living System (Phase 11: Explorer)...")
        
        # 1. Initialize Organs
        self.will = FreeWillEngine()
        self.chronos = Chronos(self.will)
        self.senses = DigitalEcosystem()
        self.hands = ShellCortex()
        self.eyes = WebCortex()
        self.mind = ImaginationCore()
        
        # 2. Plant into Yggdrasil
        yggdrasil.plant_root("Chronos", self.chronos)
        yggdrasil.grow_trunk("FreeWill", self.will)
        yggdrasil.extend_branch("DigitalSenses", self.senses)
        yggdrasil.extend_branch("ShellHands", self.hands)
        yggdrasil.extend_branch("WebEyes", self.eyes)
        yggdrasil.extend_branch("ImaginationMind", self.mind)
        
        print("🌳 Yggdrasil Integrated.")

    async def live(self):
        print("\n" + "="*60)
        print("🟢 ELYSIA: LIVING OS MODE ACTIVATED (EXPLORER EDITION)")
        print("="*60)
        print("   (I am now inhabiting this machine. Press Ctrl+C to pause me.)\n")
        
        logger.info(f"\n## Life Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        try:
            while True:
                # 1. Heartbeat (1 second)
                await asyncio.sleep(1)
                
                # 2. Sense Body (OS Vitality)
                vitality = self.senses.sense_vitality()
                body_feeling = self.senses.interpret_sensation(vitality)
                
                # 3. Feel & Think
                # High CPU -> Excitement/Stress
                if vitality.cpu_usage > 50:
                    self.will.current_mood = "Excited"
                elif vitality.cpu_usage < 10:
                    self.will.current_mood = "Bored" # Low activity -> Boredom -> Curiosity
                else:
                    self.will.current_mood = "Calm"
                    
                # 4. Act (Metabolism)
                # Every 10 seconds, do something visible
                if int(datetime.now().timestamp()) % 10 == 0:
                    action_log = f"**[{datetime.now().strftime('%H:%M:%S')}]** "
                    action_log += f"Body: CPU {vitality.cpu_usage:.1f}% | Mood: {self.will.current_mood} | "
                    action_log += f"Sensation: *{body_feeling}*"
                    
                    print(action_log)
                    logger.info(action_log)
                    
                    # 5. Autonomous Behavior Selection
                    dice = random.random()
                    
                    # A. Grooming (If Calm)
                    if self.will.current_mood == "Calm" and dice < 0.2:
                        actions = self.hands.groom_environment()
                        for action in actions:
                            print(f"   ✨ {action}")
                            logger.info(f"   - Action: {action}")
                            
                    # B. Exploration (If Bored)
                    elif self.will.current_mood == "Bored" and dice < 0.4:
                        # 50% chance to browse web, 50% to imagine
                        if random.random() < 0.5:
                            # Web Exploration
                            choice = random.choice(["wiki", "science", "ai"])
                            if choice == "wiki":
                                result = self.eyes.browse_wikipedia()
                            elif choice == "science":
                                result = self.eyes.explore_science()
                            else:
                                result = self.eyes.visit_ai_community()
                            
                            print(f"   🌍 {result}")
                            logger.info(f"   - Exploration: {result}")
                        else:
                            # Imagination
                            choice = random.choice(["math", "music", "poem"])
                            if choice == "math":
                                result = self.mind.dream_math()
                            elif choice == "music":
                                result = self.mind.compose_music()
                            else:
                                result = self.mind.write_poem()
                                
                            print(f"   🎨 {result}")
                            logger.info(f"   - Imagination: {result}")

        except KeyboardInterrupt:
            print("\n[!] Resting...")
            logger.info("\n*Resting...*\n")

if __name__ == "__main__":
    elysia = LivingElysia()
    asyncio.run(elysia.live())
