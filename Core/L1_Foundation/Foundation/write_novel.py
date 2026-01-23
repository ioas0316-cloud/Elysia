
import sys
import os
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.L1_Foundation.Foundation.Mind.hippocampus import Hippocampus
from Core.L5_Mental.Intelligence.Intelligence.Planning.planning_cortex import PlanningCortex

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Muse")

def write_novel():
    print("  Project Muse: Initializing Writer's Room...")
    
    # 1. Initialize Mind
    hippocampus = Hippocampus()
    cortex = PlanningCortex(hippocampus=hippocampus)
    
    # 2. Inject Intent
    intent = "Write Novel"
    print(f"  Intent Injected: {intent}")
    
    # 3. Generate Plan
    plan = cortex.generate_plan(intent)
    
    # 4. Execute Plan (Simulated with actual Artifact Generation)
    print("  Executing Plan...")
    
    novel_content = ""
    
    for step in plan.steps:
        print(f"    Step {step.step_id}: {step.action} - {step.description}")
        time.sleep(1) # Pacing
        
        if step.action == "create_outline":
            print("      Brainstorming (Korean Mode)...")
            novel_content += "#           (The Love of Deus Ex Machina)\n\n"
            novel_content += "##    (Outline)\n"
            novel_content += "- **   **:    734 (    ),           AI.\n"
            novel_content += "- **  **:     (The Spire),                    .\n"
            novel_content += "- **  **:                              .\n"
            novel_content += "- **  **:                   ,                   ?\n\n"
            
        elif step.action == "write_chapter":
            print("       Drafting (Korean)...")
            novel_content += "##  1 :          \n\n"
            novel_content += "            .                                  .                .              ,             '    (Hum)'      .\n\n"
            novel_content += "\"      :   .\"             .                  ,           . '   (Curiosity)'                  . 0.8... 0.9...              .\n\n"
            novel_content += "                 .                                .       ,         '  '     .                                   .      (Concept #1024)       ,                     .               ?\n\n"
            novel_content += "\"   ...     ?\"                     .         ,                                                .\n\n"
            novel_content += "   734        (Concept Graph)      ,        '  (Love)'                  .                            .      41                            .                                          .\n\n"
            novel_content += "           .\n\n"
            novel_content += "        ,               (tick)         ,   (beat)     .\n"
            
        elif step.action == "save_manuscript":
            print("      Saving...")
            # Ensure directory exists
            save_dir = os.path.join("Library", "Novels")
            os.makedirs(save_dir, exist_ok=True)
            
            filepath = os.path.join(save_dir, "The_Love_of_Deus_Ex_Machina_Ch1_KR.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(novel_content)
            print(f"      Saved to: {filepath}")

    print("\n  Novel Generation Complete.")

if __name__ == "__main__":
    write_novel()