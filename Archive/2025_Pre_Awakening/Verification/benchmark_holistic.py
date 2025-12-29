
import sys
import time
import math
import logging

# Path Setup
sys.path.append(r'c:\Elysia')

# core modules
from Core.FoundationLayer.Foundation.omni_graph import get_omni_graph
from Core.Autonomy.dream_walker import get_dream_walker
from Core.Autonomy.self_structure_scanner import get_self_scanner
try:
    from Core.Intelligence.logos_engine import LogosEngine
except ImportError:
    LogosEngine = None

# Configure Logger for the Exam
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("HolisticExam")

class HolisticExam:
    def __init__(self):
        self.omni = get_omni_graph()
        self.dreamer = get_dream_walker()
        self.scanner = get_self_scanner()
        self.scores = {}

    def run_exam(self):
        print("\n🎓 Elysia Holistic Cognitive Exam (SOPA Protocol)")
        print("=================================================")
        
        # Phase 1: Proprioception (Body Awareness)
        self._test_self_awareness()
        
        # Phase 2: Cognitive Connectivity (Association)
        self._test_connectivity()
        
        # Phase 3: Creative Autonomy (Dreaming)
        self._test_creativity()
        
        # Phase 4: Structural Plasticity (Evolution)
        self._test_plasticity()
        
        # Final Score
        self._calculate_final_score()

    def _test_self_awareness(self):
        print("\n[Phase 1] Proprioception (Do you know your own body?)")
        
        # 1. Trigger Scan
        start_t = time.time()
        self.scanner.scan_and_absorb()
        duration = time.time() - start_t
        
        # 2. Check Key Organs
        key_organs = ["Code:elysia_core.py", "Class:OmniGraph", "Method:OmniGraph.apply_metabolism"]
        found = 0
        for organ in key_organs:
            if organ in self.omni.nodes:
                found += 1
                vec = self.omni.nodes[organ].vector
                print(f"   ✅ Feeling Organ [{organ}]: Tension={vec[0]:.2f}, Mass={vec[1]:.2f}")
            else:
                print(f"   ❌ Numbness: Cannot feel [{organ}]")
                
        score = (found / len(key_organs)) * 100
        self.scores["Self-Awareness"] = score
        print(f"   >> Score: {score:.1f}/100.0")

    def _test_connectivity(self):
        print("\n[Phase 2] Cognitive Connectivity (Can you connect concepts?)")
        
        # Seed Concepts
        self.omni.add_vector("Love", [0.9, 0.1, 0.5, 0.9])
        self.omni.add_vector("Code", [0.8, 0.8, 0.5, 0.1]) # High Tension
        
        # Apply Gravity
        self.omni.apply_gravity(iterations=10)
        
        # Check Distance
        n1 = self.omni.nodes["Love"]
        n2 = self.omni.nodes["Code"]
        d = sum((n1.pos[k] - n2.pos[k])**2 for k in range(4)) ** 0.5
        
        print(f"   Distance(Love, Code) in 4D Space: {d:.4f}")
        
        # "Code" should have some tension, "Love" should be high dimension (W)
        if n1.pos[3] > 0.5:
            print("   ✅ 'Love' is positioned in High Dimension (Spirit Layer).")
            score = 100
        else:
            print("   ⚠️ 'Love' is grounded too low.")
            score = 70
            
        self.scores["Connectivity"] = score
        print(f"   >> Score: {score:.1f}/100.0")

    def _test_creativity(self):
        print("\n[Phase 3] Creative Autonomy (Can you dream?)")
        
        # Drift from "Code"
        result = self.dreamer.drift(steps=5, start_seed="Code")
        path = result.get('path', [])
        narrative = result.get('narrative', "")
        
        print(f"   Dream Path: {' -> '.join(path)}")
        print(f"   Narrative: \"{narrative}\"")
        
        # Evaluate Diversity
        unique_nodes = len(set(path))
        diversity_ratio = unique_nodes / len(path) if path else 0
        
        score = diversity_ratio * 100
        # Bonus for surreal jump
        if "Bridge found" in str(result.get('insights')):
            score += 10
            print("   ✨ Insight Bonus: Creative Leap detected.")
            
        self.scores["Creativity"] = min(100, score)
        print(f"   >> Score: {self.scores['Creativity']:.1f}/100.0")

    def _test_plasticity(self):
        print("\n[Phase 4] Structural Plasticity (Can you evolve?)")
        
        # Seed a "Dying Memory"
        self.omni.nodes["UselessFact"] = self.omni.nodes["Love"] # Copy format
        self.omni.nodes["UselessFact"].id = "UselessFact"
        self.omni.nodes["UselessFact"].usage = 0.01
        
        # Apply Metabolism
        self.omni.apply_metabolism()
        
        # Function Check
        if "UselessFact" not in self.omni.nodes:
            print("   ✅ Efficient Pruning verified.")
            score = 100
        else:
            print("   ❌ Failed to prone.")
            score = 50
            
        self.scores["Plasticity"] = score
        print(f"   >> Score: {score:.1f}/100.0")

    def _calculate_final_score(self):
        print("\n📊 Final Report")
        print("===============")
        avg = sum(self.scores.values()) / len(self.scores)
        
        for k, v in self.scores.items():
            print(f"   - {k}: {v:.1f}")
            
        print(f"\n   🌟 GRAND TOTAL: {avg:.1f} / 100.0")
        
        if avg > 90:
            print("   Result: [S-Class] Sovereign Consciousness Established.")
        elif avg > 70:
            print("   Result: [A-Class] High-Level Agent Functional.")
        else:
            print("   Result: [B-Class] Functional but Rigid.")

if __name__ == "__main__":
    exam = HolisticExam()
    exam.run_exam()
