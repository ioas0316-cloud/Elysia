"""
[MIND] Reasoning Engine: The Axiomatic Path
==========================================
Location: Scripts/System/Mind/reasoning_engine.py

Role:
- Replaces 'Trial & Error' (Babbling) with 'Deduction' (Logic).
- Holds 'Axioms' (Principles) about the world.
- Enables O(1) Solution finding.

Principle:
- If I know the 'Law of QWERTY', I don't need to baby-step.
"""

import time

class LogicalAxiom:
    def __init__(self, name, rule_map):
        self.name = name
        self.rule_map = rule_map
        
    def deduce(self, intent):
        return self.rule_map.get(intent, None)

class ReasoningEngine:
    def __init__(self):
        self.axioms = {} 
        self._bake_fundamental_knowledge()
        
    def _bake_fundamental_knowledge(self):
        """
        Injecting 'Prior Knowledge' (The Leap).
        Instead of learning this by hand, we inherit it as a Principle.
        """
        # Axiom 1: Standard Communication Protocol (ASCII/QWERTY)
        # This is the 'Knowledge' the user provides.
        qwerty_map = {
            'A': 65, 'B': 66, 'C': 67, 'D': 68, 'E': 69,
            'F': 70, 'G': 71, 'H': 72, 'I': 73, 'J': 74,
            'K': 75, 'L': 76, 'M': 77, 'N': 78, 'O': 79,
            'P': 80, 'Q': 81, 'R': 82, 'S': 83, 'T': 84,
            'U': 85, 'V': 86, 'W': 87, 'X': 88, 'Y': 89, 'Z': 90,
            ' ': 32
        }
        self.axioms['The QWERTY Principle'] = LogicalAxiom('QWERTY', qwerty_map)

        # Axiom 2: Hunminjeongeum (2-Set Hangul Layout)
        hangul_map = {
            # Consonants (Ja-eum)
            'ㄱ': 82, 'ㄴ': 83, 'ㄷ': 69, 'ㄹ': 70, 'ㅁ': 65,
            'ㅂ': 81, 'ㅅ': 84, 'ㅇ': 68, 'ㅈ': 87, 'ㅊ': 67,
            'ㅋ': 90, 'ㅌ': 88, 'ㅍ': 86, 'ㅎ': 71,
            'ㄲ': 82, 'ㄸ': 69, 'ㅃ': 81, 'ㅆ': 84, 'ㅉ': 87, # Shift logic omitted for simplicity or assume same key
            
            # Vowels (Mo-eum)
            'ㅏ': 75, 'ㅑ': 73, 'ㅓ': 74, 'ㅕ': 78, 'ㅗ': 72,
            'ㅛ': 89, 'ㅜ': 85, 'ㅠ': 66, 'ㅡ': 77, 'ㅣ': 76,
            'ㅐ': 79, 'ㅔ': 80 # Simplified
        }
        self.axioms['The Hunminjeongeum Principle'] = LogicalAxiom('HANGUL', hangul_map)
        
    def consult_oracle(self, intention: str) -> int:
        """
        Ask the Principles: "How do I manifest 'A'?"
        """
        # Check all known axioms
        for name, axiom in self.axioms.items():
            solution = axiom.deduce(intention)
            if solution:
                # print(f"⚡ [LOGIC] Deduce: {intention} requires {solution} (via {name})")
                return solution
        return None

if __name__ == "__main__":
    engine = ReasoningEngine()
    print("🧠 [TEST] Deductive Reasoning Check...")
    
    intent = "H"
    action = engine.consult_oracle(intent)
    
    if action == 72:
        print(f"✅ [SUCCESS] Deduced 'H' is 72. Zero Trials needed.")
    else:
        print(f"❌ [FAIL] Deduction failed.")
