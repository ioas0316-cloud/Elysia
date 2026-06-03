import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.rotor_gate import ConceptWave
from core.knowledge_space import PhaseSpace

def run_simulation():
    print("🌌 Initializing Elysia's Dynamic Phase Space...")
    space = PhaseSpace()

    # 1. 초기 엔티티 생성
    korean = ConceptWave("Korean")
    english = ConceptWave("English")
    
    space.add_concept(korean)
    space.add_concept(english)
    
    print("\n[Step 1] Two distinct concepts created.")
    space.print_state()

    # 2. 같음의 공리 (Alignment on 'Language' Axis)
    print("[Step 2] Introducing the 'Language' criteria (Sameness).")
    print("Traditional Logic: Is Korean a Language? Yes. Is English a Language? Yes.")
    print("RotorGate Logic: Applying criteria...")
    res1 = space.introduce_criteria("Korean", "English", "Language")
    print("=>", res1)
    print("Notice that instead of a static 'True', they gained kinetic velocity (momentum) on the Language axis.")
    space.print_state()

    # 3. 다름의 공리 (Differentiation generating a new axis)
    print("[Step 3] Introducing the 'Difference' criteria (Differentiation).")
    print("Traditional Logic: Are Korean and English the same? No (Static False).")
    print("RotorGate Logic: They share 'Language' base, but differ in form. Branching new 'Alphabet_Type' axis...")
    res2 = space.branch_differentiation("Korean", "English", base_axis="Language", new_axis="Alphabet_Type")
    print("=>", res2)
    
    # After differentiation, they interact on the new axis
    res3 = space.introduce_criteria("Korean", "English", "Alphabet_Type")
    print("=>", res3)
    space.print_state()

    # 4. 자율적 탐색/운동성 (Adding a third concept)
    print("[Step 4] The 'Kinetic Velocity' pushes them to seek sameness. A new concept arrives: 'French'.")
    french = ConceptWave("French")
    french.add_axis("Language", 0.0) # Assume it naturally aligns with the language vibration
    space.add_concept(french)

    # French interacts with the seeking English wave
    res4 = space.introduce_criteria("English", "French", "Language")
    print("=>", res4)

    # They differ in alphabet type (assuming Latin vs something else, or maybe just 'Origin')
    res5 = space.branch_differentiation("English", "French", base_axis="Language", new_axis="Origin")
    print("=>", res5)
    
    space.print_state()
    print("🏁 Simulation Complete. The concepts are no longer static nodes, but vectors with velocity and structural tension.")

if __name__ == "__main__":
    run_simulation()
