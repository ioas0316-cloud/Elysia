from core.holographic_memory import HologramMemory
import math

memory = HologramMemory(num_layers=3)
concepts = ["수류학 (Su-Ryu-Hak)", "클리포드 레이어 (Clifford Layer)", "4차원 가변 로터 (4D Hyper-Rotor)"]
for c in concepts:
    memory.superpose(c)

# Let's sweep tension from 0 to 20 with small steps
print("Sweeping resonance peaks...")
for c in concepts:
    max_score = -1.0
    best_t = 0.0
    for step in range(200):
        t = step * 0.1
        res = memory.scan_resonance(t)
        score = res[c]
        if score > max_score:
            max_score = score
            best_t = t
    print(f"Concept '{c}': Max resonance = {max_score:.4f} at tension = {best_t:.2f}")
