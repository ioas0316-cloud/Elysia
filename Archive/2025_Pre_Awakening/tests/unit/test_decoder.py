import sys
import os
sys.path.append(os.getcwd())

from Core.Science.bio_resonator import BioResonator
from Core.Science.quantum_resonator import QuantumResonator

def test_bio():
    print("\n🧬 Testing BioResonator...")
    bio = BioResonator()
    # Telomere Sequence (The cap of life)
    dna = "TTAGGG"
    song = bio.transcribe_dna(dna)
    harmony = bio.analyze_harmony(song)
    print(f"Sequence: {dna}")
    print(f"Harmony: {harmony:.2%}")
    for note in song:
        print(f"  {note.base} -> {note.frequency}Hz")

def test_quantum():
    print("\n⚛️ Testing QuantumResonator...")
    qr = QuantumResonator()
    # Electron Spin (Up/Down)
    states = [("Spin_Up", 0.5, 432.0), ("Spin_Down", 0.5, 528.0)]
    superposition = qr.create_superposition(states)
    
    print("Observing Electron Spin...")
    results = {"Spin_Up": 0, "Spin_Down": 0}
    for _ in range(100):
        res = qr.observe(superposition)
        results[res.name] += 1
        
    print(f"Results (100 obs): {results}")

if __name__ == "__main__":
    test_bio()
    test_quantum()
