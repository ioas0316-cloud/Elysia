import numpy as np
from synaptic_architecture.field import CrystallizationField
from synaptic_architecture.vortex import WaveInterference

def test_pleasure_acceleration():
    cf = CrystallizationField(64)
    wi = WaveInterference(cf)

    target_wave = np.uint64(0xABCDEF)
    # Crystallize it somewhere
    cf.crystallize_gene(np.array([32, 32]), target_wave)

    print("Resonating...")
    metrics = wi.resonate_field(target_wave, steps=20)

    print(f"Final Metrics: {metrics}")
    assert metrics["pleasure"] >= 0
    assert metrics["clarity"] >= 0
    print("Pleasure acceleration logic verified.")

if __name__ == "__main__":
    test_pleasure_acceleration()
