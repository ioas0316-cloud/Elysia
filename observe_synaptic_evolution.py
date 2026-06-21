import numpy as np
from synaptic_architecture.organism import OmniModalOrganism

def run_omni_evolution():
    print("==================================================================")
    print(" [Omni-Modal Evolution] Discovering Unified Bitstream Laws")
    print("==================================================================\n")

    omo = OmniModalOrganism(resolution=256)

    # 1. Modality Bitstreams
    # ASCII 'Elysia'
    text_bits = np.unpackbits(np.frombuffer(b'Elysia', dtype=np.uint8))
    text_wave = np.pad(text_bits, (0, 64-len(text_bits)))

    # RGB 'Red' (Purely represented as 0/1 density)
    rgb_bits = np.unpackbits(np.array([255, 0, 0], dtype=np.uint8))
    rgb_wave = np.pad(rgb_bits, (0, 64-len(rgb_bits)))

    # Physics 'Force' (Sine pattern)
    phys_wave = (np.sin(np.linspace(0, 10, 64)) > 0).astype(np.uint8)

    # 2. Evolutionary Discovery Loop
    for i, (wave, label) in enumerate([(text_wave, "ASCII_TEXT"), (rgb_wave, "RGB_COLOR"), (phys_wave, "PHYSICS_LAW")]):
        print(f"\n--- [Phase {i+1}] Discovering {label} ---")

        # High Temp Exploration (High jitter/vibration)
        omo.scheduler.set_temperature(4.0)
        omo.perceive_and_map(wave, label)

        # Low Temp Solidification (Crystallization)
        omo.scheduler.set_temperature(0.1)
        omo.perceive_and_map(wave, label)

    # 3. Final Unified Mapping Observation
    final_center = np.unravel_index(np.argmax(omo.field.conductance), omo.field.conductance.shape)
    print(f"\n[System] Unified Cognitive Center crystallized at {final_center}")
    print("==================================================================")

if __name__ == "__main__":
    run_omni_evolution()
