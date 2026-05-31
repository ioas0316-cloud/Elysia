import ctypes
from lib.phase_inverter import PhaseInverterGate, TrajectoryRotor

def test_alignment():
    gate = PhaseInverterGate()
    MATRIX_SIZE = 5

    vram_array = (TrajectoryRotor * MATRIX_SIZE)()
    # Initialize unaligned/neutral
    for i in range(MATRIX_SIZE):
        vram_array[i] = TrajectoryRotor(w=1.0, x=0.0, y=0.0, z=0.0)

    # Incoming alignment field (highly aligned with w=1.0)
    incoming = TrajectoryRotor(w=0.9, x=0.1, y=0.1, z=0.1)

    print("Before Alignment:")
    for i in range(MATRIX_SIZE):
        print(f"Rotor {i}: ({vram_array[i].w:.4f}, {vram_array[i].x:.4f}, {vram_array[i].y:.4f}, {vram_array[i].z:.4f})")

    gate.project_phase_tensor(incoming, vram_array, MATRIX_SIZE)

    print("\nAfter Magnetic Imprint Alignment (1st pass):")
    for i in range(MATRIX_SIZE):
        print(f"Rotor {i}: ({vram_array[i].w:.4f}, {vram_array[i].x:.4f}, {vram_array[i].y:.4f}, {vram_array[i].z:.4f})")

    gate.project_phase_tensor(incoming, vram_array, MATRIX_SIZE)

    print("\nAfter Magnetic Imprint Alignment (2nd pass):")
    for i in range(MATRIX_SIZE):
        print(f"Rotor {i}: ({vram_array[i].w:.4f}, {vram_array[i].x:.4f}, {vram_array[i].y:.4f}, {vram_array[i].z:.4f})")

test_alignment()
