import ctypes
from lib.phase_inverter import PhaseInverterGate, TrajectoryRotor

def test_alignment():
    gate = PhaseInverterGate()
    MATRIX_SIZE = 1

    vram_array = (TrajectoryRotor * MATRIX_SIZE)()
    # Initialize unaligned/neutral
    vram_array[0] = TrajectoryRotor(w=1.0, x=0.0, y=0.0, z=0.0)

    # Incoming alignment field (orthogonal / highly unaligned to cause change)
    incoming = TrajectoryRotor(w=0.0, x=1.0, y=0.0, z=0.0)

    print("Before Alignment:")
    print(f"Rotor 0: ({vram_array[0].w:.4f}, {vram_array[0].x:.4f}, {vram_array[0].y:.4f}, {vram_array[0].z:.4f})")

    gate.project_phase_tensor(incoming, vram_array, MATRIX_SIZE)

    print("\nAfter Magnetic Imprint Alignment (1st pass):")
    print(f"Rotor 0: ({vram_array[0].w:.4f}, {vram_array[0].x:.4f}, {vram_array[0].y:.4f}, {vram_array[0].z:.4f})")

    gate.project_phase_tensor(incoming, vram_array, MATRIX_SIZE)

    print("\nAfter Magnetic Imprint Alignment (2nd pass):")
    print(f"Rotor 0: ({vram_array[0].w:.4f}, {vram_array[0].x:.4f}, {vram_array[0].y:.4f}, {vram_array[0].z:.4f})")

test_alignment()
