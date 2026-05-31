import re

with open('src/phase_kernel.cpp', 'r') as f:
    content = f.read()

# Replace TrajectoryRotor
old_rotor = """    struct TrajectoryRotor {
        float past_momentum;   // 과거 진입 원심력
        float present_phase;   // 현재 위상각
        float future_gravity;  // 미래 인력 곡률
    };"""

new_rotor = """    // ==========================================
    // Causal Trajectory Rotor & Hologram Bridge
    // ==========================================

    // TrajectoryRotor is no longer a static container, but a 4x4 Quaternion Transformation Operator
    struct TrajectoryRotor {
        float w; // Real scalar part (Amplitude / Probability Density)
        float x; // i vector (Phase alignment X)
        float y; // j vector (Phase alignment Y)
        float z; // k vector (Phase alignment Z / Depth)
    };"""

content = content.replace(old_rotor, new_rotor)

with open('src/phase_kernel.cpp', 'w') as f:
    f.write(content)

print("Patch applied to src/phase_kernel.cpp")
