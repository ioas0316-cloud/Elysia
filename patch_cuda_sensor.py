import re

with open('src/phase_kernel.cu', 'r') as f:
    content = f.read()

# Introduce Quantum Sensor logic:
# - Calculate delta (change/distortion) explicitly.
# - Instead of just modifying, return an observation metric (0 for perfectly aligned, 1 for active shifting)
# We will do this by returning an array of states or by modifying the matrix, but since the Python side
# expects the projection to just modify the Tapestry in place, we will express the sensor math internally
# for now, storing the result of the sensor's reading into a side-channel or just baking it into the kinetic equation.

# Wait, the user insight is purely about redefining the abstract meaning:
# [Alignment = 0 (Resistance)] -> Perfect match, static.
# [Change = 1 (Resistance)] -> Shift.
# Let's align the force variables with this terminology directly in the CUDA C++.

old_math = r"float force = 0.0f;.*?float dz = \(incoming_state\.z - vram_matrix\[idx\]\.z\) \* force;"

new_math = """// Quantum Sensor Logic (Spintronics / MTJ Model)
            // Alignment = 0 (Resistance Zero) -> Perfect match, no kinetic force needed, flat and stable.
            // Change = 1 (Resistance Max) -> Complete mismatch, needs massive kinetic twist to realign.

            float resistance_sensor = 1.0f - abs(alignment); // 0 when fully aligned (abs(dot)=1), 1 when orthogonal (dot=0)

            // The kinetic force applied to twist the spin is directly proportional to the resistance (change state)
            float kinetic_twist_force = 0.0f;

            if (resistance_sensor < 0.01f) {
                // State = 0 (Perfect Alignment): No change, system rests at 0 resistance
                kinetic_twist_force = 0.0f;
            } else if (resistance_sensor > 0.5f) {
                // State = 1 (High Resistance / Change): Apply strong repulsive/attractive kinetic twist
                kinetic_twist_force = 0.1f * resistance_sensor * (alignment > 0 ? 1.0f : -1.0f);
            } else {
                // Partial interference
                kinetic_twist_force = 0.05f * resistance_sensor;
            }

            // Apply Kinetic Momentum to Local VRAM Rotor (The Tapestry)
            float dw = (incoming_state.w - vram_matrix[idx].w) * kinetic_twist_force;
            float dx = (incoming_state.x - vram_matrix[idx].x) * kinetic_twist_force;
            float dy = (incoming_state.y - vram_matrix[idx].y) * kinetic_twist_force;
            float dz = (incoming_state.z - vram_matrix[idx].z) * kinetic_twist_force;"""

content = re.sub(old_math, new_math, content, flags=re.DOTALL)

with open('src/phase_kernel.cu', 'w') as f:
    f.write(content)
