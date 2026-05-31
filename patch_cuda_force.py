import re

with open('src/phase_kernel.cu', 'r') as f:
    content = f.read()

# Fix the alignment logic so that orthogonal (dot=0) also gets a kinetic twist,
# Currently `alignment > 0 ? 1.0f : -1.0f` for dot=0 will evaluate to -1.0f, which is fine,
# but maybe resistance_sensor needs to be correctly mapped.

old_math = r"float kinetic_twist_force = 0\.0f;.*?kinetic_twist_force = 0\.05f \* resistance_sensor;\n            \}"
new_math = """float kinetic_twist_force = 0.0f;

            if (resistance_sensor < 0.01f) {
                // State = 0 (Perfect Alignment): No change, system rests at 0 resistance
                kinetic_twist_force = 0.0f;
            } else if (resistance_sensor > 0.5f) {
                // State = 1 (High Resistance / Change): Apply strong repulsive/attractive kinetic twist
                kinetic_twist_force = 0.5f * resistance_sensor; // Higher multiplier to see effect in test
            } else {
                // Partial interference
                kinetic_twist_force = 0.1f * resistance_sensor;
            }"""

content = re.sub(old_math, new_math, content, flags=re.DOTALL)

with open('src/phase_kernel.cu', 'w') as f:
    f.write(content)
