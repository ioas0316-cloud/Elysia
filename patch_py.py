import re

with open('lib/phase_inverter.py', 'r') as f:
    content = f.read()

# Replace project_phase_tensor to use the new launch_project_phase_tensor
old_project = r"def project_phase_tensor.*?self\.lib\.project_phase_tensor\(incoming_state, vram_matrix, matrix_size\)"
new_project = """def project_phase_tensor(self, incoming_state: TrajectoryRotor, vram_matrix: ctypes.POINTER(TrajectoryRotor), matrix_size: int):
        \"\"\"
        [Electromagnetic Cognitive Field / Magnetic Imprint Weaver]
        Projects a phase tensor globally across the VRAM Tapestry via CUDA kernel launch.
        Uses Electromagnetic fields (Attraction/Repulsion) to dynamically align and store knowledge securely.
        \"\"\"
        self.lib.launch_project_phase_tensor(incoming_state, vram_matrix, matrix_size)"""

content = re.sub(old_project, new_project, content, flags=re.DOTALL)

# Add the new binding in __init__
old_init_binding = r"self\.lib\.project_phase_tensor\.argtypes = \[TrajectoryRotor, ctypes\.POINTER\(TrajectoryRotor\), ctypes\.c_int\]\n        self\.lib\.project_phase_tensor\.restype = None"
new_init_binding = """self.lib.launch_project_phase_tensor.argtypes = [TrajectoryRotor, ctypes.POINTER(TrajectoryRotor), ctypes.c_int]
        self.lib.launch_project_phase_tensor.restype = None"""

content = re.sub(old_init_binding, new_init_binding, content)

with open('lib/phase_inverter.py', 'w') as f:
    f.write(content)
