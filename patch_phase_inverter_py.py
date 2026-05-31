import re

with open('lib/phase_inverter.py', 'r') as f:
    content = f.read()

# Replace TrajectoryRotor ctypes struct
old_struct = """class TrajectoryRotor(ctypes.Structure):
    _fields_ = [
        ("past_momentum", ctypes.c_float),
        ("present_phase", ctypes.c_float),
        ("future_gravity", ctypes.c_float)
    ]"""

new_struct = """class TrajectoryRotor(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float)
    ]"""

content = content.replace(old_struct, new_struct)

# Add project_phase_tensor and trace_trajectory bindings
binding_point = "        self.lib.synchronize_holographic_orbit.restype = ctypes.c_bool"
new_bindings = """        self.lib.synchronize_holographic_orbit.restype = ctypes.c_bool

        # Isomorphic Phase Projection
        self.lib.project_phase_tensor.argtypes = [TrajectoryRotor, ctypes.POINTER(TrajectoryRotor), ctypes.c_int]
        self.lib.project_phase_tensor.restype = None

        # Trace Trajectory using Complex Conjugate
        self.lib.trace_trajectory.argtypes = [TrajectoryRotor, ctypes.POINTER(TrajectoryRotor), ctypes.c_int]
        self.lib.trace_trajectory.restype = TrajectoryRotor"""

content = content.replace(binding_point, new_bindings)

# Add python methods to call the bindings
new_methods = """
    def project_phase_tensor(self, incoming_state: TrajectoryRotor, vram_matrix: ctypes.POINTER(TrajectoryRotor), matrix_size: int):
        \"\"\"Project a phase tensor globally across the VRAM array without indexing.\"\"\"
        self.lib.project_phase_tensor(incoming_state, vram_matrix, matrix_size)

    def trace_trajectory(self, final_state: TrajectoryRotor, vram_matrix: ctypes.POINTER(TrajectoryRotor), matrix_size: int) -> TrajectoryRotor:
        \"\"\"Traces back the origin of a state using reverse complex conjugate projection.\"\"\"
        return self.lib.trace_trajectory(final_state, vram_matrix, matrix_size)
"""

content += new_methods

with open('lib/phase_inverter.py', 'w') as f:
    f.write(content)

print("Patch applied for python bindings")
