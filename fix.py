import re

file_hm = r'c:\Elysia\core\brain\holographic_memory.py'
with open(file_hm, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove imports
content = re.sub(r'from core\.brain\.magnetic_torus_buffer import MagneticTorusBuffer\n', '', content)
content = re.sub(r'from core\.brain\.static_oracle import StaticOracle\n', '', content)
content = re.sub(r'from core\.brain\.phase_mirror import PhaseMirrorProjector\n', '', content)

# Remove static oracle instantiation
content = re.sub(r'try:\s*global _oracle, _projector\s*_oracle = StaticOracle\(\)\s*_projector = PhaseMirrorProjector\(_oracle\.model\.config\.hidden_size\)\s*except Exception as e:\s*logging\.error\(f\"Failed to load StaticOracle.*\n', '', content)

# Replace MagneticTorusBuffer with SpacetimeRotor
content = content.replace('self.torus_buffer = MagneticTorusBuffer()', 'from core.brain.spacetime_rotor import SpacetimeRotor\n        self.torus_buffer = SpacetimeRotor("Pain_Buffer")')
content = content.replace('self.torus_buffer.inject_phase_wave(concept, projected_rotor)', 'self.torus_buffer.stream_flow(projected_rotor)')
content = content.replace('self.torus_buffer.inject_phase_wave(f"CRYSTAL_{int(time.time())}", crystal_quat)', 'self.torus_buffer.stream_flow(crystal_quat)')

with open(file_hm, 'w', encoding='utf-8') as f:
    f.write(content)

file_ev = r'c:\Elysia\core\nervous_system\evolution_sandbox.py'
with open(file_ev, 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'self\.memory\.torus_buffer\.inject_phase_wave\(f"PAIN_SCAR_\{cortex_name\}", pain_wave\)', 'self.memory.torus_buffer.stream_flow(pain_wave)', content)

with open(file_ev, 'w', encoding='utf-8') as f:
    f.write(content)

print('Cleaned up Holographic Memory and Evolution Sandbox.')
