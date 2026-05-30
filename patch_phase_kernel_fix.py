import re

with open('src/phase_kernel.cpp', 'r') as f:
    content = f.read()

# Fix the nested extern "C" error by removing the redundant specifier
content = content.replace('    extern "C" void project_phase_tensor', '    void project_phase_tensor')

with open('src/phase_kernel.cpp', 'w') as f:
    f.write(content)

print("Fix applied")
