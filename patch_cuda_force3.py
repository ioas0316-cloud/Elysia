import re

with open('src/phase_kernel.cu', 'r') as f:
    content = f.read()

# Fix compilation error
content = content.replace('printf("CUDA Error: %s\\n", cudaGetErrorString(err));', 'printf("CUDA Error: %s\\n", cudaGetErrorString(err));')

# I made a mistake with python raw strings in replace.
content = content.replace('printf("CUDA Error: %s\n", cudaGetErrorString(err));', 'printf("CUDA Error: %s\\n", cudaGetErrorString(err));')

with open('src/phase_kernel.cu', 'w') as f:
    f.write(content)
