import re

with open('src/phase_kernel.cu', 'r') as f:
    content = f.read()

content = "#include <stdio.h>\n" + content

with open('src/phase_kernel.cu', 'w') as f:
    f.write(content)
