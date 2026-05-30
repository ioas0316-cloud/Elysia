import re

with open('src/phase_kernel.cpp', 'r') as f:
    content = f.read()

# I see what's wrong. There's a stray `}` closing `extern "C" {` on line 54, right before Causal Trajectory Rotor.
# And then later functions use `extern "C"` on each function instead of being inside the block.
# Let's fix the whole file to have one big `extern "C" { ... }` block, and remove all `extern "C"` inside it.

# Step 1: Remove `extern "C" {` at line 4
content = content.replace('extern "C" {\n', '\n')

# Step 2: Remove all `extern "C"` keywords
content = content.replace('extern "C" ', '')

# Step 3: Wrap everything after `#include <cstdint>` in `extern "C" { ... }`
lines = content.split('\n')
include_lines = []
code_lines = []

for line in lines:
    if line.startswith('#include'):
        include_lines.append(line)
    else:
        code_lines.append(line)

# Remove any stray closing braces that might be un-indented or randomly placed at the end/middle without being part of a function
cleaned_code_lines = []
brace_count = 0
for line in code_lines:
    if '{' in line:
        brace_count += line.count('{')
    if '}' in line:
        brace_count -= line.count('}')

    # Ignore the stray `}` at the end of the first block or at EOF if brace_count goes negative
    if brace_count < 0:
        brace_count = 0
        continue # skip this `}`

    cleaned_code_lines.append(line)

final_content = '\n'.join(include_lines) + '\n\nextern "C" {\n' + '\n'.join(cleaned_code_lines) + '\n}\n'

with open('src/phase_kernel.cpp', 'w') as f:
    f.write(final_content)

print("Fix 4 applied")
