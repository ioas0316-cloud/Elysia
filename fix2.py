with open(r'c:\Elysia\core\brain\holographic_memory.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
new_lines = []
for line in lines:
    if '_oracle =' in line or '_projector =' in line or 'logging.error(f"Failed to load StaticOracle' in line:
        continue
    if 'try:' in line or 'global _oracle, _projector' in line or 'except Exception as e:' in line:
        # Check if they are part of the old block
        continue
    new_lines.append(line)

with open(r'c:\Elysia\core\brain\holographic_memory.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
