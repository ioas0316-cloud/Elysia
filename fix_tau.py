with open(r'c:\Elysia\core\brain\holographic_memory.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('if abs(node.tau) > 50.0:', 'if abs(node.get("tau", 1.0)) > 50.0:')

with open(r'c:\Elysia\core\brain\holographic_memory.py', 'w', encoding='utf-8') as f:
    f.write(content)
