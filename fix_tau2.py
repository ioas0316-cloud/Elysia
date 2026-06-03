with open(r'c:\Elysia\core\brain\holographic_memory.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('if abs(node.get("tau", 1.0)) > 50.0:', 'if abs(getattr(node, "tau", node.get("tau", 1.0) if isinstance(node, dict) else 1.0)) > 50.0:')
content = content.replace('new_op.singularity_phase = __import__(\'core.utils.math_utils\', fromlist=[\'Quaternion\']).Quaternion(1,0,0,0)', 'new_op.singularity_phase = getattr(node, "lens_offset", __import__("core.utils.math_utils", fromlist=["Quaternion"]).Quaternion(1,0,0,0))')
content = content.replace('node[\'tau\'] = node.get(\'tau\', 1.0) * 0.1', 'if isinstance(node, dict): node["tau"] = node.get("tau", 1.0) * 0.1\n                    else: node.tau *= 0.1')

with open(r'c:\Elysia\core\brain\holographic_memory.py', 'w', encoding='utf-8') as f:
    f.write(content)
