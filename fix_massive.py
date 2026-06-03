with open(r'c:\Elysia\test_globe_massive.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
'''            memory.ui_concept_map.append({
                "id": str(len(memory.ui_concept_map)),
                "label": label,
                "group": 2,
                "tau": 1.0
            })''',
'''            cid = str(len(memory.ui_concept_map))
            memory.ui_concept_map[cid] = {
                "id": cid,
                "label": label,
                "group": 2,
                "tau": 1.0
            }'''
)
with open(r'c:\Elysia\test_globe_massive.py', 'w', encoding='utf-8') as f:
    f.write(content)
