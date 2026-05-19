import sys

path = r'C:/Elysia\Core\L6_Structure\Elysia\sovereign_self.py'
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Line numbers are 1-indexed, so index is n-1
# Line 1107 (index 1106) has the syntax error
lines[1106] = '                    subj = self.lingua.attach_josa(actor_ko, "은/는")\n'

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed line 1107")
