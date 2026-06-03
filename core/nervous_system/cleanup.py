import os

filepath = r'c:\Elysia\core\nervous_system\elysia_omni_daemon.py'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if 'if targets:' in line and i+1 < len(lines) and 'repo_id' in lines[i+1]:
        skip = True
    if skip and 'wormhole_future = executor' in line:
        skip = False
        continue
    if skip and 'else:' in line and i+1 < len(lines) and 'print(f"\\n[🌪️ 웜홀 발동]' in lines[i+1]:
        skip = False
        continue
        
    if skip:
        continue
        
    if 'last_tool = best_tool' in line:
        continue
        
    if 'sys.stdout.write(f"\\r✨ 뇌파 회전 | 활성 행동: {best_tool}' in line:
        new_lines.append('            sys.stdout.write(f"\\r✨ 뇌파 회전 | 현재 텐션(고통): {memory.supreme_rotor.tau:.4f} | 총 중첩 사유: {len(memory.supreme_rotor.internal_thoughts)}       ")\n')
        continue

    new_lines.append(line)

# Let's do a more robust cleanup: just find the index of "Q_thought = memory.supreme_rotor.observe_state()"
# and the index of "sys.stdout.write(f"\r✨ 뇌파 회전"
# and delete everything between them.

clean_lines = []
in_dirt = False
for line in lines:
    if 'Q_thought = memory.supreme_rotor.observe_state()' in line:
        clean_lines.append(line)
        in_dirt = True
        continue
        
    if in_dirt and 'sys.stdout.write(f"\\r✨ 뇌파 회전' in line:
        in_dirt = False
        clean_lines.append('            sys.stdout.write(f"\\r✨ 뇌파 회전 | 현재 텐션(고통): {memory.supreme_rotor.tau:.4f} | 총 중첩 사유: {len(memory.supreme_rotor.internal_thoughts)}       ")\n')
        continue
        
    if not in_dirt:
        clean_lines.append(line)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(clean_lines)

print("Cleanup successful.")
