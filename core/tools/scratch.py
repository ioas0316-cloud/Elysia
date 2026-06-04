import re

file_path = r"c:\Elysia\core\nervous_system\elysia_omni_daemon.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the beginning of main() to instantiate sandbox
pattern_main = r"def main\(\):"
repl_main = r"""def main():
    from core.nervous_system.evolution_sandbox import EvolutionSandbox
"""
content = re.sub(pattern_main, repl_main, content)

# We can dynamically inject sandbox into the workers.
workers = [
    ("zero_distance_worker", "직관의 눈"),
    ("browser_mirror_worker", "위상 거울 브라우저"),
    ("autonomous_motor_worker", "자율 운동 피질"),
    ("vocal_cortex_worker", "발화 피질"),
    ("gpu_visual_worker", "GPU 시각 피질"),
    ("hardware_heartbeat_worker", "심장 박동"),
    ("parental_auditory_worker", "청각 피질"),
    ("phase_135_zero_streaming_worker", "웜홀 사냥"),
    ("archetype_hunter_worker", "아키타입 헌터"),
    ("multilingual_perception_worker", "바벨탑 붕괴"),
    ("source_code_mutation_worker", "소스코드 거울")
]

for func_name, cortex_name in workers:
    # Find the function definition and its while loop
    # We want to insert 'from core.nervous_system.evolution_sandbox import EvolutionSandbox' and 'sandbox = EvolutionSandbox(memory)' before the while loop
    
    # Regex to match function definition until `while running[0]:`
    pattern = r'(def ' + func_name + r'\(memory: HologramMemory, running: list, ans: AutonomicNervousSystem\):.*?)(    while running\[0\]:)'
    
    repl = r'\1    from core.nervous_system.evolution_sandbox import EvolutionSandbox\n    sandbox = EvolutionSandbox(memory)\n\2\n        try:'
    
    # Now we need to indent everything inside the while loop and add the except block.
    # This is tricky with regex. Instead, let's just do a string manipulation.

def process_worker(code, func_name, cortex_name):
    lines = code.split('\n')
    in_func = False
    in_while = False
    new_lines = []
    
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            in_func = True
            new_lines.append(line)
            continue
            
        if in_func and line.startswith("def "):
            in_func = False # exited function
            
        if in_func and "while running[0]:" in line:
            in_while = True
            indent = line.split("while")[0]
            # Inject sandbox init
            new_lines.append(indent + "from core.nervous_system.evolution_sandbox import EvolutionSandbox")
            new_lines.append(indent + "sandbox = EvolutionSandbox(memory)")
            new_lines.append(line)
            new_lines.append(indent + "    try:")
            continue
            
        if in_while:
            if line.strip() == "":
                new_lines.append(line)
                continue
            # check if we exited the while loop
            current_indent = len(line) - len(line.lstrip())
            while_indent = len(line.split("while")[0]) if "while" in line else 4 # approximate
            # Actually we can just indent everything by 4 spaces until the indent goes back to the while loop's indent
            if current_indent <= 4 and not line.strip().startswith("ans.breathe") and not line.strip() == "":
                # exited while loop
                in_while = False
                in_func = False # effectively done with this function
                
        if in_while:
            # check if it's the ans.breathe line which usually ends the loop
            if "ans.breathe" in line:
                new_lines.append("    " + line)
                indent = line.split("ans")[0]
                new_lines.append(indent + f"except Exception as e:")
                new_lines.append(indent + f"    sandbox.absorb_pain(\"{cortex_name}\", e)")
                new_lines.append(indent + f"    ans.breathe(1.0)")
                in_while = False # Assume it's the last line
            else:
                new_lines.append("    " + line)
        else:
            if not in_while or (in_while and not "while running" in line):
                if line != lines[i-1] if i>0 else True: # simplistic check
                    if not in_func or not in_while:
                        pass
            
    # The above loop logic is a bit brittle. Let's do a more robust approach:
    pass

# ACTUALLY, it is much safer to just use multi_replace_file_content for the 3 most error-prone cortexes:
# 1. autonomous_motor_worker
# 2. phase_135_zero_streaming_worker
# 3. archetype_hunter_worker
# And just manually wrap their while loop contents.

pass
