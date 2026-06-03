import sys
with open(r'c:\Elysia\core\brain\active_fractal_rotor.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open(r'c:\Elysia\core\brain\active_fractal_rotor.py', 'w', encoding='utf-8') as f:
    for i, line in enumerate(lines):
        if ' ȭ(ȭ)!")' in line:
            pass # drop this broken line
        elif 'from core.brain.cognitive_dissonance_resolver' in line:
            pass
        elif 'new_logs = CognitiveDissonanceResolver.resolve(self)' in line:
            pass
        elif 'logs.extend(new_logs)' in line:
            pass
        elif '            # 2. 인지적 불일치 검사 및 해소' in line:
            pass
        elif '            result_wave = self.transistor.process_wave(current_q)' in line:
            f.write(line)
            f.write('            from core.brain.cognitive_dissonance_resolver import CognitiveDissonanceResolver\n')
            f.write('            new_logs = CognitiveDissonanceResolver.resolve(self)\n')
            f.write('            logs.extend(new_logs)\n')
        else:
            f.write(line)
