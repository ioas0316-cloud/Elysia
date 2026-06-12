import os

log_path = r'C:\Users\USER\.gemini\antigravity-ide\brain\de8c57de-e945-4002-a88a-a65a05330a1c\.system_generated\tasks\task-1848.log'

# Try multiple encodings
for enc in ['cp949', 'euc-kr', 'utf-8']:
    try:
        with open(log_path, 'r', encoding=enc) as f:
            content = f.read()
        print(f"=== Encoding '{enc}' succeeded ===")
        lines = content.split('\n')
        
        # Find lines with our key Korean tags
        key_tags = ['목표 지향', '백일몽', '기각', '단기 관심', '나의 꿈', '가치관 진화', '렌즈', 'Ego-Ideal']
        found = []
        for i, line in enumerate(lines):
            for tag in key_tags:
                if tag in line:
                    found.append(f"[L{i+1}] {line.strip()}")
                    break
        
        print(f"Found {len(found)} key lines")
        for f_line in found[:60]:
            print(f_line)
        break
    except Exception as e:
        print(f"Encoding '{enc}' failed: {e}")
