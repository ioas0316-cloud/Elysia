import re

filepath = "c:/Elysia/Core/Foundation/reasoning_engine.py"

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_line = 337 # def think
end_line = 541   # def think_quantum

print(f"Scanning lines {start_line} to {end_line} for 'Quaternion'...")

for i, line in enumerate(lines):
    lineno = i + 1
    if start_line <= lineno < end_line:
        if "Quaternion" in line:
            print(f"{lineno}: {line.strip()}")
