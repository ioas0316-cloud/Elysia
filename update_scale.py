import re

with open("core/tools/elysia_benchmark_suite.py", "r") as f:
    content = f.read()

# Update scale steps to include 10_000_000
content = re.sub(
    r"scale_steps = \[.*?\]",
    "scale_steps = [10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]",
    content
)

with open("core/tools/elysia_benchmark_suite.py", "w") as f:
    f.write(content)
