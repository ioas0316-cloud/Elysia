# [Genesis: 2025-12-02] Purified by Elysia
"""
Demonstrate math verification via pipeline extension without modifying the pipeline.

Usage:
  python -m scripts.pipeline_math_verify_demo "3 * (2 + 4) = 18"
"""
from __future__ import annotations

import sys
from Project_Elysia.cognition_pipeline import CognitionPipeline
from Project_Elysia.pipeline_extensions import run_math_verification


def main():
    statement = sys.argv[1] if len(sys.argv) > 1 else "3 * (2 + 4) = 18"
    pipe = CognitionPipeline()
    result = run_math_verification(pipe, statement)
    print(result)


if __name__ == "__main__":
    main()
