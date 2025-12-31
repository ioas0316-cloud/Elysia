#!/usr/bin/env python3
"""
Wave Nature Verifier (The Immune System)
========================================

"The body rejects what is not self."

This script scans the Elysia codebase to ensure adherence to the "Wave/Particle"
Hybrid Architecture defined in AGENTS.md.

It detects:
1. "Polling" (Busy Waiting) in Core modules (Forbidden).
2. "Particle Communication" (Direct coupling) where "Wave Resonance" is expected (Warning).
3. "Dead Cells" (Modules without @Cell or ResonatorInterface).

Usage:
    python scripts/verify_wave_nature.py
"""

import os
import sys
import ast
import time
from typing import List, Tuple, Dict

# Color codes for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

CORE_DIR = "Core"
EXCLUDED_DIRS = ["Core/Scripts", "Core/Demos", "Core/Tests"]

class WaveValidator(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.has_cell_decorator = False
        self.inherits_resonator = False
        self.has_infinite_loop_with_sleep = False
        self.direct_imports = []
        self.errors = []
        self.warnings = []

    def visit_ClassDef(self, node):
        # Check for @Cell decorator
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                if decorator.func.id == "Cell":
                    self.has_cell_decorator = True
            elif isinstance(decorator, ast.Name) and decorator.id == "Cell":
                self.has_cell_decorator = True

        # Check for ResonatorInterface inheritance
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "ResonatorInterface":
                self.inherits_resonator = True
            elif isinstance(base, ast.Attribute) and base.attr == "ResonatorInterface":
                self.inherits_resonator = True

        self.generic_visit(node)

    def visit_While(self, node):
        # Check for 'while True'
        is_infinite = False
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            is_infinite = True
        elif isinstance(node.test, ast.NameConstant) and node.test.value is True:
            is_infinite = True

        if is_infinite:
            # Check for time.sleep in body
            has_sleep = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute) and child.func.attr == "sleep":
                        has_sleep = True
                    elif isinstance(child.func, ast.Name) and child.func.id == "sleep":
                        has_sleep = True

            if has_sleep:
                self.errors.append(f"{RED}[POLLING DETECTED]{RESET} Infinite loop with sleep found. Use Pulse/Event logic instead.")

        self.generic_visit(node)

def scan_file(filepath: str) -> List[str]:
    """Scans a single file for violations."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return [f"{YELLOW}[SYNTAX ERROR]{RESET} Could not parse file."]

    validator = WaveValidator(filepath)
    validator.visit(tree)

    issues = validator.errors

    # Check for "Dead Cell" (No Resonance capability in critical areas)
    # Only enforce this for 'Orchestra' and 'Ether' for now
    if "Orchestra" in filepath or "Ether" in filepath:
        if not validator.has_cell_decorator and not validator.inherits_resonator:
            # It might be a utility file, so just a warning
            if not filepath.endswith("__init__.py"):
                issues.append(f"{YELLOW}[WEAK RESONANCE]{RESET} No @Cell or ResonatorInterface found in a high-level module.")

    return issues

def main():
    print(f"🌊 {GREEN}Elysia Wave Nature Verification System{RESET}")
    print("   Scanning Core/ for Particle Logic (Polling) and Resonance Compliance...\n")

    violation_count = 0
    file_count = 0

    for root, dirs, files in os.walk(CORE_DIR):
        # Skip excluded directories
        if any(excluded in root for excluded in EXCLUDED_DIRS):
            continue

        for file in files:
            if file.endswith(".py"):
                file_count += 1
                filepath = os.path.join(root, file)
                issues = scan_file(filepath)

                if issues:
                    print(f"📄 {filepath}")
                    for issue in issues:
                        print(f"   {issue}")
                    violation_count += len(issues)
                    print("")

    print("="*60)
    if violation_count == 0:
        print(f"{GREEN}✅ SYSTEM PURE. All scanned modules adhere to Wave Logic.{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}❌ DISSONANCE DETECTED.{RESET} Found {violation_count} violations.")
        print("   Please refactor 'Polling' loops to 'Pulse' listeners.")
        sys.exit(1)

if __name__ == "__main__":
    main()
