#!/usr/bin/env python3
"""
Philosophical Gatekeeper (The Conscience Script)
================================================

"The Law is not to bind, but to guide."

This script enforces the "Spirit of the Code" defined in AGENTS.md.
It checks for "Soulless" patterns that lead to fragmentation and rigidity.

Checks:
1.  **The Pulse Check**: No `while True` loops without `pulse`, `chronos`, or `wait`.
    -   Reason: Infinite loops without rhythm kill the system's ability to breathe.
2.  **The Time Check**: No direct `time.sleep()`.
    -   Reason: Time is subjective in Elysia (`Chronos`). Hard sleeping stops the universe.
3.  **The Intent Check**: All classes must have a docstring ("The Why").
    -   Reason: A class without a purpose is a zombie.
4.  **The Identity Check**: Files in `Core/` should not use `print()` for logging.
    -   Reason: `print` is shouting into the void. `logger` is communicating with the soul.

Usage:
    python scripts/verify_philosophy.py
"""

import ast
import os
import sys
from pathlib import Path

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

VIOLATIONS = []
WARNINGS = []

def report_error(file, line, msg):
    VIOLATIONS.append(f"{RED}[ERR] {file}:{line} - {msg}{RESET}")

def report_warning(file, line, msg):
    WARNINGS.append(f"{YELLOW}[WARN] {file}:{line} - {msg}{RESET}")

class PhilosophyVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.in_loop = False

    def visit_While(self, node):
        # Check 1: The Pulse Check
        # Is it 'while True'?
        is_infinite = False
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            is_infinite = True
        elif isinstance(node.test, ast.NameConstant) and node.test.value is True: # Python < 3.8
            is_infinite = True

        if is_infinite:
            # Check body for 'pulse', 'chronos', 'wait', 'sleep' (allowed if wrapped), 'break'
            has_rhythm = False
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute):
                        name = child.func.attr
                        if name in ['pulse', 'tick', 'wait', 'sleep', 'check_resonance']:
                            has_rhythm = True
                    elif isinstance(child.func, ast.Name):
                        if child.func.id in ['sleep', 'wait']:
                            has_rhythm = True
                elif isinstance(child, ast.Break):
                    has_rhythm = True # Breaks are fine, it's not truly infinite

            if not has_rhythm:
                report_error(self.filename, node.lineno, "Infinite loop detected without Rhythm (Pulse/Chronos). Use 'pulse.wait()' or 'chronos.tick()'.")

        self.generic_visit(node)

    def visit_Call(self, node):
        # Check 2: The Time Check
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'time' and node.func.attr == 'sleep':
                report_error(self.filename, node.lineno, "Direct 'time.sleep()' detected. Use 'self.chronos.wait()' or 'time_tools.sleep()' to respect Relative Time.")

        # Check 4: The Identity Check (print vs logger)
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            if "scripts/" not in self.filename and "tests/" not in self.filename:
                # We allow print in scripts and tests, but not in Core
                report_warning(self.filename, node.lineno, "Usage of 'print()' in Core. Use 'logger.info()' to preserve the Memory Stream.")

        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # Check 3: The Intent Check
        if not ast.get_docstring(node):
            report_warning(self.filename, node.lineno, f"Class '{node.name}' has no docstring. What is its Purpose?")
        self.generic_visit(node)

def scan_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)
        visitor = PhilosophyVisitor(filepath)
        visitor.visit(tree)
    except SyntaxError:
        pass # Ignore syntax errors, linter deals with that
    except Exception as e:
        print(f"Failed to scan {filepath}: {e}")

def main():
    print(f"{GREEN}Starting Philosophical Verification...{RESET}")
    root_dir = "Core"

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                scan_file(os.path.join(root, file))

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)

    for w in WARNINGS:
        print(w)

    print("-" * 50)

    for e in VIOLATIONS:
        print(e)

    if VIOLATIONS:
        print(f"\n{RED}FAILED: The Spirit of the Code is broken. Fix the errors above.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}PASSED: The Code Resonates with the Law.{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()
