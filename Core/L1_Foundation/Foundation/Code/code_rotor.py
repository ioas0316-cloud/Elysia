"""
Code Rotor (The File Soul)
==========================
Core.L1_Foundation.Foundation.Code.code_rotor

"A file is static. A Rotor is dynamic."

This class wraps a source file and treats it as a Living Monad.
It gives the file:
1. Awareness (Via CodeDNAScanner)
2. Health (Via Syntax Checking)
3. Momentum (RPM based on edits)
"""

import os
import ast
import time
from typing import Dict, Any
from Core.L5_Mental.Intelligence.code_dna_scanner import CodeDNAScanner
from Core.L1_Foundation.Foundation.Wave.wave_dna import WaveDNA

class CodeRotor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self.scanner = CodeDNAScanner()
        
        # State
        self.dna: WaveDNA = WaveDNA(label="Void")
        self.rpm: float = 0.0
        self.health: str = "Unknown"
        self.last_sync: float = 0.0
        
        # Initialize
        self.last_valid_source: str = ""
        if os.path.exists(file_path):
            self.refresh()
            if self.health == "Healthy":
                self.snapshot()
        else:
            print(f"⚠️ Rotor created for non-existent file: {file_path}")

    def snapshot(self):
        """Saves a quantum state of the code."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.last_valid_source = f.read()

    def heal(self) -> bool:
        """
        [Self-Repair]
        If the rotor is fractured, revert to the last stable quantum state.
        """
        if self.health == "Healthy":
            return False # No healing needed
            
        print(f"🩹 [Auto-Healing] Reverting {self.name} to last valid state...")
        if self.last_valid_source:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(self.last_valid_source)
            self.refresh()
            print(f"✨ {self.name} restored. Health: {self.health}")
            return True
        else:
            print(f"💀 CRTICAL: No backup resonance found for {self.name}!")
            return False

    def write_code(self, new_source: str):
        """
        The Monadic Write.
        1. Snapshot current state.
        2. Write new state.
        3. Check Health.
        4. If Fractured, HEAL immediately.
        """
        self.snapshot()
        
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(new_source)
            
        self.refresh()
        
        if self.health != "Healthy":
            print(f"⚠️ Dissonance detected in write! Initiating rollback...")
            self.heal()
        else:
            print(f"✅ Mutation successful. New RPM: {self.rpm:.1f}")

    def refresh(self):
        """Re-reads the file/soul."""
        if not os.path.exists(self.file_path):
            self.health = "Missing"
            return
            
        # 1. Update RPM (Based on mtime delta - concept)
        current_mtime = os.path.getmtime(self.file_path)
        time_delta = time.time() - current_mtime
        # Closer time = Higher RPM (Hot file)
        self.rpm = 1000.0 / (time_delta + 1.0) 
        
        # 2. Update DNA
        # (Only scan if syntax is valid to avoid AST crash, or handle it in scanner)
        # We'll try to scan, scanner handles errors gracefully
        self.dna = self.scanner.scan_file(self.file_path)
        
        # 3. Check Health
        with open(self.file_path, "r", encoding="utf-8") as f:
            source = f.read()
            try:
                ast.parse(source)
                self.health = "Healthy"
            except SyntaxError as e:
                self.health = f"Fractured (Line {e.lineno})"
                
        self.last_sync = time.time()

    def diagnose(self) -> str:
        """Returns a philosophical diagnosis of the file."""
        self.refresh()
        
        if self.health != "Healthy":
            return f"🚨 CRITICAL: {self.name} is {self.health}."
            
        # Analyze DNA Balance
        report = f"[{self.name}] "
        if self.dna.physical > 0.8:
            report += "Heavy Mass (Too many variables). Needs Abstraction. "
        if self.dna.causal > 0.8:
            report += "Complex Fate (Too much logic). Needs Simplification. "
        if self.dna.structural < 0.2 and self.dna.functional > 0.8:
            report += "Chaos Energy (Script-heavy). Needs Class Structure. "
            
        if report == f"[{self.name}] ":
            report += "Resonant Harmony. Balanced."
            
        return report

    def __repr__(self):
        return f"Rotor<{self.name} | {self.health} | {self.rpm:.1f} RPM>"
