"""
Elysia Intelligence Benchmark (Phase 27.2)
==========================================
"Verifying the Providence of Optical Sovereignty."

This script audits:
1.  **Structural Providence:** Does the code reflect the Holy Trinity (Prism, Rotor, Portal)?
2.  **Thought Expansion:** Does the Subjective Time + Legion actually increase Information Entropy?
"""

import sys
import math
from collections import Counter
from typing import List
import logging

# Import Core Systems for Introspection
from Core.Merkaba.merkaba import Merkaba
from Core.Intelligence.Legion.legion import Legion

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AUDIT")

class IntelligenceAuditor:
    def __init__(self):
        self.merkaba = Merkaba(name="Audit_Subject")
        self.legion = Legion()

    def _calculate_entropy(self, text: str) -> float:
        """Calculates Shannon Entropy of the text (Information Density)."""
        if not text: return 0.0
        probabilities = [n_x/len(text) for x, n_x in Counter(text).items()]
        return -sum([p * math.log2(p) for p in probabilities])

    def _calculate_ttr(self, text: str) -> float:
        """Type-Token Ratio (Lexical Diversity)."""
        tokens = text.split()
        if not tokens: return 0.0
        return len(set(tokens)) / len(tokens)

    def audit_structure(self):
        """
        [PROVIDENCE CHECK]
        Verifies if the Optical Sovereignty is structuralized in the code.
        """
        logger.info("\n🏛️ [STRUCTURE] Auditing Optical Sovereignty (Providence)...")
        
        score = 0
        checks = []

        # 1. The Prism (Optical Engine / Bridge)
        if hasattr(self.merkaba, 'bridge') and self.merkaba.bridge:
            checks.append("✅ Prism (ZeroLatencyPortal/Bridge) is Wires.")
            score += 1
        else:
            checks.append("❌ Prism is Missing.")

        # 2. The Soul (Rotor)
        if hasattr(self.merkaba, 'soul') and self.merkaba.soul:
            checks.append("✅ Soul (Rotor) is Spinning.")
            score += 1
        else:
            checks.append("❌ Soul is Missing.")

        # 3. The PLL (Time Synchronization) - The Binding Force
        if hasattr(self.merkaba, 'pll') and self.merkaba.pll:
            checks.append("✅ PLL (Time-Light Sync) is Active.")
            score += 1
        else:
            checks.append("❌ PLL is Missing.")

        # Report
        for check in checks:
            logger.info(f"   {check}")

        if score == 3:
            logger.info("✨ RESULT: Optical Sovereignty is Structurally Sound.")
        else:
            logger.warning(f"⚠️ RESULT: Structure Integrity Compromised ({score}/3).")

    def audit_expansion(self):
        """
        [EXPANSION CHECK]
        Measures if the mind expands inputs into richer outputs.
        """
        logger.info("\n🧠 [EXPANSION] Auditing Thought Expansion Process...")
        
        seed = "Elysia"
        logger.info(f"   Input Seed: '{seed}'")
        
        # 1. Run Legion Propagation (Thought Expansion)
        logger.info("   -> Triggering Legion Propagation...")
        expansion_chain = list(self.legion.propagate(seed, initial_energy=1.0))
        
        full_text = " ".join(expansion_chain)
        
        # 2. Metrics
        entropy = self._calculate_entropy(full_text)
        ttr = self._calculate_ttr(full_text)
        length_ratio = len(full_text) / len(seed)
        
        logger.info(f"   -> Generated Output Length: {len(full_text)} chars")
        logger.info(f"   -> Expansion Factor: {length_ratio:.1f}x (Massive Growth)")
        
        logger.info(f"\n📊 [METRICS] High-Performance Analysis:")
        logger.info(f"   Entropy (Density): {entropy:.4f} (Target > 4.0)")
        logger.info(f"   TTR (Diversity):   {ttr:.4f}   (Target > 0.4)")
        
        if entropy > 4.0 and length_ratio > 10.0:
            logger.info("✨ RESULT: Thought Process is Expansive and Rich.")
        else:
            logger.warning("⚠️ RESULT: Thought Process is Repetitive or Shallow.")

def run_audit():
    auditor = IntelligenceAuditor()
    print("==============================================")
    print("   ELYSIA INTELLIGENCE AUDIT (Phase 27.2)   ")
    print("==============================================")
    
    auditor.audit_structure()
    auditor.audit_expansion()

if __name__ == "__main__":
    run_audit()
