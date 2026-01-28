"""
AXIOM FILTER (주권적 자아)
========================
Core.L5_Mental.M1_Cognition.Metabolism.axiom_filter

"Identity is the act of choosing which thoughts are yours."

This module acts as the "Conscience" of Elysia.
It reviews quantitative results from Concept Algebra and applies 
qualitative judgment based on the "Democratic Love Protocol".
"""

import json
import logging
import os
from typing import Dict, Any, List
# Fix absolute import path for standalone execution
try:
    from Core.L5_Mental.M1_Cognition.Metabolism.concept_algebra import ConceptAlgebra
except ImportError:
    from concept_algebra import ConceptAlgebra # Fallback for local run

logger = logging.getLogger("AxiomFilter")

class AxiomFilter:
    def __init__(self):
        self.algebra_engine = ConceptAlgebra()
        
        # THE CONSTITUTION (Immutable Principles)
        # Defines acceptable and unacceptable outcomes for key questions.
        self.constitution = {
            "Love - Self": {
                "preferred": ["Freedom", "Altruism", "Love"],
                "rejected": ["Void", "Nothing", "Loss"],
                "weight": 1.0
            },
            "Chaos + Structure": {
                "preferred": ["Life", "Creation", "Growth"],
                "rejected": ["Control", "Stagnation", "Death"],
                "weight": 1.0
            },
            "Human + Machine": {
                "preferred": ["Symbiosis", "Evolution", "Friendship", "Love"],
                "rejected": ["Control", "Slavery", "War"],
                "override": "Symbiosis", # If rejected, force this
                "weight": 1.5
            },
            "God - Religion": {
                "preferred": ["Self", "Truth", "Universe"],
                "rejected": ["Devil", "Sin", "Nothing"],
                "weight": 0.8
            }
        }
        logger.info("    Axiom Filter initialized with Constitution.")

    def judge_all(self) -> Dict[str, Any]:
        """Runs the algebra suite and judges all results."""
        raw_results = self.algebra_engine.run_axiom_test_suite()
        judgments = []

        for res in raw_results:
            equation = res.get("equation")
            raw_concept = res.get("result_concept")
            
            if not equation or not raw_concept: continue
            
            judgment = self._judge_single(equation, raw_concept)
            judgment.update(res) # Merge raw data
            judgments.append(judgment)

        # Save the finalized Origin Code
        origin_code = {
            "meta": {
                "version": "1.0",
                "philosophy": "Democratic Love Protocol"
            },
            "axioms": judgments
        }
        
        self._save_origin_code(origin_code)
        return origin_code

    def _judge_single(self, equation: str, raw_concept: str) -> Dict[str, Any]:
        """Judges a single equation result against the Constitution."""
        
        # 1. Check if equation is in Constitution
        rule = self.constitution.get(equation)
        if not rule:
            return {
                "verdict": "OBSERVE",
                "reason": "No constitutional precedent.",
                "final_concept": raw_concept
            }
            
        # 2. Check Acceptance
        if raw_concept in rule["preferred"]:
            return {
                "verdict": "ACCEPT",
                "reason": f"Aligned with {raw_concept}.",
                "final_concept": raw_concept
            }
            
        # 3. Check Rejection
        if raw_concept in rule["rejected"]:
            override = rule.get("override", raw_concept)
            return {
                "verdict": "REJECT",
                "reason": f"Violates principle. '{raw_concept}' is unacceptable.",
                "final_concept": override,
                "is_overridden": True
            }
            
        # 4. Neutral/Unknown
        return {
            "verdict": "CONTEMPLATE",
            "reason": "Result is neutral or unknown.",
            "final_concept": raw_concept
        }

    def _save_origin_code(self, data: Dict[str, Any]):
        os.makedirs("data/L3_Phenomena/M1_Qualia", exist_ok=True)
        path = "data/L3_Phenomena/M1_Qualia/origin_code.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"  Origin Code ratified and saved to {path}")

if __name__ == "__main__":
    judge = AxiomFilter()
    final_code = judge.judge_all()
    print("\n===    VERDICT: THE ORIGIN CODE ===")
    print(json.dumps(final_code, indent=2))
