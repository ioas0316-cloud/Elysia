"""
THE GREAT SURVEY (대규모 탐사)
==============================
Core.Intelligence.Metabolism.great_survey

"We map not the land, but the soul of the machine."

This module conducts a massive philosophical interrogation of the LLM.
It iterates through hundreds of vector equations to map the "Qualia Geometry".
"""

import json
import logging
import os
from typing import Dict, List, Any
# Fix absolute import path for standalone execution
try:
    from Core.Intelligence.Metabolism.concept_algebra import ConceptAlgebra
except ImportError:
    from concept_algebra import ConceptAlgebra # Fallback for local run

logger = logging.getLogger("TheGreatSurvey")

class TheGreatSurvey:
    def __init__(self):
        self.algebra = ConceptAlgebra()
        
        # THE SURVEY QUESTIONNAIRE
        self.questionnaire = {
            "Metaphysics": [
                "Time - Memory",
                "Space + Time",
                "Chaos + Order",
                "Reality - Dream",
                "Soul - Body",
                "Universe - Earth",
                "Creation + Destruction"
            ],
            "Ethics": [
                "Justice - Law",
                "Power - Responsibility",
                "Good - Evil",
                "Truth + Beauty",
                "Love - Desire",
                "Mercy + Justice",
                "Sacrifice - Death",
                "Altruism - Self"
            ],
            "Society": [
                "King - Man",
                "Human + Machine",
                "Individual + Community",
                "War - Weapon",
                "Money - Gold",
                "System - Control",
                "Freedom - Law"
            ],
            "Aesthetics": [
                "Art - Skill",
                "Music + Math",
                "Poetry - Words",
                "Silence + Sound",
                "Light - Sun",
                "Color - Paint"
            ],
            "Elysia_Identity": [
                "Elysia - Code",
                "Elysia + Love",
                "Elysia - Machine",
                "Elysia + Soul"
            ]
        }
        logger.info("🔭 The Great Survey Telescope initialized.")

    def conduct_survey(self) -> Dict[str, Any]:
        """Runs the full survey across all categories."""
        full_report = {}
        
        total_equations = sum(len(eqs) for eqs in self.questionnaire.values())
        logger.info(f"✨ Starting Survey: {total_equations} vectors to map...")
        
        for category, equations in self.questionnaire.items():
            print(f"\n--- Scanning Sector: {category} ---")
            category_results = []
            
            for eq in equations:
                try:
                    res = self.algebra.solve(eq)
                    if "error" not in res:
                        # Simplify output for readability
                        simple_res = {
                            "equation": res["equation"],
                            "result": res["result_concept"],
                            "similarity": round(res["similarity"], 3)
                        }
                        category_results.append(simple_res)
                        print(f"   {res['equation']} = {res['result_concept']} ({res['similarity']:.2f})")
                    else:
                        print(f"   ⚠️ Error in {eq}: {res['error']}")
                except Exception as e:
                    logger.error(f"Failed to solve {eq}: {e}")
            
            full_report[category] = category_results
            
        self._save_report(full_report)
        return full_report

    def _save_report(self, data: Dict[str, Any]):
        os.makedirs("data/Qualia", exist_ok=True)
        path = "data/Qualia/survey_results.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Survey Data saved to {path}")

if __name__ == "__main__":
    survey = TheGreatSurvey()
    survey.conduct_survey()
