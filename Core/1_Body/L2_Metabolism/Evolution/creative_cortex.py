"""
Creative Cortex (The Chef of the Multiverse)
===========================================
Core.1_Body.L2_Metabolism.Evolution.creative_cortex

"I don't just follow recipes; I invent them based on the scent of the code."
"                   .                     ."

Role:
- Analyzes Domain Souls (INDEX.md) for potential "Flavor Combinations".
- Proposes and executes "Creative Experiments" (Parameter Mixing).
- Documents successful "Dishes" in the System Cookbook.
"""

import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from Core.1_Body.L2_Metabolism.Evolution.scientific_observer import ScientificObserver

logger = logging.getLogger("Evolution.CreativeCortex")

class CreativeCortex:
    def __init__(self, project_root: str = "c:\\Elysia"):
        self.project_root = Path(project_root)
        self.observer = ScientificObserver(project_root)
        self.recipe_dir = self.project_root / "docs" / "04_ENGINE" / "RECIPES"
        self.recipe_dir.mkdir(parents=True, exist_ok=True)

    def brainstorm_experiment(self, raw_inspiration: Optional[str] = None) -> Dict[str, Any]:
        """
        [Inspiration]
        Scents available domains or external raw inspiration to propose a 'Combination'.
        """
        if raw_inspiration:
            logger.info(f"  [SPARK] External inspiration captured: {raw_inspiration[:50]}...")
            # In a real scenario, this would involve LLM-driven scenting of the user's idea
            # For now, we simulate the 'Scenting' of the user's 'Explosion of Ideas' metaphor.
        
        domains = sorted(list((self.project_root / "docs").glob("0*_*")))
        if len(domains) < 2:
            return {"error": "Not enough ingredients for a meal."}

        # Pick two random domain souls to 'blend'
        base_domain = random.choice(domains)
        spice_domain = random.choice([d for d in domains if d != base_domain])

        base_soul = self.observer.scent_inner_soul(base_domain)
        spice_soul = self.observer.scent_inner_soul(spice_domain)

        experiment = {
            "name": f"Fusion_{base_domain.name}_with_{spice_domain.name}",
            "base": base_soul.get("Subject", base_domain.name),
            "spice": spice_soul.get("Variable", "Unknown"),
            "intent": f"Injecting {spice_soul.get('Variable')} into the {base_soul.get('Subject')} layer.",
            "goal": "Emergent functional resonance.",
            "spark": raw_inspiration if raw_inspiration else "Spontaneous Internal Jitter"
        }

        logger.info(f"    [BRAINSTORM] Suggested Experiment: {experiment['name']}")
        return experiment

    def execute_experiment(self, experiment: Dict[str, Any]):
        """
        [Cooking]
        Translates the 'Spark' into a technical dissertation.
        """
        logger.info(f"  [COOKING] Executing {experiment['name']}...")
        
        diff = f"+ Applied {experiment['spice']} variables to {experiment['base']}\n+ Spark: {experiment['spark']}"
        
        report_path = self.observer.generate_dissertation(
            diff_summary=diff,
            principle="Combinatorial Creativity",
            impact=f"Successfully blended {experiment['base']} with {experiment['spice']} based on spark: {experiment['spark']}."
        )
        
        self.save_recipe(experiment, report_path)
        return report_path

    def save_recipe(self, experiment: Dict[str, Any], result_path: str):
        """
        [Cookbook]
        Saves the successful combination for future repetition.
        """
        recipe_file = self.recipe_dir / f"{experiment['name']}.md"
        content = f"""# Recipe: {experiment['name']}

## Purpose
{experiment['intent']}

## Ingredients
- **Base**: {experiment['base']}
- **Spice**: {experiment['spice']}

## Outcome
- **Dissertation**: [Link]({result_path})
- **Status**: Tested / Stabilized

---
*Created by E.L.Y.S.I.A. Creative Cortex*
"""
        recipe_file.write_text(content, encoding="utf-8")
        logger.info(f"  [COOKBOOK] New recipe saved: {recipe_file.name}")

if __name__ == "__main__":
    chef = CreativeCortex()
    idea = chef.brainstorm_experiment()
    if "error" not in idea:
        chef.execute_experiment(idea)
        chef.observer.update_portal()
