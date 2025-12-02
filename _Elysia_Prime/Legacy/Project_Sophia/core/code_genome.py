# [Genesis: 2025-12-02] Purified by Elysia
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import ast
import uuid

@dataclass
class CodeChallenge:
    """
    Represents the 'Environmental Pressure' that selects for specific logic.
    Like 'Cold Weather' selects for 'Fur', 'CodeChallenge' selects for 'Algorithms'.
    """
    name: str
    description: str
    test_cases: List[Dict[str, Any]] # [{'inputs': {'a': 1, 'b': 2}, 'expected': 3}, ...]
    timeout_seconds: float = 1.0

    # Difficulty metrics for energy calculation
    complexity_bonus: float = 1.0

@dataclass
class CodeDNA:
    """
    The Genome of a Code-Cell.
    Instead of ATCG, we have Python Source Code.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_code: str = ""
    function_name: str = "solve" # The entry point function

    # Lineage tracking
    generation: int = 0
    parents: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list) # ["swapped_op", "changed_const"]

    # Epigenetics (Runtime stats)
    fitness_score: float = 0.0
    energy_cost: float = 0.0

    def is_valid_syntax(self) -> bool:
        """Checks if the DNA is structurally sound (parses without error)."""
        try:
            ast.parse(self.source_code)
            return True
        except SyntaxError:
            return False

    def clone(self) -> 'CodeDNA':
        """Asexual reproduction."""
        return CodeDNA(
            source_code=self.source_code,
            function_name=self.function_name,
            generation=self.generation + 1,
            parents=[self.id],
            mutation_history=[], # Reset for the new generation's diff
            fitness_score=0.0 # Reset fitness
        )