"""
Logic Cortex (Symbolic Reasoning Engine)
========================================
"Logic is the skeleton of Truth."

This module provides Functional Competence:
1.  **Symbolic Solver**: Solves equations (A=B, B=C -> A=C).
2.  **Axiom Storage**: Remembers rules.
3.  **Proof Generation**: Explains "Why".
4.  **Transfer Learning**: Maps Logic Rules to Abstract Domains.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

# Use SymPy if available for heavy lifting, else internal logic
try:
    from sympy import symbols, Eq, solve
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from elysia_core import Cell

logger = logging.getLogger("LogicCortex")

@dataclass
class Axiom:
    """A fundamental rule."""
    name: str # e.g., "Transitivity"
    definition: str # e.g., "If A=B and B=C, then A=C"
    domain: str # "Math", "Ethics", "Physics"

@Cell("LogicCortex")
class LogicCortex:
    def __init__(self):
        self.axioms: Dict[str, Axiom] = {}
        self.knowledge_base: Dict[str, Any] = {} # Symbol -> Value/Expression
        self.relations: List[Tuple[str, str, str]] = [] # (Subject, Predicate, Object)
        logger.info("📐 LogicCortex Initialized (Symbolic Engine Ready)")

    def learn_axiom(self, name: str, definition: str, domain: str = "Math"):
        """Installs a new rule into the logic engine."""
        self.axioms[name] = Axiom(name, definition, domain)
        logger.info(f"Learned Axiom: {name} ({definition})")

    def define_variable(self, symbol: str, value: Any = None):
        """Sets a variable in the symbolic space."""
        self.knowledge_base[symbol] = value
        logger.info(f"Defined: {symbol} = {value}")

    def add_relation(self, subject: str, predicate: str, object: str):
        """Adds a logical relation e.g., ('A', 'equals', 'B')."""
        self.relations.append((subject, predicate, object))
        logger.info(f"Relation: {subject} {predicate} {object}")

    def register_operator(self, symbol: str, logic_lambda: callable):
        """Learns a new operation dynamically (e.g., '@' -> lambda a,b: a*b)."""
        self.knowledge_base[symbol] = logic_lambda
        logger.info(f"Learned Operator: {symbol}")

    def register_isomorphic_principle(self, name: str, behavior_map: Dict[str, callable]):
        """
        Registers a Universal Principle that behaves differently in different domains.
        Structure:
        {
            "Math": lambda a,b: a+b,
            "Language": lambda a,b: a+b (Concat),
            "Chemistry": lambda a,b: f"{a}2{b}"
        }
        """
        self.axioms[name] = Axiom(name, "Isomorphic Rule", "Universal")
        self.knowledge_base[f"principle_{name}"] = behavior_map
        logger.info(f"Learned Isomorphic Principle: {name} ({list(behavior_map.keys())})")

    def apply_principle(self, name: str, inputs: List[Any], domain: str) -> Dict[str, Any]:
        """Applies an abstract principle to a specific domain."""
        key = f"principle_{name}"
        if key not in self.knowledge_base:
            return {"status": "error", "reason": f"Principle '{name}' not known."}
        
        behavior_map = self.knowledge_base[key]
        
        # 1. Exact Domain Match
        if domain in behavior_map:
            func = behavior_map[domain]
            try:
                result = func(*inputs)
                return {"value": result, "proof": f"Applied {name} using {domain} Logic"}
            except Exception as e:
                return {"status": "error", "reason": str(e)}
                
        # 2. Fallback/Generative Logic (The "Intelligence" Part)
        # If no exact match, try to find a 'closest fit' or use a generic 'Concept Merger'
        # For simulation, we return failure to prompt learning.
        return {"status": "fail", "reason": f"System knows '{name}' but not how to apply it to '{domain}'."}

    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """
        Evaluates an expression with learned operators.
        Format: "3 @ 4"
        """
        try:
            parts = expression.split()
            if len(parts) == 3:
                a, op, b = parts
                # Resolve values
                val_a = float(self.knowledge_base.get(a, a))
                val_b = float(self.knowledge_base.get(b, b))
                
                if op in self.knowledge_base and callable(self.knowledge_base[op]):
                    result = self.knowledge_base[op](val_a, val_b)
                    return {"value": result, "proof": f"Applied {op} to {val_a}, {val_b}"}
            
            return {"status": "error", "reason": "Invalid syntax"}
        except Exception as e:
             return {"status": "error", "reason": str(e)}

    def solve(self, query: str) -> Dict[str, Any]:
        """
        Attempts to solve a query using stored axioms and relations.
        Query format: "Value of X?", "Relation between A and C?", or "Eval: 3 # 4"
        """
        # 0. Expression Evaluation (Dynamic Math)
        if query.startswith("Eval:"):
            expr = query.replace("Eval:", "").strip()
            return self.evaluate_expression(expr)

        # 1. Direct Value Lookup
        if query.startswith("Value of"):
            symbol = query.replace("Value of", "").strip(" ?")
            return self._derive_value(symbol)
            
        # 2. Transitive Relation (The "Math Test")

        if "Relation between" in query:
            parts = query.replace("Relation between", "").strip(" ?").split(" and ")
            if len(parts) == 2:
                return self._derive_relation(parts[0], parts[1])

        return {"status": "unsolvable", "reason": "Unknown query format"}

    def _derive_value(self, target: str) -> Dict:
        """Derives value using Transitivity (A=B=5)."""
        # DFS search for value
        visited = set()
        stack = [target]
        
        while stack:
            current = stack.pop()
            if current in visited: continue
            visited.add(current)
            
            # If explicit value exists
            if self.knowledge_base.get(current) is not None:
                return {
                    "value": self.knowledge_base[current],
                    "proof": f"Found direct value for {current}"
                }
                
            # Check relations for equality
            for s, p, o in self.relations:
                if p in ["equals", "=", "is"]:
                    if s == current:
                        # If Object has value?
                        val = self.knowledge_base.get(o)
                        if val is not None:
                             return {"value": val, "proof": f"{target} = {o} = {val}"}
                        stack.append(o)
                    elif o == current:
                        val = self.knowledge_base.get(s)
                        if val is not None:
                             return {"value": val, "proof": f"{target} = {s} = {val}"}
                        stack.append(s)
                        
        return {"status": "unknown"}

    def _derive_relation(self, start: str, end: str) -> Dict:
        """Finds path between two symbols."""
        # Bidirectional BFS
        # Simplified: Just check for 'equality' chain
        path = self._find_path(start, end, ["equals", "=", "is"])
        if path:
            return {
                "relation": "Equal",
                "proof": " -> ".join(path),
                "axiom_used": "Transitivity"
            }
        
        return {"relation": "Unknown"}

    def _find_path(self, start: str, end: str, predicates: List[str]) -> Optional[List[str]]:
        queue = [(start, [start])]
        visited = set()
        
        while queue:
            curr, path = queue.pop(0)
            if curr == end:
                return path
            
            if curr in visited: continue
            visited.add(curr)
            
            for s, p, o in self.relations:
                if p in predicates:
                    if s == curr and o not in visited:
                        queue.append((o, path + [o]))
                    elif o == curr and s not in visited:
                        queue.append((s, path + [s]))
        return None

    def transfer_learning(self, source_rule: str, target_domain: str) -> str:
        """
        Maps a logic rule to a new domain.
        Input: "Balance", Target: "Justice"
        Output: "Crime must equal Punishment"
        """
        if source_rule == "Balance" and target_domain == "Ethics":
            # Simple symbolic mapping for demonstration
            return "Justice Equation: Action (Crime) + Reaction (Punishment) = 0"
            
        return "Transfer failed: No mapping found."

# Singleton Access
_logic_instance = None
def get_logic_cortex():
    global _logic_instance
    if _logic_instance is None:
        _logic_instance = LogicCortex()
    return _logic_instance
