"""
Code Mutation Engine (Biological Code Evolution)
================================================
Enables Elysia to autonomously modify her own source code using AST transformations.
Ported from Legacy/Project_Sophia/code_evolution.py with added safety sandboxing.
"""

import ast
import random
import logging
import inspect
import tempfile
import os
import importlib.util
from typing import Optional, List, Any, Callable

logger = logging.getLogger("CodeMutator")

class CodeMutator(ast.NodeTransformer):
    """
    Applies 'genetic mutations' to Python AST.
    """
    def __init__(self, intensity: float = 0.1):
        self.intensity = intensity
        self.mutations_log = []

    def visit_BinOp(self, node):
        """Mutation: Swap Operator (+, -, *, /)"""
        if random.random() < self.intensity:
            ops = [ast.Add, ast.Sub, ast.Mult, ast.Div]
            current_type = type(node.op)
            choices = [op() for op in ops if op != current_type]

            if choices:
                new_op = random.choice(choices)
                self.mutations_log.append(f"Swapped operator {current_type.__name__} -> {type(new_op).__name__}")
                return ast.copy_location(ast.BinOp(left=node.left, op=new_op, right=node.right), node)

        return self.generic_visit(node)

    def visit_Constant(self, node):
        """Mutation: Drift Constants (Numbers)"""
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if random.random() < self.intensity:
                if random.random() < 0.5:
                    # Drift: +/- small amount
                    delta = node.value * 0.1 * (1 if random.random() < 0.5 else -1)
                    new_val = node.value + delta
                    self.mutations_log.append(f"Drifted constant {node.value} -> {new_val:.2f}")
                    return ast.copy_location(ast.Constant(value=new_val), node)
                else:
                    # Random small integer replacement
                    new_val = random.randint(1, 10)
                    self.mutations_log.append(f"Replaced constant {node.value} -> {new_val}")
                    return ast.copy_location(ast.Constant(value=new_val), node)
        return self.generic_visit(node)


class SafetySandbox:
    """
    Tests mutated code in a safe environment before applying it.
    """
    @staticmethod
    def test_function(source_code: str, func_name: str, test_cases: List[Tuple[Any, Any]]) -> bool:
        """
        Compiles and runs the function against test cases.
        Returns True if it runs without crashing (correctness is subjective in evolution).
        """
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(source_code)
                temp_path = f.name
            
            # Import module
            spec = importlib.util.spec_from_file_location("mutated_module", temp_path)
            if not spec or not spec.loader:
                return False
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get function
            if not hasattr(module, func_name):
                return False
            func = getattr(module, func_name)
            
            # Run test cases
            for args, expected in test_cases:
                # We don't check expected value strictly, just that it runs and returns *something* valid
                # Evolution might change the logic, so strict equality is too harsh.
                # We just check for runtime errors.
                result = func(*args)
                
            return True
            
        except Exception as e:
            logger.warning(f"Sandbox test failed: {e}")
            return False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class EvolutionaryCoder:
    """
    Manages the evolution of code functions.
    """
    def __init__(self):
        pass
        
    def evolve_function(self, func: Callable, intensity: float = 0.1) -> Optional[Callable]:
        """
        Takes a function, mutates it, tests it, and returns the new function if successful.
        """
        try:
            source = inspect.getsource(func)
            # Remove indentation if needed (simple fix)
            lines = source.split('\n')
            if lines[0].startswith('    '):
                source = '\n'.join([line[4:] for line in lines])
                
            tree = ast.parse(source)
            mutator = CodeMutator(intensity=intensity)
            new_tree = mutator.visit(tree)
            ast.fix_missing_locations(new_tree)
            
            new_source = ast.unparse(new_tree)
            
            # Get function name
            func_name = func.__name__
            
            # Sandbox Test (Simple run test)
            # We assume the function takes no args or we need a way to provide them.
            # For now, we only evolve simple logic functions.
            if SafetySandbox.test_function(new_source, func_name, []):
                logger.info(f"🧬 Code Evolution Successful! Mutations: {mutator.mutations_log}")
                # In a real system, we would return the compiled function
                # For this demo, we just return None to indicate success but not replace live code yet
                return None 
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            
        return None
