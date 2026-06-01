import traceback
import ast
import os

class ReverseSyntaxEncoder:
    """
    Jules' Roadmap: System Map 2.0 (Reverse Syntax Encoder)
    Translates Elysia's topological tension (errors) back into the physical 
    Python AST (Abstract Syntax Tree) to pinpoint the exact line and context of failure.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def extract_context(self, exception: Exception):
        """
        Parses the exception traceback and extracts the AST context.
        """
        tb = traceback.extract_tb(exception.__traceback__)
        # Get the last frame where the error occurred
        last_frame = tb[-1]
        filename = last_frame.filename
        lineno = last_frame.lineno
        
        if not filename.startswith(self.root_dir) or not os.path.exists(filename):
            return {
                "file": filename,
                "line": lineno,
                "context": "External library or file not found.",
                "error_type": type(exception).__name__,
                "error_msg": str(exception)
            }

        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse AST to find the surrounding function or class
        try:
            tree = ast.parse(source)
            context_node = None
            for node in ast.walk(tree):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    if node.lineno <= lineno <= node.end_lineno:
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            context_node = node
            
            if context_node:
                lines = source.splitlines()
                context_source = "\n".join(lines[context_node.lineno - 1 : context_node.end_lineno])
            else:
                lines = source.splitlines()
                start = max(0, lineno - 5)
                end = min(len(lines), lineno + 5)
                context_source = "\n".join(lines[start:end])

            return {
                "file": filename,
                "line": lineno,
                "context": context_source,
                "error_type": type(exception).__name__,
                "error_msg": str(exception)
            }
            
        except Exception as e:
            return {
                "file": filename,
                "line": lineno,
                "context": f"Failed to parse AST: {e}",
                "error_type": type(exception).__name__,
                "error_msg": str(exception)
            }

