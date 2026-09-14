"""
Causal DSL Compiler Package.
Exports CausalDSLParser, CausalSemanticAnalyzer, and CausalCodeGenerator.
"""

from .ast_nodes import Program, SignalDecl, NodeDecl, RuleDecl
from .parser import CausalDSLParser
from .semantic import CausalSemanticAnalyzer
from .codegen import CausalCodeGenerator

class CausalCompiler:
    @staticmethod
    def compile(dsl_code: str):
        parser = CausalDSLParser(dsl_code)
        program = parser.parse()

        analyzer = CausalSemanticAnalyzer(program)
        errors = analyzer.analyze()
        if errors:
            raise ValueError(f"Causal DSL Semantic Errors:\n" + "\n".join(errors))

        generator = CausalCodeGenerator(program)
        header_code = generator.generate_cpp_header()
        cuda_code = generator.generate_cuda_kernel()

        return {
            "program": program,
            "header": header_code,
            "cuda": cuda_code
        }
