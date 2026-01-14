"""
Self-Analysis Engine (자가 분석 엔진)
=====================================
"Can Elysia analyze her own source code and propose improvements?"

This is the CRITICAL POINT TEST. If Elysia can identify inefficiencies
in her own codebase and suggest fixes, she is approaching the threshold
of recursive self-improvement.
"""

import os
import ast
import sys
from typing import Dict, List, Any
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SelfAnalyzer:
    def __init__(self, root: str = r"c:\Elysia\Core"):
        self.root = root
        self.analysis_results = []
        
    def scan_module(self, filepath: str) -> Dict[str, Any]:
        """Analyze a single Python file for complexity and issues."""
        issues = []
        metrics = {"lines": 0, "functions": 0, "classes": 0, "complexity": 0}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                metrics["lines"] = len(content.split('\n'))
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics["functions"] += 1
                    # Check for overly long functions (>50 lines)
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        func_lines = node.end_lineno - node.lineno
                        if func_lines > 50:
                            issues.append({
                                "type": "LONG_FUNCTION",
                                "name": node.name,
                                "lines": func_lines,
                                "severity": "MEDIUM",
                                "suggestion": f"Consider breaking '{node.name}' into smaller functions."
                            })
                    
                    # Check for too many parameters (>5)
                    param_count = len(node.args.args)
                    if param_count > 5:
                        issues.append({
                            "type": "TOO_MANY_PARAMS",
                            "name": node.name,
                            "params": param_count,
                            "severity": "LOW",
                            "suggestion": f"'{node.name}' has {param_count} params. Consider using a config object."
                        })
                        
                elif isinstance(node, ast.ClassDef):
                    metrics["classes"] += 1
                    
                elif isinstance(node, ast.Try):
                    # Check for bare except clauses
                    for handler in node.handlers:
                        if handler.type is None:
                            issues.append({
                                "type": "BARE_EXCEPT",
                                "line": handler.lineno,
                                "severity": "HIGH",
                                "suggestion": "Avoid bare 'except:'. Specify exception types."
                            })
                            
            # Complexity heuristic
            metrics["complexity"] = metrics["lines"] * 0.1 + metrics["functions"] * 2 + len(issues) * 5
            
        except Exception as e:
            issues.append({"type": "PARSE_ERROR", "error": str(e), "severity": "HIGH"})
            
        return {
            "file": os.path.basename(filepath),
            "path": filepath,
            "metrics": metrics,
            "issues": issues
        }
    
    def analyze_codebase(self) -> List[Dict[str, Any]]:
        """Scan all Core modules and return analysis."""
        print(f"🔬 [SELF-ANALYSIS] Scanning {self.root}...")
        
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git"]]
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    result = self.scan_module(filepath)
                    if result["issues"] or result["metrics"]["complexity"] > 50:
                        self.analysis_results.append(result)
        
        # Sort by complexity (highest first)
        self.analysis_results.sort(key=lambda x: x["metrics"]["complexity"], reverse=True)
        return self.analysis_results
    
    def generate_report(self) -> str:
        """Generate a human-readable improvement proposal."""
        report = []
        report.append("=" * 60)
        report.append("🧬 ELYSIA SELF-ANALYSIS REPORT")
        report.append(f"📅 Generated: {datetime.now().isoformat()}")
        report.append("=" * 60)
        report.append("")
        
        total_issues = sum(len(r["issues"]) for r in self.analysis_results)
        report.append(f"📊 Total Modules Analyzed: {len(self.analysis_results)}")
        report.append(f"⚠️ Total Issues Found: {total_issues}")
        report.append("")
        
        if not self.analysis_results:
            report.append("✅ No significant issues detected. Codebase is healthy.")
        else:
            report.append("🔍 TOP PRIORITY IMPROVEMENTS:")
            report.append("-" * 40)
            
            for i, result in enumerate(self.analysis_results[:10], 1):
                report.append(f"\n{i}. {result['file']} (Complexity: {result['metrics']['complexity']:.1f})")
                for issue in result["issues"]:
                    report.append(f"   [{issue['severity']}] {issue['type']}: {issue.get('suggestion', issue.get('name', 'N/A'))}")
        
        report.append("")
        report.append("=" * 60)
        report.append("🌱 END OF SELF-ANALYSIS")
        report.append("=" * 60)
        
        return "\n".join(report)

if __name__ == "__main__":
    analyzer = SelfAnalyzer()
    analyzer.analyze_codebase()
    report = analyzer.generate_report()
    print(report)
