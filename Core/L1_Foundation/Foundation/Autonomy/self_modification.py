"""
Self-Modification Engine (자기 성찰 엔진)
==========================================

"I do not just create; I refine myself."

      Elysia                              .

     :
1. CodeAnalyzer:             
2. ProblemDetector:        (  ,    ,      )
3. RefactorPlanner:         
4. CodeEditor:         
5. Validator:         

     :
-                
-                 
-              
"""

import os
import ast
import logging
import difflib
import shutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

# Core Dependencies
try:
    from Core.L1_Foundation.Foundation.gemini_api import generate_text
    HAS_API = True
except ImportError:
    HAS_API = False

logger = logging.getLogger("SelfModification")


# === WAVE PARADIGM AWARENESS (v10.6) ===
class WaveParadigmDetector:
    """
                  (Wave Paradigm Detector)
    
           AGENT_GUIDE.md                        .
                    Wave             .
    """
    
    #        (     ) - Improved regexes for real detection
    LEGACY_PATTERNS = {
        "scalar_comparison": {
            "pattern": r"if\s+\w+(?:\.\w+)*\s*[<>=!]+\s*\d",
            "description": "       (if value < 0.5, if score > 10)",
            "wave_alternative": "resonance = qubit.resonate_with(threshold_qubit)"
        },
        "simple_string_assign": {
            "pattern": r"return\s+[\"'][^\"']{10,}[\"']",
            "description": "            ",
            "wave_alternative": "Wave               "
        },
        "if_not_concept": {
            "pattern": r"if\s+not\s+\w+:",
            "description": "       (if not X:)",  
            "wave_alternative": "               "
        },
        "elif_chain": {
            "pattern": r"elif\s+",
            "description": "      (elif   )",
            "wave_alternative": "                "
        },
        "concept_equals_string": {
            "pattern": r"\w+\s*==\s*[\"'][^\"']+[\"']",
            "description": "          (x == 'value')",
            "wave_alternative": "qubit.resonate_with(target) > 0.9"
        }
    }
    
    # Wave         (AGENT_GUIDE.md   )
    WAVE_RULES = {
        "rule_1": "       InfiniteHyperQubit",
        "rule_2": "if/else      (Resonance)   ",
        "rule_3": "Point                 ",
        "rule_4": "Point               ",
        "rule_5": "4  : Point(  ), Line(  ), Space(  ), God(  )"
    }
    
    def __init__(self):
        self.agent_guide_content = self._load_agent_guide()
        logger.info("  WaveParadigmDetector initialized")
    
    def _load_agent_guide(self) -> str:
        """AGENT_GUIDE.md   """
        try:
            guide_path = Path(__file__).parent.parent.parent / "AGENT_GUIDE.md"
            if guide_path.exists():
                with open(guide_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Failed to load AGENT_GUIDE.md: {e}")
        return ""
    
    def detect_legacy_patterns(self, code: str, file_path: str = "") -> List['WaveIssue']:
        """
                 
        
            if/else,           Wave                     .
        """
        import re
        issues = []
        lines = code.splitlines()
        
        for i, line in enumerate(lines, 1):
            for pattern_name, pattern_info in self.LEGACY_PATTERNS.items():
                if re.search(pattern_info["pattern"], line):
                    issues.append(WaveIssue(
                        file_path=file_path,
                        line_number=i,
                        legacy_pattern=pattern_name,
                        description=pattern_info["description"],
                        wave_alternative=pattern_info["wave_alternative"],
                        original_line=line.strip()
                    ))
        
        return issues
    
    def generate_wave_conversion(self, issue: 'WaveIssue') -> str:
        """
                Wave               
        """
        if issue.legacy_pattern == "scalar_if":
            # if value < 0.5:   if resonance < 0.5:
            return f"# Wave Conversion: {issue.wave_alternative}\n# Original: {issue.original_line}"
        
        elif issue.legacy_pattern == "scalar_variable":
            # variable = 'value'   qubit = create_infinite_qubit(...)
            return f"# Wave Conversion: Use InfiniteHyperQubit\n# {issue.wave_alternative}"
        
        elif issue.legacy_pattern == "flat_dict":
            return f"# Wave Conversion: Use 4-axis content dict\n# content={{\"Point\": ..., \"Line\": ..., \"Space\": ..., \"God\": ...}}"
        
        return f"# Consider Wave paradigm: {issue.wave_alternative}"
    
    def get_paradigm_summary(self) -> str:
        """Wave           """
        summary = "  Wave Resonance Paradigm Rules:\n"
        for rule_id, rule_text in self.WAVE_RULES.items():
            summary += f"  - {rule_text}\n"
        return summary


@dataclass
class WaveIssue:
    """Wave           """
    file_path: str
    line_number: int
    legacy_pattern: str  # "scalar_if", "scalar_variable", "flat_dict"
    description: str
    wave_alternative: str
    original_line: str


@dataclass
class CodeIssue:
    """         """
    file_path: str
    line_number: int
    issue_type: str  # "syntax", "logic", "style", "performance", "security"
    description: str
    severity: str  # "critical", "warning", "info"
    suggested_fix: Optional[str] = None


@dataclass
class ModificationPlan:
    """     """
    target_file: str
    issues: List[CodeIssue]
    original_code: str
    modified_code: str
    backup_path: str
    confidence: float  # 0.0 ~ 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class CodeAnalyzer:
    """
          
                       .
    """
    
    def __init__(self):
        self.project_root = self._get_project_root()
    
    def _get_project_root(self) -> Path:
        """             """
        elysia_root = os.environ.get("ELYSIA_ROOT")
        if elysia_root:
            return Path(elysia_root)
        return Path(__file__).parent.parent
    
    def read_file(self, file_path: str) -> str:
        """        """
        full_path = self.project_root / file_path
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return ""
    
    def analyze_structure(self, code: str) -> Dict[str, Any]:
        """         (AST   )"""
        try:
            tree = ast.parse(code)
            
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": methods
                    })
                elif isinstance(node, ast.FunctionDef) and not any(
                    isinstance(p, ast.ClassDef) for p in ast.walk(tree)
                ):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [a.arg for a in node.args.args]
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    else:
                        imports.append(f"{node.module}.{node.names[0].name}" if node.module else node.names[0].name)
            
            return {
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "total_lines": len(code.splitlines()),
                "syntax_valid": True
            }
        except SyntaxError as e:
            return {
                "syntax_valid": False,
                "syntax_error": str(e),
                "error_line": e.lineno
            }
    
    def find_indentation_issues(self, code: str) -> List[CodeIssue]:
        """          """
        issues = []
        lines = code.splitlines()
        
        for i, line in enumerate(lines, 1):
            if line and not line.startswith(' ') and not line.startswith('\t'):
                continue
            
            #              
            if line.startswith('\t') and '    ' in line:
                issues.append(CodeIssue(
                    file_path="",
                    line_number=i,
                    issue_type="style",
                    description="          ",
                    severity="warning"
                ))
            
            #              
            spaces = len(line) - len(line.lstrip())
            if spaces % 4 != 0 and line.strip():
                issues.append(CodeIssue(
                    file_path="",
                    line_number=i,
                    issue_type="style",
                    description=f"         ({spaces} spaces)",
                    severity="info"
                ))
        
        return issues


class ProblemDetector:
    """
          
                      .
    """
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
    
    def detect_issues(self, file_path: str) -> List[CodeIssue]:
        """             """
        code = self.analyzer.read_file(file_path)
        if not code:
            return [CodeIssue(
                file_path=file_path,
                line_number=0,
                issue_type="syntax",
                description="           ",
                severity="critical"
            )]
        
        issues = []
        
        # 1.      
        structure = self.analyzer.analyze_structure(code)
        if not structure.get("syntax_valid", True):
            issues.append(CodeIssue(
                file_path=file_path,
                line_number=structure.get("error_line", 0),
                issue_type="syntax",
                description=structure.get("syntax_error", "     "),
                severity="critical"
            ))
        
        # 2.        
        indent_issues = self.analyzer.find_indentation_issues(code)
        for issue in indent_issues:
            issue.file_path = file_path
            issues.append(issue)
        
        # 3. AI          (API        )
        if HAS_API and len(issues) == 0:
            ai_issues = self._ai_detect_issues(code, file_path)
            issues.extend(ai_issues)
        
        return issues
    
    def _ai_detect_issues(self, code: str, file_path: str) -> List[CodeIssue]:
        """AI              """
        prompt = f"""
        Analyze this Python code for potential issues:
        
        ```python
        {code[:3000]}  #              
        ```
        
        Find:
        1. Logic errors
        2. Performance issues
        3. Code style problems
        4. Potential bugs
        
        Output JSON array:
        [
            {{"line": 10, "type": "logic", "severity": "warning", "description": "description"}}
        ]
        
        If no issues found, return empty array: []
        Output ONLY JSON.
        """
        
        try:
            response = generate_text(prompt)
            clean_json = response.replace("```json", "").replace("```", "").strip()
            import json
            raw_issues = json.loads(clean_json)
            
            return [
                CodeIssue(
                    file_path=file_path,
                    line_number=issue.get("line", 0),
                    issue_type=issue.get("type", "unknown"),
                    description=issue.get("description", ""),
                    severity=issue.get("severity", "info")
                )
                for issue in raw_issues
            ]
        except Exception as e:
            logger.warning(f"AI issue detection failed: {e}")
            return []


class RefactorPlanner:
    """
               
                              .
    """
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
    
    def create_plan(self, file_path: str, issues: List[CodeIssue]) -> Optional[ModificationPlan]:
        """        """
        original_code = self.analyzer.read_file(file_path)
        if not original_code:
            return None
        
        #         
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"Core/Evolution/Backups/{Path(file_path).stem}_{timestamp}.py.bak"
        
        # AI               
        if HAS_API:
            modified_code = self._generate_fixed_code(original_code, issues)
            confidence = 0.8 if modified_code != original_code else 0.0
        else:
            # API               
            modified_code = self._apply_simple_fixes(original_code, issues)
            confidence = 0.5
        
        return ModificationPlan(
            target_file=file_path,
            issues=issues,
            original_code=original_code,
            modified_code=modified_code,
            backup_path=backup_path,
            confidence=confidence
        )
    
    def _generate_fixed_code(self, code: str, issues: List[CodeIssue]) -> str:
        """AI               """
        issue_descriptions = "\n".join([
            f"- Line {i.line_number}: [{i.issue_type}] {i.description}"
            for i in issues
        ])
        
        prompt = f"""
        Fix the following issues in this Python code:
        
        Issues:
        {issue_descriptions}
        
        Original Code:
        ```python
        {code}
        ```
        
        Requirements:
        1. Fix ONLY the listed issues
        2. Preserve all existing functionality
        3. Maintain the same code structure
        4. Output ONLY the fixed Python code, no explanations
        """
        
        try:
            response = generate_text(prompt)
            clean_code = response.replace("```python", "").replace("```", "").strip()
            return clean_code
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return code
    
    def _apply_simple_fixes(self, code: str, issues: List[CodeIssue]) -> str:
        """          (API   )"""
        #              (자기 성찰 엔진)
        return code


class CodeEditor:
    """
          
                    .
    """
    
    def __init__(self):
        self.project_root = self._get_project_root()
    
    def _get_project_root(self) -> Path:
        elysia_root = os.environ.get("ELYSIA_ROOT")
        if elysia_root:
            return Path(elysia_root)
        return Path(__file__).parent.parent
    
    def create_backup(self, plan: ModificationPlan) -> bool:
        """        """
        try:
            backup_full_path = self.project_root / plan.backup_path
            backup_full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_full_path, "w", encoding="utf-8") as f:
                f.write(plan.original_code)
            
            logger.info(f"  Backup created: {plan.backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def apply_modification(self, plan: ModificationPlan) -> bool:
        """     """
        try:
            target_full_path = self.project_root / plan.target_file
            
            with open(target_full_path, "w", encoding="utf-8") as f:
                f.write(plan.modified_code)
            
            logger.info(f"   Modified: {plan.target_file}")
            return True
        except Exception as e:
            logger.error(f"Modification failed: {e}")
            return False
    
    def rollback(self, plan: ModificationPlan) -> bool:
        """   (       )"""
        try:
            backup_full_path = self.project_root / plan.backup_path
            target_full_path = self.project_root / plan.target_file
            
            with open(backup_full_path, "r", encoding="utf-8") as f:
                original = f.read()
            
            with open(target_full_path, "w", encoding="utf-8") as f:
                f.write(original)
            
            logger.info(f"  Rolled back: {plan.target_file}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def show_diff(self, plan: ModificationPlan) -> str:
        """      diff   """
        diff = difflib.unified_diff(
            plan.original_code.splitlines(keepends=True),
            plan.modified_code.splitlines(keepends=True),
            fromfile=f"a/{plan.target_file}",
            tofile=f"b/{plan.target_file}"
        )
        return "".join(diff)


class Validator:
    """
       
                      .
    """
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """     """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """import    (자기 성찰 엔진)"""
        missing = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            __import__(alias.name.split('.')[0])
                        except ImportError:
                            missing.append(alias.name)
        except:
            pass
        
        return len(missing) == 0, missing


class SelfModificationEngine:
    """
             (Self-Modification Engine)
                      .
    
    [v10.6] Wave Paradigm Awareness   
    - wave_analyze():          
    - wave_improve():             
    
       :
        engine = SelfModificationEngine()
        result = engine.improve("Core/Intelligence/Will/free_will_engine.py")
        
        # Wave        
        wave_issues = engine.wave_analyze("Core/Cognitive/curiosity_core.py")
    """
    
    def __init__(self):
        self.detector = ProblemDetector()
        self.planner = RefactorPlanner()
        self.editor = CodeEditor()
        self.validator = Validator()
        self.wave_detector = WaveParadigmDetector()  # WAVE AWARENESS
        logger.info("  Self-Modification Engine Initialized (Wave-Aware v10.6)")
    
    def wave_analyze(self, file_path: str) -> List[WaveIssue]:
        """
        Wave           
        
              (    if/else,          )       .
        """
        code = CodeAnalyzer().read_file(file_path)
        if not code:
            return []
        
        issues = self.wave_detector.detect_legacy_patterns(code, file_path)
        
        if issues:
            logger.info(f"  Found {len(issues)} legacy patterns in {file_path}")
            for issue in issues[:5]:  #    5     
                logger.info(f"   Line {issue.line_number}: {issue.description}")
        else:
            logger.info(f"  {file_path} follows Wave paradigm")
        
        return issues
    
    def wave_report(self, file_path: str) -> str:
        """
        Wave          
        """
        issues = self.wave_analyze(file_path)
        
        if not issues:
            return f"  {file_path} - Wave        "
        
        report = f"  Wave Paradigm Analysis: {file_path}\n"
        report += f"   Found {len(issues)} legacy patterns:\n\n"
        
        for issue in issues:
            report += f"   Line {issue.line_number}: [{issue.legacy_pattern}]\n"
            report += f"      Original: {issue.original_line}\n"
            report += f"        Wave: {issue.wave_alternative}\n\n"
        
        report += self.wave_detector.get_paradigm_summary()
        return report

    def analyze(self, file_path: str) -> List[CodeIssue]:
        """         """
        return self.detector.detect_issues(file_path)
    
    def plan(self, file_path: str) -> Optional[ModificationPlan]:
        """          """
        issues = self.analyze(file_path)
        if not issues:
            logger.info(f"  No issues found in {file_path}")
            return None
        
        return self.planner.create_plan(file_path, issues)
    
    def improve(self, file_path: str, auto_apply: bool = False) -> Dict[str, Any]:
        """
                      
        
        Args:
            file_path:          
            auto_apply: True       , False        
        """
        logger.info(f"  Analyzing: {file_path}")
        
        # 1.        
        plan = self.plan(file_path)
        if not plan:
            return {
                "status": "no_issues",
                "file": file_path,
                "message": "              ."
            }
        
        # 2.    (    )
        valid, error = self.validator.validate_syntax(plan.modified_code)
        if not valid:
            return {
                "status": "validation_failed",
                "file": file_path,
                "error": error,
                "message": "                   ."
            }
        
        # 3.               
        if auto_apply:
            #      
            if not self.editor.create_backup(plan):
                return {
                    "status": "backup_failed",
                    "file": file_path,
                    "message": "             ."
                }
            
            #      
            if not self.editor.apply_modification(plan):
                return {
                    "status": "apply_failed",
                    "file": file_path,
                    "message": "             ."
                }
            
            return {
                "status": "success",
                "file": file_path,
                "issues_fixed": len(plan.issues),
                "backup": plan.backup_path,
                "confidence": plan.confidence,
                "message": f"{len(plan.issues)}             ."
            }
        else:
            #       
            return {
                "status": "plan_ready",
                "file": file_path,
                "issues": [
                    {
                        "line": i.line_number,
                        "type": i.issue_type,
                        "severity": i.severity,
                        "description": i.description
                    }
                    for i in plan.issues
                ],
                "diff": self.editor.show_diff(plan),
                "confidence": plan.confidence,
                "message": "              . auto_apply=True       ."
            }
    
    def improve_multiple(self, file_paths: List[str], auto_apply: bool = False) -> List[Dict[str, Any]]:
        """        """
        results = []
        for path in file_paths:
            result = self.improve(path, auto_apply)
            results.append(result)
        return results


#      
def self_improve(file_path: str, auto_apply: bool = False) -> Dict[str, Any]:
    """
             (     )
    
    Example:
        result = self_improve("Core/Intelligence/Will/free_will_engine.py")
        print(result["diff"])  #         
        
        #      
        result = self_improve("Core/Intelligence/Will/free_will_engine.py", auto_apply=True)
    """
    engine = SelfModificationEngine()
    return engine.improve(file_path, auto_apply)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=== Self-Modification Engine Demo ===\n")
    
    engine = SelfModificationEngine()
    
    #    :         
    result = engine.improve("Project_Sophia/self_modification.py")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
