"""
Self-Maintenance Hub (          )
==========================================

"     ,       ,         ."

                              ,
        ,            ,                      .

          :
- SelfReflector: AST         
- SystemSelfAwareness:   /     
- self_modification:         

     :
1. diagnose()               
2. identify_issues()         
3. propose_fixes()           
4. execute_with_consent()              
"""

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("SelfMaintenanceHub")

#          
try:
    from Core.1_Body.L1_Foundation.Foundation.self_reflector import SelfReflector, CodeMetrics
    HAS_REFLECTOR = True
except ImportError as e:
    HAS_REFLECTOR = False
    logger.warning(f"SelfReflector not available: {e}")

try:
    from Core.1_Body.L5_Mental.Reasoning_Core.Intelligence.system_self_awareness import SystemSelfAwareness
    HAS_AWARENESS = True
except ImportError as e:
    HAS_AWARENESS = False
    logger.warning(f"SystemSelfAwareness not available: {e}")

try:
    from Core.1_Body.L1_Foundation.Foundation.self_modification import (
        CodeAnalyzer, ProblemDetector, RefactorPlanner, 
        CodeEditor, Validator, CodeIssue, ModificationPlan
    )
    HAS_MODIFICATION = True
except ImportError as e:
    HAS_MODIFICATION = False
    logger.warning(f"SelfModification not available: {e}")


@dataclass
class SystemDiagnosis:
    """            """
    timestamp: datetime = field(default_factory=datetime.now)
    
    #       
    total_files: int = 0
    total_lines: int = 0
    total_classes: int = 0
    total_functions: int = 0
    
    #    
    bottlenecks: List[str] = field(default_factory=list)
    issues: List[Any] = field(default_factory=list)
    
    #     
    suggestions: List[str] = field(default_factory=list)
    
    #   
    health_score: float = 1.0  # 0~1, 1    
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "total_classes": self.total_classes,
            "total_functions": self.total_functions,
            "bottlenecks": self.bottlenecks,
            "issue_count": len(self.issues),
            "suggestions": self.suggestions,
            "health_score": self.health_score
        }
    
    def summary(self) -> str:
        """      """
        lines = [
            f"=== System Diagnosis ({self.timestamp.strftime('%Y-%m-%d %H:%M')}) ===",
            f"Files: {self.total_files} | Lines: {self.total_lines:,}",
            f"Classes: {self.total_classes} | Functions: {self.total_functions}",
            f"Health Score: {self.health_score:.1%}",
        ]
        
        if self.bottlenecks:
            lines.append(f"\n   Bottlenecks ({len(self.bottlenecks)}):")
            for b in self.bottlenecks[:5]:
                lines.append(f"    {b}")
        
        if self.suggestions:
            lines.append(f"\n  Suggestions ({len(self.suggestions)}):")
            for s in self.suggestions[:5]:
                lines.append(f"    {s}")
        
        return "\n".join(lines)


class SelfMaintenanceHub:
    """
                 
    
                   /  /           
    """
    
    def __init__(self, root_path: str = "c:/Elysia"):
        self.root_path = root_path
        
        #           
        self.reflector = SelfReflector(root_path) if HAS_REFLECTOR else None
        self.awareness = SystemSelfAwareness() if HAS_AWARENESS else None
        
        if HAS_MODIFICATION:
            self.analyzer = CodeAnalyzer()
            self.detector = ProblemDetector()
            self.planner = RefactorPlanner()
            self.editor = CodeEditor()
            self.validator = Validator()
        else:
            self.analyzer = None
            self.detector = None
            self.planner = None
            self.editor = None
            self.validator = None
        
        #            
        self._last_diagnosis: Optional[SystemDiagnosis] = None
        self._pending_plans: List[Any] = []
        
        logger.info("  SelfMaintenanceHub initialized")
        logger.info(f"   Reflector: {' ' if HAS_REFLECTOR else ' '}")
        logger.info(f"   Awareness: {' ' if HAS_AWARENESS else ' '}")
        logger.info(f"   Modification: {' ' if HAS_MODIFICATION else ' '}")
    
    def diagnose(self) -> SystemDiagnosis:
        """
                 
        
        Returns:
            SystemDiagnosis   
        """
        diagnosis = SystemDiagnosis()
        
        # 1.          
        if self.reflector:
            metrics_map = self.reflector.reflect_on_core()
            
            diagnosis.total_files = len(metrics_map)
            diagnosis.total_lines = sum(m.loc for m in metrics_map.values())
            diagnosis.total_classes = sum(m.classes for m in metrics_map.values())
            diagnosis.total_functions = sum(m.functions for m in metrics_map.values())
            diagnosis.bottlenecks = self.reflector.identify_bottlenecks(metrics_map)
        
        # 2.             
        if self.awareness:
            try:
                suggestions = self.awareness.suggest_improvements()
                diagnosis.suggestions = suggestions.get("suggestions", [])
            except Exception as e:
                logger.warning(f"Awareness suggestions failed: {e}")
        
        # 3.          
        #         ,               
        penalty = (len(diagnosis.bottlenecks) * 0.05) + (len(diagnosis.issues) * 0.02)
        diagnosis.health_score = max(0.0, 1.0 - penalty)
        
        self._last_diagnosis = diagnosis
        logger.info(f"  Diagnosis complete: Health={diagnosis.health_score:.1%}")
        
        return diagnosis
    
    def identify_issues(self, file_path: str = None) -> List[Any]:
        """
              
        
        Args:
            file_path:       (None     )
        """
        if not self.detector:
            return []
        
        issues = []
        
        if file_path:
            issues = self.detector.detect_issues(file_path)
        else:
            #                bottleneck    
            if self._last_diagnosis and self._last_diagnosis.bottlenecks:
                for bottleneck in self._last_diagnosis.bottlenecks[:5]:
                    # bottleneck             
                    filename = bottleneck.split(" ")[0]
                    file_issues = self.detector.detect_issues(filename)
                    issues.extend(file_issues)
        
        return issues
    
    def propose_fix(self, file_path: str, issues: List[Any] = None) -> Optional[Any]:
        """
                
        
        Args:
            file_path:      
            issues:         (None        )
        """
        if not self.planner:
            return None
        
        if issues is None:
            issues = self.identify_issues(file_path)
        
        if not issues:
            logger.info(f"No issues found in {file_path}")
            return None
        
        plan = self.planner.create_plan(file_path, issues)
        self._pending_plans.append(plan)
        
        return plan
    
    def execute_with_consent(self, plan: Any, consent: bool = False) -> bool:
        """
                      
        
        Args:
            plan:      
            consent:          
        
        Returns:
                 
        """
        if not consent:
            logger.warning("  Execution denied: User consent required")
            return False
        
        if not self.editor:
            logger.error("Editor not available")
            return False
        
        try:
            # 1.      
            self.editor.create_backup(plan)
            
            # 2.      
            self.editor.apply_modification(plan)
            
            # 3.   
            if self.validator:
                validation = self.validator.validate_syntax(plan.modified_code)
                if not validation:
                    logger.error("Validation failed, rolling back...")
                    self.editor.rollback(plan)
                    return False
            
            logger.info(f"  Modification applied to {plan.target_file}")
            return True
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            if self.editor:
                self.editor.rollback(plan)
            return False
    
    def quick_health_check(self) -> Dict[str, Any]:
        """
                 (   )
        """
        return {
            "reflector": " " if HAS_REFLECTOR else " ",
            "awareness": " " if HAS_AWARENESS else " ",
            "modification": " " if HAS_MODIFICATION else " ",
            "last_diagnosis": self._last_diagnosis.timestamp.isoformat() if self._last_diagnosis else None,
            "health_score": self._last_diagnosis.health_score if self._last_diagnosis else None,
            "pending_plans": len(self._pending_plans)
        }


#       
_hub_instance: Optional[SelfMaintenanceHub] = None

def get_maintenance_hub() -> SelfMaintenanceHub:
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = SelfMaintenanceHub()
    return _hub_instance


#    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("SelfMaintenanceHub Test")
    print("=" * 60)
    
    hub = get_maintenance_hub()
    
    #      
    print("\n  Quick Health Check:")
    for k, v in hub.quick_health_check().items():
        print(f"  {k}: {v}")
    
    #      
    print("\n  Full Diagnosis:")
    diagnosis = hub.diagnose()
    print(diagnosis.summary())
