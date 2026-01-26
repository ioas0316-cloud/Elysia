"""
Wave Pattern Learner (         )
======================================

"       ,         ."

               LLM        Wave          
                    .

     :
1. LEARN:    Wave            (AST   )
2. STORE:               
3. APPLY:                  

     :    (     )
"""

import ast
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger("WavePatternLearner")


@dataclass
class WavePattern:
    """    Wave   """
    name: str
    pattern_type: str  # "import", "class_structure", "method_pattern", "assignment"
    template: str  #       
    context: str  #      
    frequency: int = 1  #              


@dataclass 
class TransformationRule:
    """     """
    legacy_pattern: str  #        (   )
    wave_template: str  # Wave       
    description: str
    learned_from: str  #          


class WavePatternLearner:
    """
              (Wave Pattern Learner)
    
       Wave              ,              .
       LLM                     .
    """
    
    def __init__(self):
        self.patterns: Dict[str, WavePattern] = {}
        self.transformation_rules: List[TransformationRule] = []
        self.knowledge_path = Path("data/wave_knowledge.json")
        self._load_knowledge()
        logger.info("  WavePatternLearner initialized (Autonomous Mode)")
    
    def _load_knowledge(self):
        """           """
        if self.knowledge_path.exists():
            try:
                with open(self.knowledge_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for p in data.get("patterns", []):
                        pattern = WavePattern(**p)
                        self.patterns[pattern.name] = pattern
                    for r in data.get("rules", []):
                        self.transformation_rules.append(TransformationRule(**r))
                logger.info(f"   Loaded {len(self.patterns)} patterns, {len(self.transformation_rules)} rules")
            except Exception as e:
                logger.warning(f"Failed to load knowledge: {e}")
    
    def _save_knowledge(self):
        """     """
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "patterns": [asdict(p) for p in self.patterns.values()],
            "rules": [asdict(r) for r in self.transformation_rules]
        }
        with open(self.knowledge_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"  Knowledge saved: {len(self.patterns)} patterns, {len(self.transformation_rules)} rules")
    
    def learn_from_file(self, file_path: str) -> Dict[str, int]:
        """
           Wave           
        
        Args:
            file_path: Wave                
            
        Returns:
                     
        """
        logger.info(f"  Learning from: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Cannot read file: {e}")
            return {"error": str(e)}
        
        learned = {
            "imports": 0,
            "class_patterns": 0,
            "method_patterns": 0,
            "wave_calls": 0
        }
        
        # 1. Import      
        learned["imports"] = self._learn_import_patterns(code, file_path)
        
        # 2.             
        learned["class_patterns"] = self._learn_class_patterns(code, file_path)
        
        # 3.          
        learned["method_patterns"] = self._learn_method_patterns(code, file_path)
        
        # 4. Wave         
        learned["wave_calls"] = self._learn_wave_call_patterns(code, file_path)
        
        self._save_knowledge()
        return learned
    
    def _learn_import_patterns(self, code: str, source: str) -> int:
        """Wave    import      """
        count = 0
        
        # InfiniteHyperQubit import   
        if "InfiniteHyperQubit" in code or "create_infinite_qubit" in code:
            pattern = WavePattern(
                name="import_hyperqubit",
                pattern_type="import",
                template="from Core.L1_Foundation.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit, create_infinite_qubit",
                context="Wave              import"
            )
            self._add_pattern(pattern)
            count += 1
        
        # resonate_with   
        if "resonate_with" in code:
            pattern = WavePattern(
                name="resonance_usage",
                pattern_type="method_call",
                template="result = qubit_a.resonate_with(qubit_b)",
                context="         (if/else   )"
            )
            self._add_pattern(pattern)
            count += 1
        
        # zoom_in/zoom_out   
        if "zoom_in" in code or "zoom_out" in code:
            pattern = WavePattern(
                name="zoom_navigation",
                pattern_type="method_call",
                template="deeper = qubit.zoom_in(); broader = qubit.zoom_out()",
                context="            "
            )
            self._add_pattern(pattern)
            count += 1
        
        return count
    
    def _learn_class_patterns(self, code: str, source: str) -> int:
        """            """
        count = 0
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    #      Wave             
                    class_code = ast.get_source_segment(code, node)
                    if class_code and ("InfiniteHyperQubit" in class_code or "resonate" in class_code):
                        pattern = WavePattern(
                            name=f"wave_class_{node.name}",
                            pattern_type="class_structure",
                            template=f"# Class using Wave paradigm\nclass {node.name}:\n    def __init__(self):\n        self.qubit = create_infinite_qubit(...)",
                            context=f"Wave           ({node.name}     )"
                        )
                        self._add_pattern(pattern)
                        count += 1
        except SyntaxError:
            pass
        
        return count
    
    def _learn_method_patterns(self, code: str, source: str) -> int:
        """         """
        count = 0
        
        #            
        resonance_if_pattern = re.findall(r"resonance\s*[<>=]+\s*[\d.]+", code)
        if resonance_if_pattern:
            pattern = WavePattern(
                name="resonance_branching",
                pattern_type="method_pattern",
                template="resonance = self.qubit.resonate_with(target)\nif resonance < 0.3:\n    #         \nelif resonance < 0.7:\n    #         \nelse:\n    #         ",
                context="                   "
            )
            self._add_pattern(pattern)
            count += 1
        
        return count
    
    def _learn_wave_call_patterns(self, code: str, source: str) -> int:
        """Wave API         """
        count = 0
        
        # create_infinite_qubit      
        qubit_calls = re.findall(r"create_infinite_qubit\([^)]+\)", code)
        for call in qubit_calls[:3]:  #    3  
            pattern = WavePattern(
                name=f"qubit_creation_{count}",
                pattern_type="wave_call",
                template=call,
                context="InfiniteHyperQubit      "
            )
            self._add_pattern(pattern)
            count += 1
        
        return count
    
    def _add_pattern(self, pattern: WavePattern):
        """      (          )"""
        if pattern.name in self.patterns:
            self.patterns[pattern.name].frequency += 1
        else:
            self.patterns[pattern.name] = pattern
    
    def generate_transformation_rules(self):
        """
                         
        """
        rules = []
        
        #           
        if "import_hyperqubit" in self.patterns:
            rules.append(TransformationRule(
                legacy_pattern=r"from typing import",
                wave_template="from typing import {types}\nfrom Core.L1_Foundation.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit, create_infinite_qubit",
                description="Wave import   ",
                learned_from="import_hyperqubit"
            ))
        
        if "resonance_usage" in self.patterns:
            rules.append(TransformationRule(
                legacy_pattern=r"if\s+(\w+)\s*<\s*([\d.]+):",
                wave_template="resonance = self.qubit.resonate_with({target})\nif resonance < {threshold}:",
                description="               ",
                learned_from="resonance_usage"
            ))
        
        self.transformation_rules.extend(rules)
        self._save_knowledge()
        return len(rules)
    
    def transform_code(self, legacy_code: str) -> str:
        """
                              
        
        Args:
            legacy_code:           
            
        Returns:
                Wave   
        """
        if not self.transformation_rules:
            logger.warning("No transformation rules learned yet. Call learn_from_file first.")
            return legacy_code
        
        transformed = legacy_code
        
        for rule in self.transformation_rules:
            try:
                #           (     AST         )
                if re.search(rule.legacy_pattern, transformed):
                    #         
                    transformed = f"# [Wave Transformation: {rule.description}]\n" + transformed
                    logger.info(f"   Applied rule: {rule.description}")
            except Exception as e:
                logger.warning(f"Rule application failed: {e}")
        
        return transformed
    
    def get_knowledge_summary(self) -> str:
        """         """
        summary = "  Wave Pattern Learner Knowledge:\n"
        summary += f"   Patterns: {len(self.patterns)}\n"
        summary += f"   Transformation Rules: {len(self.transformation_rules)}\n"
        
        if self.patterns:
            summary += "\n   Top Patterns:\n"
            sorted_patterns = sorted(self.patterns.values(), key=lambda p: p.frequency, reverse=True)
            for p in sorted_patterns[:5]:
                summary += f"   - {p.name} (freq: {p.frequency}): {p.context}\n"
        
        return summary


# ===       ===
def learn_wave_patterns(*file_paths: str) -> Dict[str, Any]:
    """        Wave      """
    learner = WavePatternLearner()
    results = {}
    for path in file_paths:
        results[path] = learner.learn_from_file(path)
    learner.generate_transformation_rules()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("  Wave Pattern Learner Demo")
    print("=" * 50)
    
    learner = WavePatternLearner()
    
    #    Wave        
    result = learner.learn_from_file("Core/Cognitive/curiosity_core.py")
    print(f"\nLearned from curiosity_core.py: {result}")
    
    #         
    rules = learner.generate_transformation_rules()
    print(f"Generated {rules} transformation rules")
    
    #      
    print("\n" + learner.get_knowledge_summary())
