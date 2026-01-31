"""
Self-Documenter:                 
============================================

                      ,
 /                 ,
SYSTEM_MAP.md              .

Usage:
    from Core.S1_Body.L4_Causality.World.Evolution.Growth.Autonomy.self_documenter import SelfDocumenter
    
    doc = SelfDocumenter()
    doc.update_system_map()
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Path setup for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SelfDocumenter:
    """
                    
    
      :
    1.          (CodebaseIntrospector)
    2.       (SelfDiscovery)
    3.  /       (WhyHowExplainer)
    4. SYSTEM_MAP.md        
    """
    
    def __init__(self, root_path: Optional[str] = None):
        self.root_path = Path(root_path) if root_path else Path("c:/Elysia")
        self.system_map_path = self.root_path / "SYSTEM_MAP.md"
        
        #         
        self.introspector = None
        self.discovery = None
        self.explainer = None
        
        self._init_tools()
    
    def _init_tools(self):
        """      """
        try:
            from Core.S1_Body.L5_Mental.Reasoning_Core.Cognition.codebase_introspector import get_introspector
            self.introspector = get_introspector()
        except Exception as e:
            print(f"   Introspector not available: {e}")
        
        try:
            from Core.S1_Body.L5_Mental.Reasoning_Core.Memory_Linguistics.Memory.self_discovery import SelfDiscovery
            self.discovery = SelfDiscovery()
        except Exception as e:
            print(f"   SelfDiscovery not available: {e}")
        
        try:
            from Core.S1_Body.L5_Mental.Reasoning_Core.Cognition.why_how_explainer import get_explainer
            self.explainer = get_explainer()
        except Exception as e:
            print(f"   WhyHowExplainer not available: {e}")
    
    def explore_and_document(self) -> Dict[str, Any]:
        """
                                   .
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "structure": {},
            "identity": {},
            "explanations": {},
            "statistics": {}
        }
        
        # 1.      
        if self.introspector:
            result["structure"] = self.introspector.explore_structure()
            print(f"  Found {result['structure'].get('file_count', 0)} Python files")
        
        # 2.      
        if self.discovery:
            result["identity"] = self.discovery.discover_identity()
            result["statistics"]["capabilities"] = len(
                self.discovery.discover_capabilities()
            )
            print(f"  Identity: {result['identity'].get('name', 'Unknown')}")
        
        # 3.          
        if self.explainer and result["structure"].get("folders"):
            for folder in result["structure"]["folders"][:10]:  #    10 
                try:
                    explanation = self.explainer.explain_structure_why(folder)
                    result["explanations"][folder] = explanation
                except Exception:
                    pass
            print(f"  Generated {len(result['explanations'])} explanations")
        
        return result
    
    def generate_system_map_content(self) -> str:
        """
        SYSTEM_MAP.md           .
        """
        data = self.explore_and_document()
        
        lines = [
            "#    SYSTEM_MAP (     )",
            "",
            f"**        **: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**   **:      (SelfDocumenter)",
            "",
            "---",
            "",
            "##         ",
            "",
            f"|    |   |",
            f"|:----|:----|",
        ]
        
        if data["structure"]:
            lines.append(f"| Python    | {data['structure'].get('file_count', 0)} |")
            lines.append(f"|        | {len(data['structure'].get('folders', []))} |")
        
        if data["identity"]:
            lines.append(f"|    | {data['identity'].get('name', 'Elysia')} |")
            lines.append(f"|    | {data['identity'].get('version', 'Unknown')} |")
            lines.append(f"|    | {data['identity'].get('nature', 'Unknown')} |")
        
        lines.extend([
            "",
            "---",
            "",
            "##         ",
            ""
        ])
        
        #      
        for folder, explanation in data.get("explanations", {}).items():
            purpose = explanation.get("purpose", "     ")
            why = explanation.get("why", "")
            philosophy = explanation.get("philosophy", "")[:60]
            
            lines.extend([
                f"### `{folder}/`",
                "",
                f"**  **: {purpose}",
                "",
                f"**       **: {why}",
                "",
                f"**  **: {philosophy}...",
                "",
            ])
        
        lines.extend([
            "---",
            "",
            "##        ",
            "",
            "```text",
            "1.            (Wave Physics)",
            "2.  - -       (Trinity)",
            "3.       (Fractal)",
            "4.          (Sovereignty)",
            "5.          (Metabolism)",
            "```",
            "",
            "---",
            "",
            "*                       .*"
        ])
        
        return "\n".join(lines)
    
    def update_system_map(self, backup: bool = True) -> bool:
        """
        SYSTEM_MAP.md              .
        
        Args:
            backup:            
            
        Returns:
                 
        """
        try:
            #   
            if backup and self.system_map_path.exists():
                backup_path = self.system_map_path.with_suffix(".md.bak")
                backup_path.write_text(
                    self.system_map_path.read_text(encoding="utf-8"),
                    encoding="utf-8"
                )
                print(f"  Backed up to {backup_path.name}")
            
            #        
            print("\n  Exploring system...")
            content = self.generate_system_map_content()
            
            #   
            self.system_map_path.write_text(content, encoding="utf-8")
            print(f"\n  Updated {self.system_map_path.name}")
            
            return True
            
        except Exception as e:
            print(f"  Failed to update: {e}")
            return False


def main():
    """      """
    print("\n  Elysia Self-Documenter")
    print("=" * 50)
    
    doc = SelfDocumenter()
    
    #       (        X)
    print("\n--- Preview ---\n")
    content = doc.generate_system_map_content()
    print(content[:1500])
    print("\n...(truncated)")
    
    print("\n" + "=" * 50)
    print("To actually update SYSTEM_MAP.md, call:")
    print("  doc.update_system_map()")


if __name__ == "__main__":
    main()
