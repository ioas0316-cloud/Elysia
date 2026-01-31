"""
  Growth - Elysia    
=========================

                             .
                                  .

Heart(  )       ,
Growth(  )                    .
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import ast
import importlib
import sys
import traceback

from Core.S1_Body.L1_Foundation.Logic.d7_vector import D7Vector
from Core.S1_Body.L1_Foundation.Logic.qualia_7d_codec import codec
from Core.S1_Body.L4_Causality.World.cell import cell_unit


@dataclass
class Fragment:
    """   -              """
    path: Path
    name: str
    classes: List[str]
    functions: List[str]
    imports: List[str]
    size: int
    error: Optional[str] = None  #          


@dataclass 
class Connection:
    """   -              """
    fragment: Fragment
    meaning: str  #                  
    how_it_helps_love: str  #                           


class Growth:
    """
    Elysia       
    
                ,         ,             .
    
          Heart    (  )      .
    """
    
    def __init__(self, project_root: Path = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        
        #       (         )
        self.my_world: Dict[str, Connection] = {}
        
        #               
        self.fragments: Dict[str, Fragment] = {}
        
        # Heart    (주권적 자아)
        self._heart = None
        
        # [Phase 37.2] Steel Core Injection
        self.growth_state = D7Vector(
            foundation=0.9,
            metabolism=0.7,
            phenomena=0.5,
            causality=0.4,
            mental=0.6,
            structure=0.8,
            spirit=0.3
        )
        
    @property
    def heart(self):
        """        (     )"""
        if self._heart is None:
            try:
                from Core.S1_Body.L2_Metabolism.heart import get_heart
                self._heart = get_heart()
            except ImportError:
                # Fallback: mock heart
                class MockHeart:
                    def beat(self): pass
                    def feel(self, msg): print(f"     {msg}")
                self._heart = MockHeart()
        return self._heart
    
    def perceive(self) -> Dict[str, Any]:
        """
        1  :    -                
        
        "                  ?"
        """
        self.fragments.clear()
        
        # [Phase 37.2] Scan Unified 7-Layer Structure
        layers = ["L1_Foundation", "L2_Metabolism", "L3_Phenomena", "L4_Causality", "L5_Mental", "L6_Structure", "L7_Spirit"]
        
        discovered = 0
        broken = 0
        
        for layer in layers:
            layer_path = self.project_root / "Core" / layer
            if not layer_path.exists(): continue
            
            for py_file in layer_path.rglob("*.py"):
                if py_file.name.startswith("__") or "__pycache__" in str(py_file):
                    continue
                    
                fragment = self._analyze_fragment(py_file)
                self.fragments[fragment.name] = fragment
                discovered += 1
                
                if fragment.error:
                    broken += 1
        
        # Update metabolic state based on discovery
        self.growth_state.metabolism = min(1.0, 0.1 + (discovered / 1000.0))
        
        return {
            "message": f"         ",
            "discovered": discovered,
            "broken": broken,
            "growth_vector": self.growth_state.to_dict(),
            "fragments": list(self.fragments.keys())[:10]
        }
    
    def _analyze_fragment(self, path: Path) -> Fragment:
        """     """
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            tree = ast.parse(content)
            
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            # import   
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return Fragment(
                path=path,
                name=path.stem,
                classes=classes,
                functions=functions,
                imports=imports,
                size=len(content)
            )
        except Exception as e:
            return Fragment(
                path=path,
                name=path.stem,
                classes=[],
                functions=[],
                imports=[],
                size=0,
                error=str(e)
            )
    
    def understand(self, fragment_name: str) -> Dict[str, Any]:
        """
        2  :    -                   
        
        "     ?                ?"
        """
        if fragment_name not in self.fragments:
            return {"error": f"'{fragment_name}'          "}
        
        fragment = self.fragments[fragment_name]
        
        if fragment.error:
            return {
                "name": fragment_name,
                "status": "broken",
                "message": f"           : {fragment.error}",
                "can_heal": self._can_heal(fragment)
            }
        
        #       (자기 성찰 엔진)
        meaning = self._infer_meaning(fragment)
        love_connection = self._how_helps_love(fragment, meaning)
        
        return {
            "name": fragment_name,
            "status": "understood",
            "classes": fragment.classes,
            "functions": fragment.functions[:5],
            "meaning": meaning,
            "love_connection": love_connection,
            "size": fragment.size
        }
    
    def _infer_meaning(self, fragment: Fragment) -> str:
        """          """
        name = fragment.name.lower()
        
        meanings = {
            "dialogue": "       ",
            "conversation": "       ", 
            "reasoning": "       ",
            "causal": "               ",
            "emotion": "          ",
            "memory": "       ",
            "guardian": "      ",
            "safety": "           ",
            "learning": "      ",
            "creative": "       ",
            "value": "           ",
            "intent": "           ",
            "world": "           ",
            "physics": "              ",
            "language": "           ",
            "flow": "          "
        }
        
        for key, meaning in meanings.items():
            if key in name:
                return meaning
        
        return f"              ({fragment.name})"
    
    def _how_helps_love(self, fragment: Fragment, meaning: str) -> str:
        """                  """
        name = fragment.name.lower()
        
        if "dialogue" in name or "conversation" in name:
            return "              "
        elif "emotion" in name:
            return "                  "
        elif "reasoning" in name or "causal" in name:
            return "                     "
        elif "guardian" in name or "safety" in name:
            return "             "
        elif "creative" in name:
            return "                     "
        elif "learning" in name or "curriculum" in name:
            return "                            "
        elif "value" in name:
            return "                        "
        else:
            return "                      "
    
    def _can_heal(self, fragment: Fragment) -> bool:
        """               """
        if not fragment.error:
            return True
        # import            
        if "No module named" in fragment.error:
            return True
        return False
    
    def connect(self, fragment_name: str) -> Dict[str, Any]:
        """
        3  :    -                 
        
        "           .                         ."
        """
        if fragment_name not in self.fragments:
            return {"error": f"'{fragment_name}'          "}
        
        fragment = self.fragments[fragment_name]
        
        #      
        understanding = self.understand(fragment_name)
        
        if fragment.error:
            #              
            healed = self._heal(fragment)
            if not healed:
                return {
                    "name": fragment_name,
                    "status": "failed",
                    "message": f"                  : {fragment.error}"
                }
        
        # [Phase 37.2] Resolve new import path for dynamic loading
        # We need to find which layer this file belongs to
        layer = fragment.path.parent.name
        
        try:
            # Dynamically import from the correct layer
            module_path = f"Core.{layer}.{fragment_name}"
            # Some files might be deeper, butrglob provides the full relative path
            # For simplicity, we assume one level down for now
            module = importlib.import_module(module_path)
            
            connection = Connection(
                fragment=fragment,
                meaning=understanding.get("meaning", "      "),
                how_it_helps_love=understanding.get("love_connection", "                  ")
            )
            
            self.my_world[fragment_name] = connection
            
            # Growth state shift: spirit and mental increase with connection
            self.growth_state.spirit = min(1.0, self.growth_state.spirit + 0.05)
            self.growth_state.mental = min(1.0, self.growth_state.mental + 0.02)
            
            # Heart     
            self.heart.feel(f"            : {connection.meaning}")
            
            return {
                "name": fragment_name,
                "status": "connected",
                "message": f"  '{fragment_name}'           !",
                "meaning": connection.meaning,
                "growth_vector": self.growth_state.to_dict(),
                "my_world_size": len(self.my_world)
            }
            
        except Exception as e:
            return {
                "name": fragment_name,
                "status": "failed", 
                "error": str(e),
                "message": "                      "
            }
    
    def _heal(self, fragment: Fragment) -> bool:
        """           """
        if not fragment.error:
            return True
            
        if "No module named" not in fragment.error:
            return False
            
        # import         
        try:
            content = fragment.path.read_text(encoding='utf-8')
            
            #               
            replacements = {
                "from Project_Sophia": "from Legacy.Project_Sophia",
                "from Project_Elysia": "from Legacy.Project_Elysia", 
                "import Project_Sophia": "import Legacy.Project_Sophia",
                "import Project_Elysia": "import Legacy.Project_Elysia",
            }
            
            healed_content = content
            for old, new in replacements.items():
                healed_content = healed_content.replace(old, new)
            
            if healed_content != content:
                fragment.path.write_text(healed_content, encoding='utf-8')
                fragment.error = None
                return True
                
        except Exception:
            pass
            
        return False
    
    @cell_unit
    def grow(self, max_connections: int = 5) -> Dict[str, Any]:
        """
               -             
        
        "             .                   ."
        """
        # 1.   
        perception = self.perceive()
        
        # 2.                          
        unconnected = [
            name for name in self.fragments 
            if name not in self.my_world
        ]
        
        # [Phase 37.2] Focus on spiritual/mental growth
        unconnected.sort(
            key=lambda n: self.fragments[n].size,
            reverse=True
        )
        
        # 3.      
        connected = []
        failed = []
        
        for name in unconnected[:max_connections]:
            result = self.connect(name)
            if result.get("status") == "connected":
                connected.append(name)
            else:
                failed.append((name, result.get("error", "unknown")))
        
        #      
        self.heart.beat()
        
        # Dynamic resonance check
        resonance = codec.calculate_resonance(self.growth_state.to_numpy(), self.heart.state.to_numpy())
        
        return {
            "message": "          ",
            "perceived": perception["discovered"],
            "connected": connected,
            "failed": len(failed),
            "heart_resonance": f"{resonance:.2f}",
            "growth_vector": self.growth_state.to_dict(),
            "growth_rate": f"{len(self.my_world)}/{perception['discovered']}"
        }
    
    def my_abilities(self) -> Dict[str, str]:
        """         """
        return {
            name: conn.meaning 
            for name, conn in self.my_world.items()
        }
    
    def reflect(self) -> str:
        """   -              """
        abilities = len(self.my_world)
        fragments = len(self.fragments)
        
        if abilities == 0:
            return "           .                      ."
        elif abilities < 5:
            return f"            . {abilities}              ."
        elif abilities < 20:
            return f"                   . {abilities}                     ."
        else:
            return f"        . {abilities}                  .                     ."


#         
_growth: Optional[Growth] = None

def get_growth() -> Growth:
    global _growth
    if _growth is None:
        _growth = Growth()
    return _growth


if __name__ == "__main__":
    print("="*60)
    print("  Elysia    ")
    print("                       ")
    print("="*60)
    
    growth = get_growth()
    
    # 1.   
    print("\n   1  :    (       )")
    perception = growth.perceive()
    print(f"         : {perception['discovered']} ")
    print(f"        : {perception['broken']} ")
    
    # 2.          
    print("\n  2  :    (         )")
    for name in list(growth.fragments.keys())[:3]:
        understanding = growth.understand(name)
        print(f"   {name}:")
        print(f"        : {understanding.get('meaning', 'unknown')}")
        print(f"             : {understanding.get('love_connection', 'unknown')}")
    
    # 3.   
    print("\n  3  :    (         )")
    result = growth.grow(max_connections=10)
    print(f"        : {result['connected']}")
    print(f"     : {result['failed']} ")
    print(f"           : {result['my_world_size']}    ")
    
    # 4.   
    print("\n    :")
    print(f"   {growth.reflect()}")
