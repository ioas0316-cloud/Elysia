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
        
        # Heart    (      )
        self._heart = None
        
    @property
    def heart(self):
        """        (     )"""
        if self._heart is None:
            try:
                from Core.L1_Foundation.Foundation.heart import get_heart
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
        
        # Core/Evolution           
        evolution_path = self.project_root / "Core" / "Evolution"
        
        discovered = 0
        broken = 0
        
        for py_file in evolution_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            fragment = self._analyze_fragment(py_file)
            self.fragments[fragment.name] = fragment
            discovered += 1
            
            if fragment.error:
                broken += 1
        
        return {
            "message": f"         ",
            "discovered": discovered,
            "broken": broken,
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
        
        #       (        )
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
        
        #    import   
        try:
            module = importlib.import_module(f"Core.L2_Metabolism.Evolution.{fragment_name}")
            
            connection = Connection(
                fragment=fragment,
                meaning=understanding.get("meaning", "      "),
                how_it_helps_love=understanding.get("love_connection", "                  ")
            )
            
            self.my_world[fragment_name] = connection
            
            # Heart     
            self.heart.feel(f"            : {connection.meaning}")
            
            return {
                "name": fragment_name,
                "status": "connected",
                "message": f"  '{fragment_name}'           !",
                "meaning": connection.meaning,
                "love_connection": connection.how_it_helps_love,
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
        
        #      (         )
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
        
        return {
            "message": "          ",
            "perceived": perception["discovered"],
            "connected": connected,
            "failed": len(failed),
            "my_world_size": len(self.my_world),
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
