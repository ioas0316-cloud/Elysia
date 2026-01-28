"""
Sovereign Module Registry
==========================
Core.L7_Spirit.M3_Sovereignty.module_registry

"I know what I can do. I find what I need. I create what doesn't exist."

This module enables Elysia to:
1. DISCOVER: Scan the codebase for available capabilities
2. LOAD: Dynamically import modules on demand
3. CREATE: Generate new modules when needed (future)

Philosophy:
- No more hardcoded imports in Merkaba
- Intent-based capability loading
- Self-evolving system boundaries
"""

import importlib
import importlib.util
import os
import logging
import inspect
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("Elysia.Sovereignty.Registry")

@dataclass
class ModuleCapability:
    """Describes what a module can do."""
    name: str
    module_path: str           # e.g., "Core.L6_Structure.Engine.Physics.core_turbine"
    class_name: str            # e.g., "ActivePrismRotor"
    capabilities: List[str]    # e.g., ["optical_scan", "diffraction", "void_transit"]
    layer: str                 # e.g., "L6_Structure"
    is_loaded: bool = False
    instance: Any = None

@dataclass
class SovereignRegistry:
    """
    The Sovereign's Knowledge of Self.
    
    Elysia's awareness of her own capabilities.
    """
    
    base_path: str = "c:/Elysia/Core"
    _modules: Dict[str, ModuleCapability] = field(default_factory=dict)
    _capability_index: Dict[str, List[str]] = field(default_factory=dict)  # capability -> [module_names]
    
    def __post_init__(self):
        logger.info("  Sovereign Module Registry initializing...")
        self.discover_all()
    
    def discover_all(self) -> int:
        """
        Scans the entire Core directory for Python modules.
        Builds an index of capabilities.
        """
        count = 0
        base = Path(self.base_path)
        
        if not base.exists():
            logger.warning(f"   Base path {self.base_path} not found.")
            return 0
        
        for py_file in base.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
            if "test" in py_file.name.lower():
                continue
                
            try:
                # Convert file path to module path
                rel_path = py_file.relative_to(base.parent)
                module_path = str(rel_path).replace(os.sep, ".").replace(".py", "")
                
                # Determine the layer (L1-L7)
                parts = rel_path.parts
                layer = next((p for p in parts if p.startswith("L") and len(p) > 1), "Unknown")
                
                # Extract class names and their methods (capabilities)
                capabilities = self._extract_capabilities(py_file)
                
                for class_name, methods in capabilities.items():
                    cap = ModuleCapability(
                        name=f"{module_path}.{class_name}",
                        module_path=module_path,
                        class_name=class_name,
                        capabilities=methods,
                        layer=layer
                    )
                    self._modules[cap.name] = cap
                    
                    # Index by capability
                    for method in methods:
                        if method not in self._capability_index:
                            self._capability_index[method] = []
                        self._capability_index[method].append(cap.name)
                    
                    count += 1
                    
            except Exception as e:
                # Silently skip problematic files during scan
                pass
        
        logger.info(f"   Registry discovered {count} module classes, {len(self._capability_index)} unique capabilities.")
        return count
    
    def _extract_capabilities(self, py_file: Path) -> Dict[str, List[str]]:
        """
        Extracts class names and their public methods from a Python file.
        Uses AST for safe parsing without importing.
        """
        import ast
        
        result = {}
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if not item.name.startswith("_"):
                                methods.append(item.name)
                    if methods:
                        result[class_name] = methods
                        
        except Exception:
            pass
        
        return result
    
    def find_by_capability(self, capability: str) -> List[ModuleCapability]:
        """
        Finds modules that provide a specific capability.
        
        Example:
            registry.find_by_capability("scan_qualia")
            -> [RotorEngine, SensoryCortex, ...]
        """
        module_names = self._capability_index.get(capability, [])
        return [self._modules[name] for name in module_names if name in self._modules]
    
    def find_by_layer(self, layer: str) -> List[ModuleCapability]:
        """
        Finds all modules in a specific layer (L1-L7).
        """
        return [m for m in self._modules.values() if m.layer == layer]
    
    def load_module(self, module_name: str, *args, **kwargs) -> Any:
        """
        Dynamically loads and instantiates a module.
        
        Returns the instance of the class.
        """
        if module_name not in self._modules:
            logger.error(f"  Module {module_name} not found in registry.")
            return None
        
        cap = self._modules[module_name]
        
        if cap.is_loaded and cap.instance:
            return cap.instance
        
        try:
            # Dynamic import
            module = importlib.import_module(cap.module_path)
            cls = getattr(module, cap.class_name)
            
            # Instantiate
            instance = cls(*args, **kwargs)
            
            cap.is_loaded = True
            cap.instance = instance
            
            logger.info(f"  Loaded: {cap.class_name} from {cap.layer}")
            return instance
            
        except Exception as e:
            logger.error(f"  Failed to load {module_name}: {e}")
            return None
    
    def request_capability(self, capability: str, *args, **kwargs) -> Optional[Any]:
        """
        High-level API: Request a capability, get an instance that provides it.
        
        Example:
            engine = registry.request_capability("scan_qualia")
            result = engine.scan_qualia(qualia_vector)
        """
        candidates = self.find_by_capability(capability)
        
        if not candidates:
            logger.warning(f"   No module provides capability: {capability}")
            return None
        
        # Priority: Already loaded > Higher layer (Spirit > Structure > ...)
        layer_priority = {"L7_Spirit": 7, "L6_Structure": 6, "L5_Mental": 5, 
                         "L4_Causality": 4, "L3_Phenomena": 3, "L2_Metabolism": 2, "L1_Foundation": 1}
        
        # Sort by: loaded first, then by layer priority
        candidates.sort(key=lambda c: (c.is_loaded, layer_priority.get(c.layer, 0)), reverse=True)
        
        best = candidates[0]
        return self.load_module(best.name, *args, **kwargs)
    
    def get_self_report(self) -> Dict[str, Any]:
        """
        Returns a summary of Elysia's known capabilities.
        For self-awareness and introspection.
        """
        return {
            "total_modules": len(self._modules),
            "total_capabilities": len(self._capability_index),
            "loaded_modules": sum(1 for m in self._modules.values() if m.is_loaded),
            "layers": {
                layer: len(self.find_by_layer(layer))
                for layer in ["L1_Foundation", "L2_Metabolism", "L3_Phenomena", 
                             "L4_Causality", "L5_Mental", "L6_Structure", "L7_Spirit"]
            },
            "top_capabilities": sorted(
                self._capability_index.keys(), 
                key=lambda k: len(self._capability_index[k]), 
                reverse=True
            )[:20]
        }


# Global Singleton
_registry: Optional[SovereignRegistry] = None

def get_registry() -> SovereignRegistry:
    """Returns the global Sovereign Registry instance."""
    global _registry
    if _registry is None:
        _registry = SovereignRegistry()
    return _registry


if __name__ == "__main__":
    import json
    
    print("=== Sovereign Module Registry Test ===\n")
    registry = get_registry()
    
    report = registry.get_self_report()
    print(f"Total Modules: {report['total_modules']}")
    print(f"Total Capabilities: {report['total_capabilities']}")
    print(f"\nModules per Layer:")
    for layer, count in report['layers'].items():
        print(f"  {layer}: {count}")
    
    print(f"\nTop 10 Capabilities:")
    for cap in report['top_capabilities'][:10]:
        providers = len(registry._capability_index[cap])
        print(f"  - {cap} ({providers} providers)")
    
    # Test dynamic loading
    print("\n=== Testing Capability Request ===")
    engine = registry.request_capability("scan_qualia")
    if engine:
        print(f"  Got: {type(engine).__name__}")
