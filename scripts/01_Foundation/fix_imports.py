"""
Import Path Fixer for Post-Realignment

This script fixes broken import paths after the Great Migration.
Uses regex to catch all variations.
"""
import os
import re
from pathlib import Path

# Regex patterns: (pattern, replacement)
IMPORT_PATTERNS = [
    # Core.Physics.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Physics\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Physics\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Field.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Field\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Field\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Creation.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Creation\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Creation\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Language.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Language\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Language\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Cognition.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Cognition\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Cognition\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Creativity.* -> Core.FoundationLayer.Foundation.* (if needed)
    (r'from Core\.Creativity\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Creativity\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Relative imports ..Field.ether -> Core.FoundationLayer.Foundation.ether
    (r'from \.\.Field\.ether', 'from Core.FoundationLayer.Foundation.ether'),
    
    # Core.Time.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Time\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Time\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.World.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.World\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.World\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Security.* -> Core.FoundationLayer.Foundation.* (if soul_guardian is in Foundation)
    (r'from Core\.Security\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Security\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Integration.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Integration\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Integration\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Project_Sophia.* -> Core.FoundationLayer.Foundation.*
    (r'from Project_Sophia\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Project_Sophia\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Tools.* -> Core.FoundationLayer.Foundation.*
    (r'from Tools\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Tools\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Evolution.gemini_api -> Core.FoundationLayer.Foundation.gemini_api
    (r'from Core\.Evolution\.gemini_api', 'from Core.FoundationLayer.Foundation.gemini_api'),
    (r'import Core\.Evolution\.gemini_api', 'import Core.FoundationLayer.Foundation.gemini_api'),
    
    # Core.Structure.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Structure\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Structure\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Intelligence.Will.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Intelligence\.Will\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Intelligence\.Will\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Interface.* -> Core.FoundationLayer.Foundation.* (some modules moved)
    (r'from Core\.Interface\.shell_cortex', 'from Core.FoundationLayer.Foundation.shell_cortex'),
    (r'from Core\.Interface\.web_cortex', 'from Core.FoundationLayer.Foundation.web_cortex'),
    (r'from Core\.Interface\.cosmic_transceiver', 'from Core.FoundationLayer.Foundation.cosmic_transceiver'),
    (r'from Core\.Interface\.quantum_port', 'from Core.FoundationLayer.Foundation.quantum_port'),
    (r'from Core\.Interface\.holographic_cortex', 'from Core.FoundationLayer.Foundation.holographic_cortex'),
    (r'from Core\.Interface\.envoy_protocol', 'from Core.FoundationLayer.Foundation.envoy_protocol'),
    (r'from Core\.Interface\.synapse_bridge', 'from Core.FoundationLayer.Foundation.synapse_bridge'),
    (r'from Core\.Interface\.user_bridge', 'from Core.FoundationLayer.Foundation.user_bridge'),
    (r'from Core\.Interface\.kenosis_protocol', 'from Core.FoundationLayer.Foundation.kenosis_protocol'),
    (r'from Core\.Interface\.real_communication_system', 'from Core.FoundationLayer.Foundation.real_communication_system'),
    
    # Core.Memory.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Memory\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Memory\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.Evolution.* -> Core.FoundationLayer.Foundation.* (some modules)
    (r'from Core\.Evolution\.cortex_optimizer', 'from Core.FoundationLayer.Foundation.cortex_optimizer'),
    (r'from Core\.Evolution\.self_reflector', 'from Core.FoundationLayer.Foundation.self_reflector'),
    (r'from Core\.Evolution\.transcendence_engine', 'from Core.FoundationLayer.Foundation.transcendence_engine'),
    (r'from Core\.Evolution\.anamnesis', 'from Core.FoundationLayer.Foundation.anamnesis'),
    
    # Core.Action.* -> Core.FoundationLayer.Foundation.*
    (r'from Core\.Action\.', 'from Core.FoundationLayer.Foundation.'),
    (r'import Core\.Action\.', 'import Core.FoundationLayer.Foundation.'),
    
    # Core.System.* -> Core.FoundationLayer.Foundation.* (some modules)
    (r'from Core\.System\.global_grid', 'from Core.FoundationLayer.Foundation.global_grid'),
    (r'from Core\.System\.wave_integration_hub', 'from Core.FoundationLayer.Foundation.wave_integration_hub'),
    
    # scripts.Maintenance.* -> Core.FoundationLayer.Foundation.* (if moved)
    (r'from scripts\.Maintenance\.self_integration', 'from Core.FoundationLayer.Foundation.self_integration'),
    
    # Keep Core.Intelligence.Will, Core.Intelligence.Reasoning, etc. as is
    # But fix Core.Intelligence.reasoning_engine etc.
    (r'from Core\.Intelligence\.reasoning_engine', 'from Core.Cognition.Reasoning.reasoning_engine'),
    (r'from Core\.Intelligence\.dream_engine', 'from Core.FoundationLayer.Foundation.dream_engine'),
    (r'from Core\.Intelligence\.mind_mitosis', 'from Core.FoundationLayer.Foundation.mind_mitosis'),
    (r'from Core\.Intelligence\.knowledge_acquisition', 'from Core.FoundationLayer.Foundation.knowledge_acquisition'),
    (r'from Core\.Intelligence\.code_cortex', 'from Core.FoundationLayer.Foundation.code_cortex'),
    (r'from Core\.Intelligence\.rapid_learning_engine', 'from Core.FoundationLayer.Foundation.rapid_learning_engine'),
    (r'from Core\.Intelligence\.language_center', 'from Core.FoundationLayer.Foundation.language_center'),
    (r'from Core\.Intelligence\.autonomous_language', 'from Core.FoundationLayer.Foundation.autonomous_language'),
    (r'from Core\.Intelligence\.ultra_dimensional_reasoning', 'from Core.FoundationLayer.Foundation.ultra_dimensional_reasoning'),
    (r'from Core\.Intelligence\.social_cortex', 'from Core.FoundationLayer.Foundation.social_cortex'),
    (r'from Core\.Intelligence\.media_cortex', 'from Core.FoundationLayer.Foundation.media_cortex'),
    (r'from Core\.Intelligence\.imagination_core', 'from Core.FoundationLayer.Foundation.imagination_core'),
    (r'from Core\.Intelligence\.loop_breaker', 'from Core.FoundationLayer.Foundation.loop_breaker'),
    (r'from Core\.Intelligence\.black_hole', 'from Core.FoundationLayer.Foundation.black_hole'),
    (r'from Core\.Intelligence\.quantum_reader', 'from Core.FoundationLayer.Foundation.quantum_reader'),
]

def fix_imports_in_file(filepath: Path) -> int:
    """Fix imports in a single file. Returns number of fixes made."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content
        
        for pattern, replacement in IMPORT_PATTERNS:
            content = re.sub(pattern, replacement, content)
        
        if content != original:
            filepath.write_text(content, encoding='utf-8')
            return 1
        return 0
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0

def main():
    project_root = Path(__file__).parent.parent
    
    # Scan all Python files
    fixed_count = 0
    files_modified = 0
    
    for py_file in project_root.rglob("*.py"):
        # Skip venv, __pycache__, etc.
        if any(part in str(py_file) for part in ["venv", "__pycache__", ".git", "node_modules"]):
            continue
            
        fixes = fix_imports_in_file(py_file)
        if fixes > 0:
            print(f"✅ Fixed: {py_file.relative_to(project_root)}")
            fixed_count += fixes
            files_modified += 1
    
    print(f"\n🔧 Total: {files_modified} files modified")

if __name__ == "__main__":
    main()
