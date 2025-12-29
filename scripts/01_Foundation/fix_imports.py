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
    # Core.Physics.* -> Core.Foundation.Foundation.*
    (r'from Core\.Physics\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Physics\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Field.* -> Core.Foundation.Foundation.*
    (r'from Core\.Field\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Field\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Creation.* -> Core.Foundation.Foundation.*
    (r'from Core\.Creation\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Creation\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Language.* -> Core.Foundation.Foundation.*
    (r'from Core\.Language\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Language\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Cognition.* -> Core.Foundation.Foundation.*
    (r'from Core\.Cognition\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Cognition\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Creativity.* -> Core.Foundation.Foundation.* (if needed)
    (r'from Core\.Creativity\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Creativity\.', 'import Core.Foundation.Foundation.'),
    
    # Relative imports ..Field.ether -> Core.Foundation.Foundation.ether
    (r'from \.\.Field\.ether', 'from Core.Foundation.Foundation.ether'),
    
    # Core.Time.* -> Core.Foundation.Foundation.*
    (r'from Core\.Time\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Time\.', 'import Core.Foundation.Foundation.'),
    
    # Core.World.* -> Core.Foundation.Foundation.*
    (r'from Core\.World\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.World\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Security.* -> Core.Foundation.Foundation.* (if soul_guardian is in Foundation)
    (r'from Core\.Security\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Security\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Integration.* -> Core.Foundation.Foundation.*
    (r'from Core\.Integration\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Integration\.', 'import Core.Foundation.Foundation.'),
    
    # Project_Sophia.* -> Core.Foundation.Foundation.*
    (r'from Project_Sophia\.', 'from Core.Foundation.Foundation.'),
    (r'import Project_Sophia\.', 'import Core.Foundation.Foundation.'),
    
    # Tools.* -> Core.Foundation.Foundation.*
    (r'from Tools\.', 'from Core.Foundation.Foundation.'),
    (r'import Tools\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Evolution.gemini_api -> Core.Foundation.Foundation.gemini_api
    (r'from Core\.Evolution\.gemini_api', 'from Core.Foundation.Foundation.gemini_api'),
    (r'import Core\.Evolution\.gemini_api', 'import Core.Foundation.Foundation.gemini_api'),
    
    # Core.Structure.* -> Core.Foundation.Foundation.*
    (r'from Core\.Structure\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Structure\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Intelligence.Will.* -> Core.Foundation.Foundation.*
    (r'from Core\.Intelligence\.Will\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Intelligence\.Will\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Interface.* -> Core.Foundation.Foundation.* (some modules moved)
    (r'from Core\.Interface\.shell_cortex', 'from Core.Foundation.Foundation.shell_cortex'),
    (r'from Core\.Interface\.web_cortex', 'from Core.Foundation.Foundation.web_cortex'),
    (r'from Core\.Interface\.cosmic_transceiver', 'from Core.Foundation.Foundation.cosmic_transceiver'),
    (r'from Core\.Interface\.quantum_port', 'from Core.Foundation.Foundation.quantum_port'),
    (r'from Core\.Interface\.holographic_cortex', 'from Core.Foundation.Foundation.holographic_cortex'),
    (r'from Core\.Interface\.envoy_protocol', 'from Core.Foundation.Foundation.envoy_protocol'),
    (r'from Core\.Interface\.synapse_bridge', 'from Core.Foundation.Foundation.synapse_bridge'),
    (r'from Core\.Interface\.user_bridge', 'from Core.Foundation.Foundation.user_bridge'),
    (r'from Core\.Interface\.kenosis_protocol', 'from Core.Foundation.Foundation.kenosis_protocol'),
    (r'from Core\.Interface\.real_communication_system', 'from Core.Foundation.Foundation.real_communication_system'),
    
    # Core.Memory.* -> Core.Foundation.Foundation.*
    (r'from Core\.Memory\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Memory\.', 'import Core.Foundation.Foundation.'),
    
    # Core.Evolution.* -> Core.Foundation.Foundation.* (some modules)
    (r'from Core\.Evolution\.cortex_optimizer', 'from Core.Foundation.Foundation.cortex_optimizer'),
    (r'from Core\.Evolution\.self_reflector', 'from Core.Foundation.Foundation.self_reflector'),
    (r'from Core\.Evolution\.transcendence_engine', 'from Core.Foundation.Foundation.transcendence_engine'),
    (r'from Core\.Evolution\.anamnesis', 'from Core.Foundation.Foundation.anamnesis'),
    
    # Core.Action.* -> Core.Foundation.Foundation.*
    (r'from Core\.Action\.', 'from Core.Foundation.Foundation.'),
    (r'import Core\.Action\.', 'import Core.Foundation.Foundation.'),
    
    # Core.System.* -> Core.Foundation.Foundation.* (some modules)
    (r'from Core\.System\.global_grid', 'from Core.Foundation.Foundation.global_grid'),
    (r'from Core\.System\.wave_integration_hub', 'from Core.Foundation.Foundation.wave_integration_hub'),
    
    # scripts.Maintenance.* -> Core.Foundation.Foundation.* (if moved)
    (r'from scripts\.Maintenance\.self_integration', 'from Core.Foundation.Foundation.self_integration'),
    
    # Keep Core.Intelligence.Will, Core.Intelligence.Reasoning, etc. as is
    # But fix Core.Intelligence.reasoning_engine etc.
    (r'from Core\.Intelligence\.reasoning_engine', 'from Core.Cognition.Reasoning.reasoning_engine'),
    (r'from Core\.Intelligence\.dream_engine', 'from Core.Foundation.Foundation.dream_engine'),
    (r'from Core\.Intelligence\.mind_mitosis', 'from Core.Foundation.Foundation.mind_mitosis'),
    (r'from Core\.Intelligence\.knowledge_acquisition', 'from Core.Foundation.Foundation.knowledge_acquisition'),
    (r'from Core\.Intelligence\.code_cortex', 'from Core.Foundation.Foundation.code_cortex'),
    (r'from Core\.Intelligence\.rapid_learning_engine', 'from Core.Foundation.Foundation.rapid_learning_engine'),
    (r'from Core\.Intelligence\.language_center', 'from Core.Foundation.Foundation.language_center'),
    (r'from Core\.Intelligence\.autonomous_language', 'from Core.Foundation.Foundation.autonomous_language'),
    (r'from Core\.Intelligence\.ultra_dimensional_reasoning', 'from Core.Foundation.Foundation.ultra_dimensional_reasoning'),
    (r'from Core\.Intelligence\.social_cortex', 'from Core.Foundation.Foundation.social_cortex'),
    (r'from Core\.Intelligence\.media_cortex', 'from Core.Foundation.Foundation.media_cortex'),
    (r'from Core\.Intelligence\.imagination_core', 'from Core.Foundation.Foundation.imagination_core'),
    (r'from Core\.Intelligence\.loop_breaker', 'from Core.Foundation.Foundation.loop_breaker'),
    (r'from Core\.Intelligence\.black_hole', 'from Core.Foundation.Foundation.black_hole'),
    (r'from Core\.Intelligence\.quantum_reader', 'from Core.Foundation.Foundation.quantum_reader'),
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
