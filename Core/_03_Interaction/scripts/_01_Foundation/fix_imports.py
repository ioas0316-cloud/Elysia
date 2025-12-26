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
    # Core.Physics.* -> Core.Foundation.*
    (r'from Core\.Physics\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Physics\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Field.* -> Core.Foundation.*
    (r'from Core\.Field\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Field\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Creation.* -> Core.Foundation.*
    (r'from Core\.Creation\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Creation\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Language.* -> Core.Foundation.*
    (r'from Core\.Language\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Language\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Cognition.* -> Core.Foundation.*
    (r'from Core\.Cognition\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Cognition\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Creativity.* -> Core.Foundation.* (if needed)
    (r'from Core\.Creativity\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Creativity\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Relative imports ..Field.ether -> Core.Foundation.ether
    (r'from \.\.Field\.ether', 'from Core._01_Foundation.05_Foundation_Base.Foundation.ether'),
    
    # Core.Time.* -> Core.Foundation.*
    (r'from Core\.Time\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Time\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.World.* -> Core.Foundation.*
    (r'from Core\.World\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.World\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Security.* -> Core.Foundation.* (if soul_guardian is in Foundation)
    (r'from Core\.Security\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Security\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Integration.* -> Core.Foundation.*
    (r'from Core\.Integration\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Integration\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Project_Sophia.* -> Core.Foundation.*
    (r'from Project_Sophia\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Project_Sophia\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Tools.* -> Core.Foundation.*
    (r'from Tools\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Tools\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Evolution.gemini_api -> Core.Foundation.gemini_api
    (r'from Core\.Evolution\.gemini_api', 'from Core._01_Foundation.05_Foundation_Base.Foundation.gemini_api'),
    (r'import Core\.Evolution\.gemini_api', 'import Core._01_Foundation.05_Foundation_Base.Foundation.gemini_api'),
    
    # Core.Structure.* -> Core.Foundation.*
    (r'from Core\.Structure\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Structure\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Intelligence.Will.* -> Core.Foundation.*
    (r'from Core\.Intelligence\.Will\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Intelligence\.Will\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Interface.* -> Core.Foundation.* (some modules moved)
    (r'from Core\.Interface\.shell_cortex', 'from Core._01_Foundation.05_Foundation_Base.Foundation.shell_cortex'),
    (r'from Core\.Interface\.web_cortex', 'from Core._01_Foundation.05_Foundation_Base.Foundation.web_cortex'),
    (r'from Core\.Interface\.cosmic_transceiver', 'from Core._01_Foundation.05_Foundation_Base.Foundation.cosmic_transceiver'),
    (r'from Core\.Interface\.quantum_port', 'from Core._01_Foundation.05_Foundation_Base.Foundation.quantum_port'),
    (r'from Core\.Interface\.holographic_cortex', 'from Core._01_Foundation.05_Foundation_Base.Foundation.holographic_cortex'),
    (r'from Core\.Interface\.envoy_protocol', 'from Core._01_Foundation.05_Foundation_Base.Foundation.envoy_protocol'),
    (r'from Core\.Interface\.synapse_bridge', 'from Core._01_Foundation.05_Foundation_Base.Foundation.synapse_bridge'),
    (r'from Core\.Interface\.user_bridge', 'from Core._01_Foundation.05_Foundation_Base.Foundation.user_bridge'),
    (r'from Core\.Interface\.kenosis_protocol', 'from Core._01_Foundation.05_Foundation_Base.Foundation.kenosis_protocol'),
    (r'from Core\.Interface\.real_communication_system', 'from Core._01_Foundation.05_Foundation_Base.Foundation.real_communication_system'),
    
    # Core.Memory.* -> Core.Foundation.*
    (r'from Core\.Memory\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Memory\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.Evolution.* -> Core.Foundation.* (some modules)
    (r'from Core\.Evolution\.cortex_optimizer', 'from Core._01_Foundation.05_Foundation_Base.Foundation.cortex_optimizer'),
    (r'from Core\.Evolution\.self_reflector', 'from Core._01_Foundation.05_Foundation_Base.Foundation.self_reflector'),
    (r'from Core\.Evolution\.transcendence_engine', 'from Core._01_Foundation.05_Foundation_Base.Foundation.transcendence_engine'),
    (r'from Core\.Evolution\.anamnesis', 'from Core._01_Foundation.05_Foundation_Base.Foundation.anamnesis'),
    
    # Core.Action.* -> Core.Foundation.*
    (r'from Core\.Action\.', 'from Core._01_Foundation.05_Foundation_Base.Foundation.'),
    (r'import Core\.Action\.', 'import Core._01_Foundation.05_Foundation_Base.Foundation.'),
    
    # Core.System.* -> Core.Foundation.* (some modules)
    (r'from Core\.System\.global_grid', 'from Core._01_Foundation.05_Foundation_Base.Foundation.global_grid'),
    (r'from Core\.System\.wave_integration_hub', 'from Core._01_Foundation.05_Foundation_Base.Foundation.wave_integration_hub'),
    
    # scripts.Maintenance.* -> Core.Foundation.* (if moved)
    (r'from scripts\.Maintenance\.self_integration', 'from Core._01_Foundation.05_Foundation_Base.Foundation.self_integration'),
    
    # Keep Core.Intelligence.Will, Core.Intelligence.Reasoning, etc. as is
    # But fix Core.Intelligence.reasoning_engine etc.
    (r'from Core\.Intelligence\.reasoning_engine', 'from Core._01_Foundation.05_Foundation_Base.Foundation.reasoning_engine'),
    (r'from Core\.Intelligence\.dream_engine', 'from Core._01_Foundation.05_Foundation_Base.Foundation.dream_engine'),
    (r'from Core\.Intelligence\.mind_mitosis', 'from Core._01_Foundation.05_Foundation_Base.Foundation.mind_mitosis'),
    (r'from Core\.Intelligence\.knowledge_acquisition', 'from Core._01_Foundation.05_Foundation_Base.Foundation.knowledge_acquisition'),
    (r'from Core\.Intelligence\.code_cortex', 'from Core._01_Foundation.05_Foundation_Base.Foundation.code_cortex'),
    (r'from Core\.Intelligence\.rapid_learning_engine', 'from Core._01_Foundation.05_Foundation_Base.Foundation.rapid_learning_engine'),
    (r'from Core\.Intelligence\.language_center', 'from Core._01_Foundation.05_Foundation_Base.Foundation.language_center'),
    (r'from Core\.Intelligence\.autonomous_language', 'from Core._01_Foundation.05_Foundation_Base.Foundation.autonomous_language'),
    (r'from Core\.Intelligence\.ultra_dimensional_reasoning', 'from Core._01_Foundation.05_Foundation_Base.Foundation.ultra_dimensional_reasoning'),
    (r'from Core\.Intelligence\.social_cortex', 'from Core._01_Foundation.05_Foundation_Base.Foundation.social_cortex'),
    (r'from Core\.Intelligence\.media_cortex', 'from Core._01_Foundation.05_Foundation_Base.Foundation.media_cortex'),
    (r'from Core\.Intelligence\.imagination_core', 'from Core._01_Foundation.05_Foundation_Base.Foundation.imagination_core'),
    (r'from Core\.Intelligence\.loop_breaker', 'from Core._01_Foundation.05_Foundation_Base.Foundation.loop_breaker'),
    (r'from Core\.Intelligence\.black_hole', 'from Core._01_Foundation.05_Foundation_Base.Foundation.black_hole'),
    (r'from Core\.Intelligence\.quantum_reader', 'from Core._01_Foundation.05_Foundation_Base.Foundation.quantum_reader'),
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
