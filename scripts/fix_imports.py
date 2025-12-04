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
    (r'from Core\.Physics\.', 'from Core.Foundation.'),
    (r'import Core\.Physics\.', 'import Core.Foundation.'),
    
    # Core.Language.* -> Core.Foundation.*
    (r'from Core\.Language\.', 'from Core.Foundation.'),
    (r'import Core\.Language\.', 'import Core.Foundation.'),
    
    # Keep Core.Intelligence.Will, Core.Intelligence.Reasoning, etc. as is
    # But fix Core.Intelligence.reasoning_engine etc.
    (r'from Core\.Intelligence\.reasoning_engine', 'from Core.Foundation.reasoning_engine'),
    (r'from Core\.Intelligence\.dream_engine', 'from Core.Foundation.dream_engine'),
    (r'from Core\.Intelligence\.mind_mitosis', 'from Core.Foundation.mind_mitosis'),
    (r'from Core\.Intelligence\.knowledge_acquisition', 'from Core.Foundation.knowledge_acquisition'),
    (r'from Core\.Intelligence\.code_cortex', 'from Core.Foundation.code_cortex'),
    (r'from Core\.Intelligence\.rapid_learning_engine', 'from Core.Foundation.rapid_learning_engine'),
    (r'from Core\.Intelligence\.language_center', 'from Core.Foundation.language_center'),
    (r'from Core\.Intelligence\.autonomous_language', 'from Core.Foundation.autonomous_language'),
    (r'from Core\.Intelligence\.ultra_dimensional_reasoning', 'from Core.Foundation.ultra_dimensional_reasoning'),
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
