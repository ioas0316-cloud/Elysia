
import os
import re

replacements = {
    'Core.FoundationLayer.Foundation.emotional_engine': 'Core.L5_Mental.M4_Meaning.emotional_engine',
    'Core.L1_Foundation.Foundation.emotional_engine': 'Core.L5_Mental.M4_Meaning.emotional_engine',
    'Core.FoundationLayer.Foundation.core_memory': 'Core.L5_Mental.Memory.core_memory',
    'Core.L1_Foundation.Foundation.core_memory': 'Core.L5_Mental.Memory.core_memory',
    'Core.FoundationLayer.Foundation.logical_reasoner': 'Core.L5_Mental.Logic.logical_reasoner',
    'Core.L1_Foundation.Foundation.logical_reasoner': 'Core.L5_Mental.Logic.logical_reasoner',
    'Core.FoundationLayer.Foundation.wave_mechanics': 'Core.L5_Mental.Intelligence.wave_mechanics',
    'Core.L1_Foundation.Foundation.wave_mechanics': 'Core.L5_Mental.Intelligence.wave_mechanics',
    'Core.FoundationLayer.Foundation.response_styler': 'Core.L5_Mental.Intelligence.M3_Lexicon.response_styler',
    'Core.L1_Foundation.Foundation.response_styler': 'Core.L5_Mental.Intelligence.M3_Lexicon.response_styler',
    'Core.FoundationLayer.Foundation.insight_synthesizer': 'Core.L5_Mental.Intelligence.M4_Meaning.insight_synthesizer',
    'Core.L1_Foundation.Foundation.insight_synthesizer': 'Core.L5_Mental.Intelligence.M4_Meaning.insight_synthesizer',
    'Core.FoundationLayer.Foundation.question_generator': 'Core.L5_Mental.Intelligence.question_generator',
    'Core.L1_Foundation.Foundation.question_generator': 'Core.L5_Mental.Intelligence.question_generator',
    'Core.FoundationLayer.Foundation.arithmetic_cortex': 'Core.L5_Mental.Intelligence.arithmetic_cortex',
    'Core.L1_Foundation.Foundation.arithmetic_cortex': 'Core.L5_Mental.Intelligence.arithmetic_cortex',
    'Core.FoundationLayer.Foundation.relationship_extractor': 'Core.L5_Mental.Intelligence.relationship_extractor',
    'Core.L1_Foundation.Foundation.relationship_extractor': 'Core.L5_Mental.Intelligence.relationship_extractor',
    'Core.FoundationLayer.Foundation.core.tensor_wave': 'Core.L5_Mental.Intelligence.Memory.unified_types',
    'Core.L1_Foundation.Foundation.core.tensor_wave': 'Core.L5_Mental.Intelligence.Memory.unified_types',
    'Core.FoundationLayer.Foundation.vector_utils': 'Core.L5_Mental.Intelligence.vector_utils',
    'Core.L1_Foundation.Foundation.vector_utils': 'Core.L5_Mental.Intelligence.vector_utils',
    'from emotional_engine': 'from Core.L5_Mental.M4_Meaning.emotional_engine',
    'from core_memory': 'from Core.L5_Mental.Memory.core_memory',
    'from tools.kg_manager': 'from Core.L5_Mental.Memory.kg_manager',
    'from tools.vector_utils': 'from Core.L5_Mental.Intelligence.vector_utils',
    'from Project_Elysia.architecture': 'from Core.L6_Structure.Architecture',
    'from Project_Elysia.core_memory': 'from Core.L5_Mental.Memory.core_memory',
    'from thought import Thought': 'from Core.L5_Mental.Logic.thought import Thought',
    'from tensor_wave import Tensor3D': 'from Core.L5_Mental.Intelligence.Memory.unified_types import Tensor3D',
    'from kg_manager import KGManager': 'from Core.L5_Mental.Memory.kg_manager import KGManager',
    'from wave_mechanics import WaveMechanics': 'from Core.L5_Mental.Intelligence.wave_mechanics import WaveMechanics',
    'from perspective_cortex import PerspectiveCortex': 'from Core.L5_Mental.Intelligence.perspective_cortex import PerspectiveCortex',
    'from value_centered_decision import ValueCenteredDecision': 'from Core.L5_Mental.Intelligence.value_centered_decision import ValueCenteredDecision',
    'from planning_cortex import PlanningCortex': 'from Core.L5_Mental.Intelligence.planning_cortex import PlanningCortex',
    'from action_cortex import ActionCortex': 'from Core.L5_Mental.Intelligence.action_cortex import ActionCortex',
    'from arithmetic_cortex import ArithmeticCortex': 'from Core.L5_Mental.Intelligence.arithmetic_cortex import ArithmeticCortex',
    'from Project_Elysia.high_engine': 'from Core.L5_Mental.Intelligence.HighEngine',
    'from Project_Elysia.llm': 'from Core.L5_Mental.Intelligence.LLM',
    'from Project_Elysia.physiology': 'from Core.L2_Metabolism.Physiology'
}

def fix_file(path):
    encodings = ['utf-8', 'cp949', 'utf-16', 'latin-1']
    content = None
    used_enc = None
    
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
                used_enc = enc
                break
        except:
            continue
            
    if content is None:
        print(f"Failed to read {path}")
        return

    original_content = content
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    if content != original_content:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed {path} (Original encoding: {used_enc})")
        except Exception as e:
             print(f"Failed to write {path}: {e}")

for root, dirs, files in os.walk('c:/Elysia/Core'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))
