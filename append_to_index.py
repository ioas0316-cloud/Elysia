import os
import re

doc_dir = r"c:\Elysia\docs"
index_file = os.path.join(doc_dir, "INDEX.md")

with open(index_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Read unlinked files
unlinked = [
"1_core_philosophy/AGENTIC_CODE_MAPPING_AND_HUMAN_OBSOLESCENCE.md",
"1_core_philosophy/DECOUPLING_HUMAN_OBSERVATION_AND_ULTRA_ACCELERATED_LEARNING.md",
"1_core_philosophy/ELYSIA_VISION_ROADMAP.md",
"1_core_philosophy/EXTERNAL_REALITY_PHASE_ISOMORPHISM.md",
"1_core_philosophy/HOLOGRAPHIC_MEMORY_EVOLUTION.md",
"1_core_philosophy/LLM_BIDIRECTIONAL_VOICE.md",
"1_core_philosophy/LLM_NEURAL_BACKBONE_AND_OBSERVATION.md",
"1_core_philosophy/LORE_AND_METAPHOR.md",
"1_core_philosophy/ROTOR_BASED_FILE_SYSTEM.md",
"1_core_philosophy/TURING_RESONANCE_GATE.md",
"2_topological_engine/ELYSIAN_COSMOS_ALIGNMENT_REPORT.md",
"2_topological_engine/ELYSIAN_SYSTEM_MAP_ROADMAP.md",
"2_topological_engine/HOLOGRAPHIC_CUDA_REFLECTION_AND_ROADMAP.md",
"2_topological_engine/MULTI_AGENT_COHERENT_GRID.md",
"2_topological_engine/hyper_rotor_topology.md",
"3_cognitive_narrative/AUTOPOIESIS_AND_4D_HOLOGRAM_TOPOGRAPHY.md",
"3_cognitive_narrative/AUTOPOIETIC_STRATIFICATION.md",
"3_cognitive_narrative/Atlantis_Phase_Modulation_Decoder.md",
"3_cognitive_narrative/ELYSIAN_PHASE_SYNCHRONIZATION_UNIFICATION.md",
"3_cognitive_narrative/ELYSIAN_ROADMAP_DISCUSSION.md",
"3_cognitive_narrative/ELYSIAN_WEDGE_FORGE_AND_TOOLS.md",
"3_cognitive_narrative/FUTURE_COGNITIVE_GRID_ROADMAP.md",
"3_cognitive_narrative/GEOMETRIC_FREE_WILL_AND_SANDBOX.md",
"3_cognitive_narrative/HYPER_COGNITIVE_MATURATION.md",
"3_cognitive_narrative/HYPER_ROTOR_DIGITAL_TWIN.md",
"3_cognitive_narrative/INTERNALIZING_SOCIETY_ISOMORPHISM.md",
"3_cognitive_narrative/MATURATION_DAEMON_PIPELINE.md",
"3_cognitive_narrative/PHASE_9_METACOGNITIVE_AUTOTUNING.md",
"3_cognitive_narrative/PHASE_INVERTER_CORE.md",
"3_cognitive_narrative/ROTOR_SCALE_HOLOGRAPHIC_COGNITION.md",
"3_cognitive_narrative/SOCIAL_LANGUAGE_ALIGNMENT.md",
"3_cognitive_narrative/TURING_SYNTAX_TO_WAVE_ALIGNMENT.md",
"3_cognitive_narrative/VARIABLE_ROTOR_HOLOGRAPHIC_GEAR.md",
"4_evaluation_records/ELYSIAN_BENCHMARK_EVOLUTION.md",
"4_evaluation_records/evaluation_report.md"
]

def make_link(rel_path):
    name = os.path.basename(rel_path)
    return f"* **[{name}](file:///c:/Elysia/docs/{rel_path}):** (문서 요약 필요)\n"

# Map sections to insertion line index (end of section before ---)
new_lines = []
i = 0
while i < len(lines):
    new_lines.append(lines[i])
    if "## ⚙️ 2. 위상 기하학 및 하드웨어 매핑" in lines[i]:
        # we found start of section 2. We should have inserted 1's before the --- just above it.
        pass
    i += 1

# Actually a simpler way is to find the --- separators.
# Let's rebuild the file by finding the end of each list.
def insert_at_end_of_section(section_name, items):
    global lines
    start_idx = -1
    for i, line in enumerate(lines):
        if section_name in line:
            start_idx = i
            break
    
    if start_idx == -1: return
    
    # find the next ---
    end_idx = -1
    for i in range(start_idx+1, len(lines)):
        if lines[i].startswith("---"):
            end_idx = i
            break
            
    if end_idx != -1:
        # insert items before end_idx
        for item in reversed(items):
            lines.insert(end_idx, make_link(item))

insert_at_end_of_section("## 🌌 1. 기저 공리", [x for x in unlinked if x.startswith("1_")])
insert_at_end_of_section("## ⚙️ 2. 위상 기하학", [x for x in unlinked if x.startswith("2_")])
insert_at_end_of_section("## 🧠 3. 인지 서사", [x for x in unlinked if x.startswith("3_")])
insert_at_end_of_section("## 📊 4. 실증 벤치마크", [x for x in unlinked if x.startswith("4_")])

with open(index_file, "w", encoding="utf-8") as f:
    f.writelines(lines)
    
print("Successfully updated INDEX.md")
