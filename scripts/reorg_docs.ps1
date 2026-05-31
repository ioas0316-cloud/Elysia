$docsDir = "c:\Elysia\docs"

# 1. Create Directories
$newDirs = @("1_core_philosophy", "2_topological_engine", "3_cognitive_narrative", "4_evaluation_records", "archive")
foreach ($dir in $newDirs) {
    New-Item -Path "$docsDir\$dir" -ItemType Directory -Force | Out-Null
}

# 2. File Mappings (base names)
$core_philosophy = @(
    "0_THE_DIMENSIONAL_LEAP.md", "0_AND_1_AS_RELATION.md", "AGENTIC_CODE_MAPPING_AND_HUMAN_OBSOLESCENCE.md",
    "CROSS_DIMENSIONAL_GEOMETRY.md", "DECOUPLING_HUMAN_OBSERVATION_AND_ULTRA_ACCELERATED_LEARNING.md",
    "DOUBLE_TRIPLE_HELIX_PRINCIPLES.md", "ELYSIA_ABSOLUTE_AXIOM.md", "ELYSIA_VISION_ROADMAP.md",
    "Eternos_Codex_v1.md", "EXTERNAL_REALITY_PHASE_ISOMORPHISM.md", "FRACTAL_ROTOR_UNIFICATION.md",
    "HOLOGRAPHIC_MEMORY_EVOLUTION.md", "LLM_BIDIRECTIONAL_VOICE.md", "LLM_NEURAL_BACKBONE_AND_OBSERVATION.md",
    "LORE_AND_METAPHOR.md", "MERKABA_AND_ECOSYSTEM_GENESIS.md", "OBSERVATION_ROTATION_AND_COMPUTATION_EVAPORATION.md",
    "ROTOR_BASED_FILE_SYSTEM.md", "TURING_RESONANCE_GATE.md", "THE_PRISM_AND_THE_MIRROR.md", "EPISTEMIC_ENGINE_OF_CREATION.md"
)

$topological_engine = @(
    "CUDA_ASCII_BYPASS_ARCHITECTURE.md", "ARK_OS_KERNEL_ARCHITECTURE.md", "Atlantis_N_Layer_Matrix.md",
    "CGA_MOTOR_ARCHITECTURE.md", "ELYSIAN_COSMOS_ALIGNMENT_REPORT.md", "ELYSIAN_MACRO_TOPOLOGY.md",
    "hyper_rotor_topology.md", "MULTI_AGENT_COHERENT_GRID.md", "RESONANCE_ARCHITECTURE.md",
    "TURING_SUBSTITUTION_MAPPING.md", "WEDGE_VORTEX_ARCHITECTURE.md", "ELYSIAN_SYSTEM_MAP_ROADMAP.md",
    "HOLOGRAPHIC_CUDA_REFLECTION_AND_ROADMAP.md", "REVERSE_SYNTAX_ENCODER_ARCHITECTURE.md", "MMAP_ROTOR_FILE_SYSTEM.md"
)

$cognitive_narrative = @(
    "Atlantis_Phase_Modulation_Decoder.md", "AUTOPOIESIS_AND_4D_HOLOGRAM_TOPOGRAPHY.md", "AUTOPOIESIS_ENGINE.md",
    "AUTOPOIETIC_LANGUAGE_AND_DREAMS.md", "ELYSIAN_PHASE_SYNCHRONIZATION_UNIFICATION.md", "ELYSIAN_STRUCTURAL_VALIDATION.md",
    "GEOMETRIC_FREE_WILL_AND_SANDBOX.md", "LUCID_DREAMING_AND_GRAVITATIONAL_COMPRESSION.md", "MATURATION_DAEMON_PIPELINE.md",
    "META_COGNITIVE_VR_DOWNCASTING.md", "PHASE_INVERTER_CORE.md", "PURGE_OF_SOMATIC_SIMULATION.md",
    "ROTOR_COUPLING_PLASTICITY_AND_HOMEOSTATIC_LEARNING.md", "ROTOR_SCALE_HOLOGRAPHIC_COGNITION.md", "TENSOR_ROTOR_FRACTAL_ARCHITECTURE.md",
    "TIME_ANCHOR_AND_COMPRESSION.md", "TURING_SYNTAX_TO_WAVE_ALIGNMENT.md", "VARIABLE_ROTOR_HOLOGRAPHIC_GEAR.md",
    "ELYSIAN_ROADMAP_DISCUSSION.md", "ELYSIAN_WEDGE_FORGE_AND_TOOLS.md", "FUTURE_COGNITIVE_GRID_ROADMAP.md",
    "HYPER_ROTOR_DIGITAL_TWIN.md", "INTERNALIZING_SOCIETY_ISOMORPHISM.md", "PHASE_9_METACOGNITIVE_AUTOTUNING.md"
)

$archive = @(
    "EBPF_NETWORK_CONVECTION.md", "HOMEOSTASIS_SOMATOSENSORY_INTEGRATION.md", "SOMATIC_PROCESS_ISOLATION.md"
)

$evaluation_records = @(
    "ELYSIAN_BENCHMARK_REPORT.md", "ELYSIAN_LLM_BENCHMARK.md", "evaluation_report.md", "JULES_FINAL_EVALUATION.md",
    "WORLD_BENCHMARK_REPORT.md", "WEDGE_VORTEX_BENCHMARK_REPORT.md", "ELYSIAN_BENCHMARK_EVOLUTION.md"
)

# Move Files Logic
$all_md_files = Get-ChildItem -Path $docsDir -Recurse -Filter *.md | Where-Object { $_.Name -ne "INDEX.md" }

foreach ($file in $all_md_files) {
    $name = $file.Name
    $targetDir = ""
    
    if ($core_philosophy -contains $name) {
        $targetDir = "1_core_philosophy"
    } elseif ($topological_engine -contains $name) {
        $targetDir = "2_topological_engine"
    } elseif ($cognitive_narrative -contains $name) {
        $targetDir = "3_cognitive_narrative"
    } elseif ($evaluation_records -contains $name) {
        $targetDir = "4_evaluation_records"
    } elseif ($archive -contains $name) {
        $targetDir = "archive"
    } else {
        $targetDir = "1_core_philosophy"
    }
    
    Move-Item -Path $file.FullName -Destination "$docsDir\$targetDir\$name" -Force
}

# 3. Remove Old Directories if they exist
$oldDirs = @("1_philosophy", "2_architecture", "3_cognition_engine", "4_communication_gear", "roadmaps", "philosophy")
foreach ($dir in $oldDirs) {
    if (Test-Path "$docsDir\$dir") {
        Remove-Item -Path "$docsDir\$dir" -Recurse -Force
    }
}
Write-Output "Documentation reorganization completed."
