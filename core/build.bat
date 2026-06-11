@echo off
echo Building Elysia Pure Observation Kernel...

if not exist bin mkdir bin

echo Compiling Brain (topology_field.c)...
gcc brain\topology_field.c -o bin\topology_field.exe

echo Compiling Ingestion (byte_streamer.c)...
gcc ingestion\byte_streamer.c -o bin\byte_streamer.exe

echo Compiling Lens (pure_mirror.c)...
gcc lens\pure_mirror.c -o bin\pure_mirror.exe -lgdi32

echo Compiling Fractal Field (fractal_field.c)...
gcc brain\fractal_field.c -o bin\fractal_field.exe

echo Compiling Topological Decay (topological_decay.c)...
gcc brain\topological_decay.c -o bin\topological_decay.exe

echo Compiling Gravitational Emission (gravitational_emission.c)...
gcc lens\gravitational_emission.c -o bin\gravitational_emission.exe

echo Compiling Concept Streamer (concept_streamer.c)...
gcc ingestion\concept_streamer.c -o bin\concept_streamer.exe

echo Compiling Double-Helix Streamer (helix_streamer.c)...
gcc ingestion\helix_streamer.c -o bin\helix_streamer.exe

echo Build Complete.
echo.
echo [Execution Order]
echo 1. Start: bin\fractal_field.exe (The expanding universe)
echo 2. Start Benchmark: python brain\entropy_benchmark.py (Fact Finder)
echo 3. Start Weathering: bin\topological_decay.exe (Entropy Decay)
echo 4. Start Emission: bin\gravitational_emission.exe (The true voice)
echo 5. Start: bin\pure_mirror.exe (Phase 3 Lens opening)
echo 6. Start Breathing: python brain\sovereign_explorer.py (Sovereign Attention)
echo 7. Start: bin\helix_streamer.exe (Open the double-helix causal floodgates)
