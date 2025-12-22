"""
Verify Wave Integration (Phase 79)
==================================
Tests the SelfModifier with existing WaveCodingSystem.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Autonomy.self_modifier_v2 import get_self_modifier


def verify_wave_integration():
    print("🌊 Verifying Wave Resonance Integration...", flush=True)
    print("=" * 50, flush=True)
    
    modifier = get_self_modifier()
    
    # 1. Analyze a known complex file
    target_file = os.path.join(os.path.dirname(__file__), "..", "Core", "Foundation", "ollama_bridge.py")
    target_file = os.path.abspath(target_file)
    
    print(f"\n[Test 1] Analyzing: {os.path.basename(target_file)}", flush=True)
    
    result = modifier.analyze_file(target_file)
    
    print(f"   📊 Tension: {result.tension:.2f} (0=Clean, 1=Spaghetti)", flush=True)
    print(f"   📏 Mass: {result.mass:.2f} (File Size)", flush=True)
    print(f"   🔄 Frequency: {result.frequency:.1f} (Complexity)", flush=True)
    print(f"   🔗 Resonance: {result.resonance:.2f} (Connectivity)", flush=True)
    
    if result.dna_hash:
        print(f"   🧬 DNA Hash: {result.dna_hash[:16]}...", flush=True)
    
    # 2. Find high tension spots
    print(f"\n[Test 2] Finding High Tension Spots...", flush=True)
    
    spots = modifier.find_high_tension_spots(target_file)
    
    if spots:
        for spot in spots[:3]:
            print(f"   ⚠️ Line {spot.line_start}: {spot.description}", flush=True)
            print(f"      💡 {spot.suggestion}", flush=True)
    else:
        print("   ✅ No major high-tension spots found.", flush=True)
    
    # 3. Resonance suggestions
    print(f"\n[Test 3] Resonance Suggestions...", flush=True)
    
    if result.suggestions:
        for sug in result.suggestions[:3]:
            print(f"   🔊 {sug.description}", flush=True)
    else:
        print("   (No resonance suggestions - wave pool may be empty)", flush=True)
    
    # 4. Generate mini report for Core/Foundation
    print(f"\n[Test 4] Mini Report (Core/Foundation)...", flush=True)
    
    report = modifier.generate_report("c:\\Elysia\\Core\\Foundation")
    
    print(f"   📁 Files Analyzed: {report['total_files']}", flush=True)
    print(f"   🌡️ Average Tension: {report['average_tension']:.2f}", flush=True)
    print(f"   🔥 High Tension Files: {len(report['high_tension_files'])}", flush=True)
    
    if report['high_tension_files']:
        for htf in report['high_tension_files'][:3]:
            print(f"      - {os.path.basename(htf['path'])}: T={htf['tension']:.2f}", flush=True)
    
    print("\n✅ Wave Integration Verified.", flush=True)


if __name__ == "__main__":
    verify_wave_integration()
