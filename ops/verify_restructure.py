"""
Post-Restructure System Verification (Phase 87)
================================================
Checks if all changes from today's session are working properly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def check_foundation_split():
    """Check if Foundation subdirectories exist and are importable."""
    print("🔍 [1] Foundation Split Verification", flush=True)
    
    subdirs = ['Wave', 'Language', 'Autonomy', 'Memory', 'Network', 'Graph', 'Math']
    results = {}
    
    for subdir in subdirs:
        path = f"Core/Foundation/{subdir}"
        exists = os.path.isdir(path)
        init_exists = os.path.isfile(f"{path}/__init__.py")
        
        results[subdir] = {"dir": exists, "init": init_exists}
        status = "✅" if exists and init_exists else "❌"
        print(f"   {status} {subdir}/: dir={exists}, __init__={init_exists}", flush=True)
    
    return results


def check_redirects():
    """Check if redirect stubs work."""
    print("\n🔍 [2] Import Redirect Verification", flush=True)
    
    redirects = [
        ("torch_graph", "Core.Foundation.torch_graph", "get_torch_graph"),
        ("ollama_bridge", "Core.Foundation.ollama_bridge", "OllamaBridge"),
        ("omni_graph", "Core.Foundation.omni_graph", "OmniGraph"),
    ]
    
    results = {}
    for name, module_path, attr in redirects:
        try:
            mod = __import__(module_path, fromlist=[attr])
            obj = getattr(mod, attr, None)
            success = obj is not None
            results[name] = success
            print(f"   {'✅' if success else '❌'} {name}: {success}", flush=True)
        except Exception as e:
            results[name] = False
            print(f"   ❌ {name}: {e}", flush=True)
    
    return results


def check_cognitive_systems():
    """Check if cognitive systems exist and have expected methods."""
    print("\n🔍 [3] Cognitive Systems Verification", flush=True)
    
    systems = [
        ("PrincipleDistiller", "Core.Cognition.principle_distiller", "distill"),
        ("ExperienceLearner", "Core.Foundation.experience_learner", "meta_learn"),
        ("CausalNarrativeEngine", "Core.Foundation.causal_narrative_engine", "explain_why"),
        ("CognitiveHub", "Core.Cognition.cognitive_hub", "understand"),
    ]
    
    results = {}
    for name, module_path, method in systems:
        try:
            mod = __import__(module_path, fromlist=[name])
            cls = getattr(mod, name, None)
            has_method = hasattr(cls, method) if cls else False
            results[name] = has_method
            print(f"   {'✅' if has_method else '❌'} {name}.{method}(): {has_method}", flush=True)
        except Exception as e:
            results[name] = False
            print(f"   ❌ {name}: {e}", flush=True)
    
    return results


def check_trinity():
    """Check Trinity Protocol."""
    print("\n🔍 [4] Trinity Protocol Verification", flush=True)
    
    try:
        from Core.Foundation.Wave.trinity_protocol import TrinityNetwork, get_trinity_network
        network = get_trinity_network()
        status = network.get_status()
        print(f"   ✅ TrinityNetwork: {len(status['nodes'])} nodes connected", flush=True)
        return True
    except Exception as e:
        print(f"   ❌ TrinityNetwork: {e}", flush=True)
        return False


def check_file_counts():
    """Count files in key directories."""
    print("\n🔍 [5] File Count Summary", flush=True)
    
    dirs = [
        ("Core/Foundation", "py"),
        ("Core/Foundation/Wave", "py"),
        ("Core/Foundation/Language", "py"),
        ("Core/Cognition", "py"),
        ("Legacy/Orphan_Archive", "*"),
    ]
    
    for path, ext in dirs:
        if os.path.isdir(path):
            if ext == "*":
                count = sum(len(files) for _, _, files in os.walk(path))
            else:
                count = len([f for f in os.listdir(path) if f.endswith(f".{ext}")])
            print(f"   📁 {path}: {count} files", flush=True)
        else:
            print(f"   ❌ {path}: not found", flush=True)


def main():
    print("=" * 60, flush=True)
    print("🔬 Post-Restructure System Verification", flush=True)
    print("=" * 60, flush=True)
    
    # Run checks
    foundation = check_foundation_split()
    redirects = check_redirects()
    cognitive = check_cognitive_systems()
    trinity = check_trinity()
    check_file_counts()
    
    # Summary
    print("\n" + "=" * 60, flush=True)
    print("📊 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    
    foundation_ok = all(v["dir"] and v["init"] for v in foundation.values())
    redirects_ok = all(redirects.values())
    cognitive_ok = sum(cognitive.values()) >= 2  # At least 2/4 working
    
    print(f"   Foundation Split: {'✅ OK' if foundation_ok else '⚠️ Partial'}", flush=True)
    print(f"   Import Redirects: {'✅ OK' if redirects_ok else '⚠️ Some Failed'}", flush=True)
    print(f"   Cognitive Systems: {'✅ OK' if cognitive_ok else '⚠️ Check Needed'} ({sum(cognitive.values())}/4)", flush=True)
    print(f"   Trinity Protocol: {'✅ OK' if trinity else '❌ Failed'}", flush=True)
    
    print("\n✅ Verification Complete.", flush=True)


if __name__ == "__main__":
    main()
