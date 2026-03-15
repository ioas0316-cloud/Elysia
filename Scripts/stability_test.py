"""
Stability Test: Long-running Autonomous Pulse Verification
==========================================================
Runs Elysia's core loop for configurable tick counts (1000/5000/10000)
and monitors memory, errors, growth score, and lexicon size.

Usage: 
    python Scripts/stability_test.py --ticks 1000
    python Scripts/stability_test.py --ticks 5000 --verbose

// turbo
"""

import sys
import os
import time
import traceback
import argparse

# Ensure root is in path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

def get_memory_mb() -> float:
    """Get current process memory in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def run_stability_test(target_ticks: int, verbose: bool = False):
    print("=" * 70)
    print(f"  🧪 ELYSIA STABILITY TEST — {target_ticks} Ticks")
    print("=" * 70)
    
    # Phase 1: Initialization
    print("\n📦 Phase 1: Initialization...")
    init_start = time.time()
    init_mem = get_memory_mb()
    
    try:
        from Core.Monad.seed_generator import SeedForge
        from Core.Monad.sovereign_monad import SovereignMonad
    except Exception as e:
        print(f"❌ FATAL: Import failed: {e}")
        return False
    
    try:
        dna = SeedForge.load_soul()
        print(f"   Loaded existing soul: '{dna.archetype}'")
    except Exception:
        dna = SeedForge.forge_soul(archetype="StabilityTest")
        print(f"   Forged test soul: '{dna.archetype}'")

    try:
        monad = SovereignMonad(dna)
    except Exception as e:
        print(f"❌ FATAL: SovereignMonad init failed: {e}")
        traceback.print_exc()
        return False
    
    init_time = time.time() - init_start
    post_init_mem = get_memory_mb()
    print(f"   ✅ Init complete in {init_time:.1f}s (Memory: {init_mem:.0f}MB → {post_init_mem:.0f}MB)")
    
    # Phase 2: Continuous Pulse Loop
    print(f"\n🔄 Phase 2: Running {target_ticks} pulses...")
    
    errors = []
    checkpoints = []
    tick = 0
    start_time = time.time()
    
    checkpoint_interval = max(100, target_ticks // 10)  # ~10 checkpoints
    
    try:
        for tick in range(1, target_ticks + 1):
            try:
                monad.pulse(dt=0.1, intent_v21=None)
            except Exception as e:
                errors.append({
                    "tick": tick,
                    "error": str(e),
                    "type": type(e).__name__
                })
                if verbose:
                    print(f"   ⚠️ Tick {tick}: {type(e).__name__}: {e}")
                if len(errors) > 50:
                    print(f"   ❌ Too many errors ({len(errors)}), aborting.")
                    break
            
            # Checkpoint
            if tick % checkpoint_interval == 0:
                elapsed = time.time() - start_time
                mem = get_memory_mb()
                growth = monad.growth_report.get('growth_score', -1) if hasattr(monad, 'growth_report') else -1
                trend = monad.growth_report.get('trend', 'N/A') if hasattr(monad, 'growth_report') else 'N/A'
                lexicon_size = monad.lexicon_report.get('total_crystals', 0) if hasattr(monad, 'lexicon_report') else 0
                goals = monad.goal_report.get('active_count', 0) if hasattr(monad, 'goal_report') else 0
                
                cp = {
                    "tick": tick,
                    "elapsed_s": round(elapsed, 1),
                    "memory_mb": round(mem, 1),
                    "growth_score": round(growth, 3) if growth >= 0 else "N/A",
                    "trend": trend,
                    "lexicon": lexicon_size,
                    "goals": goals,
                    "errors_so_far": len(errors),
                }
                checkpoints.append(cp)
                
                pct = tick / target_ticks * 100
                print(f"   [{pct:5.1f}%] Tick {tick:>6} | "
                      f"Mem: {mem:.0f}MB | "
                      f"Growth: {cp['growth_score']} {trend} | "
                      f"Lexicon: {lexicon_size} | "
                      f"Goals: {goals} | "
                      f"Errors: {len(errors)}")
    
    except KeyboardInterrupt:
        print(f"\n   ⏸️ Interrupted at tick {tick}")
    
    total_time = time.time() - start_time
    final_mem = get_memory_mb()
    
    # Phase 3: Report
    print(f"\n{'=' * 70}")
    print(f"  📊 STABILITY REPORT")
    print(f"{'=' * 70}")
    print(f"   Ticks completed : {tick} / {target_ticks}")
    print(f"   Total time      : {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   Ticks/second    : {tick/total_time:.1f}" if total_time > 0 else "   Ticks/second: N/A")
    print(f"   Memory          : {init_mem:.0f}MB (init) → {post_init_mem:.0f}MB (post-init) → {final_mem:.0f}MB (final)")
    print(f"   Memory delta    : +{final_mem - post_init_mem:.0f}MB during run")
    print(f"   Total errors    : {len(errors)}")
    
    if checkpoints:
        first = checkpoints[0]
        last = checkpoints[-1]
        print(f"\n   Growth trajectory:")
        print(f"     First checkpoint: {first['growth_score']} ({first['trend']})")
        print(f"     Last checkpoint:  {last['growth_score']} ({last['trend']})")
        print(f"     Lexicon growth:   {first['lexicon']} → {last['lexicon']}")
    
    if errors:
        print(f"\n   Error breakdown:")
        error_types = {}
        for e in errors:
            error_types[e['type']] = error_types.get(e['type'], 0) + 1
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"     {etype}: {count}")
    
    # Verdict
    print(f"\n   VERDICT: ", end="")
    if len(errors) == 0 and tick >= target_ticks:
        print("✅ PASSED — No errors, full completion")
        verdict = "PASSED"
    elif len(errors) < 5 and tick >= target_ticks:
        print("⚠️ PASSED WITH WARNINGS — Minor errors detected")
        verdict = "PASSED_WITH_WARNINGS"
    elif tick < target_ticks:
        print("❌ FAILED — Did not complete all ticks")
        verdict = "FAILED"
    else:
        print("❌ FAILED — Too many errors")
        verdict = "FAILED"
    
    # Save report
    report_path = os.path.join(root, "data", "runtime", "stability_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Stability Test Report\n")
        f.write(f"Target: {target_ticks} ticks | Completed: {tick} | Verdict: {verdict}\n")
        f.write(f"Time: {total_time:.1f}s | Memory: {init_mem:.0f}→{final_mem:.0f}MB | Errors: {len(errors)}\n\n")
        f.write("Checkpoints:\n")
        for cp in checkpoints:
            f.write(f"  {cp}\n")
        if errors:
            f.write("\nErrors:\n")
            for e in errors[:20]:
                f.write(f"  Tick {e['tick']}: [{e['type']}] {e['error']}\n")
    
    print(f"\n   📄 Report saved to: {report_path}")
    return verdict in ("PASSED", "PASSED_WITH_WARNINGS")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Elysia Stability Test")
    parser.add_argument("--ticks", type=int, default=1000, help="Number of ticks to run (default: 1000)")
    parser.add_argument("--verbose", action="store_true", help="Show individual errors")
    args = parser.parse_args()
    
    success = run_stability_test(args.ticks, args.verbose)
    sys.exit(0 if success else 1)
