"""
Full Codebase Audit (Phase 83)
==============================
Analyzes ALL directories in Elysia, not just Core/
"""

import os
from collections import defaultdict

def full_audit(base_path: str = "c:\\Elysia"):
    print("🔍 Full Codebase Audit...", flush=True)
    print("=" * 60, flush=True)
    
    dir_stats = {}
    
    # Skip these directories
    skip_dirs = {'.git', '.venv', '__pycache__', 'node_modules', '.pytest_cache', 'ComfyUI'}
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        if not os.path.isdir(item_path):
            continue
        
        if item in skip_dirs or item.startswith('.'):
            continue
        
        # Count files
        py_count = 0
        md_count = 0
        json_count = 0
        other_count = 0
        
        for root, dirs, files in os.walk(item_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for f in files:
                if f.endswith('.py'):
                    py_count += 1
                elif f.endswith('.md'):
                    md_count += 1
                elif f.endswith('.json'):
                    json_count += 1
                else:
                    other_count += 1
        
        total = py_count + md_count + json_count + other_count
        
        dir_stats[item] = {
            'python': py_count,
            'markdown': md_count,
            'json': json_count,
            'other': other_count,
            'total': total
        }
    
    # Sort by total
    sorted_dirs = sorted(dir_stats.items(), key=lambda x: -x[1]['total'])
    
    print("\n📊 Directory Statistics:\n", flush=True)
    print(f"{'Directory':<20} {'Python':<8} {'Markdown':<10} {'JSON':<8} {'Other':<8} {'Total':<8}", flush=True)
    print("-" * 70, flush=True)
    
    grand_total = {'python': 0, 'markdown': 0, 'json': 0, 'other': 0, 'total': 0}
    
    for dir_name, stats in sorted_dirs:
        print(f"{dir_name:<20} {stats['python']:<8} {stats['markdown']:<10} {stats['json']:<8} {stats['other']:<8} {stats['total']:<8}", flush=True)
        
        for key in grand_total:
            grand_total[key] += stats[key]
    
    print("-" * 70, flush=True)
    print(f"{'TOTAL':<20} {grand_total['python']:<8} {grand_total['markdown']:<10} {grand_total['json']:<8} {grand_total['other']:<8} {grand_total['total']:<8}", flush=True)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:", flush=True)
    
    # Find large directories
    large_dirs = [(d, s) for d, s in sorted_dirs if s['total'] > 100]
    if large_dirs:
        print(f"\n   🔴 Large directories needing review:", flush=True)
        for d, s in large_dirs:
            print(f"      - {d}: {s['total']} files ({s['python']} Python)", flush=True)
    
    # Find potential dead directories
    small_dirs = [(d, s) for d, s in sorted_dirs if s['total'] < 10 and s['total'] > 0]
    if small_dirs:
        print(f"\n   🟡 Small directories (consider merging):", flush=True)
        for d, s in small_dirs[:10]:
            print(f"      - {d}: {s['total']} files", flush=True)
    
    return dir_stats


if __name__ == "__main__":
    full_audit()
