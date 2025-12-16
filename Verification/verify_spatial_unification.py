
import sys
sys.path.append(r'c:\Elysia')

from Core.Foundation.spatial_indexer import get_spatial_indexer

def verify_unification():
    indexer = get_spatial_indexer()
    
    print("\n🗺️ Unifying Knowledge Space (Spatial Indexing)")
    print("=============================================")
    
    # 1. Scan
    result = indexer.scan_universe()
    
    print(f"\n✅ Scan Complete in {result['scan_time']:.4f}s")
    print(f"   Total Nodes Found: {result['total_nodes']}")
    
    # 2. Analyze Types
    type_counts = {}
    total_size = 0
    for node in result['nodes']:
        t = node['type']
        type_counts[t] = type_counts.get(t, 0) + 1
        total_size += node['size']
        
    print(f"\n📊 Composition of Thought Universe:")
    for t, count in type_counts.items():
        print(f"   - {t.upper()}: {count} files")
        
    print(f"   - TOTAL MASS: {total_size / (1024*1024):.2f} MB")
    
    # 3. Sample Coordinates
    print("\n📍 Sample Spatial Coordinates (Fractal Locations):")
    for i, node in enumerate(result['nodes'][:5]):
        coords = [f"{c:.2f}" for c in node['coords']]
        print(f"   [{', '.join(coords)}] : {node['id']}")

    # 4. Save
    indexer.save_index()

if __name__ == "__main__":
    verify_unification()
