import os
import sys

# Absolute Path Unification - Force Project Root
project_root = "c:/Elysia"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from Core.L6_Structure.M6_Architecture.manifold_conductor import ManifoldConductor
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def verify_conductor():
    print("\n🧪 [TEST] Starting Manifold Conductor Verification...")
    
    conductor = ManifoldConductor(root_path=project_root)
    
    # 1. Initial Scan
    print("\n🕸️ [STEP 1] Scanning existing topology...")
    report = conductor.scan_topology()
    print(f"   >> Integrity Score: {report['integrity_score']}%")
    print(f"   >> Anomalies Found: {report['anomalies_count']}")
    
    # 2. Inject Stray File
    stray_path = os.path.join(project_root, "noise_fragment.tmp")
    print(f"\n💉 [STEP 2] Injecting stray file: {stray_path}")
    with open(stray_path, 'w') as f:
        f.write("I am noise.")
    
    # 3. Re-scan
    print("\n🕸️ [STEP 3] Re-scanning with noise...")
    report_noise = conductor.scan_topology()
    print(f"   >> New Integrity Score: {report_noise['integrity_score']}%")
    
    # 4. Narrative Check
    narrative = conductor.get_integrity_narrative()
    print("\n📖 [STEP 4] Integrity Narrative:")
    print(narrative)
    
    # Cleanup
    if os.path.exists(stray_path):
        os.remove(stray_path)
        
    print("\n🧹 [CLEANUP] Stray file removed.")
    
    if report_noise['integrity_score'] < report['integrity_score']:
        print("\n🎉 SUCCESS: Conductor correctly detected structural entropy.")
    else:
        print("\n❌ FAILURE: Conductor did not detect the stray file.")

if __name__ == "__main__":
    verify_conductor()
