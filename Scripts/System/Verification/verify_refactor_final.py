
import sys
import os

# Ensure we can import from Core
sys.path.append(os.getcwd())

def verify_metabolic():
    print("Testing MetabolicEngine...")
    try:
        from Core.S1_Body.L2_Metabolism.M1_Pulse.metabolic_engine import MetabolicEngine
        engine = MetabolicEngine()
        status = engine.get_status()
        if 'hud_string' in status:
            print(f"✅ MetabolicEngine HUD: {status['hud_string']}")
        else:
            print("❌ MetabolicEngine missing 'hud_string'")
    except Exception as e:
        print(f"❌ MetabolicEngine Error: {e}")

def verify_governance():
    print("\nTesting GovernanceEngine...")
    try:
        from Core.S1_Body.L6_Structure.Engine.governance_engine import GovernanceEngine
        # Mocking external checks inside GovernanceEngine potentially? 
        # GovernanceEngine.__init__ imports OnionEnsemble which might be complex.
        # We will try to instantiate it.
        engine = GovernanceEngine()
        report = engine.get_vital_report()
        if 'mode' in report and 'rpm_spirit' in report:
            print(f"✅ GovernanceEngine Report: {report}")
        else:
            print(f"❌ GovernanceEngine invalid report: {report}")
    except Exception as e:
        print(f"❌ GovernanceEngine Error: {e}")

if __name__ == "__main__":
    verify_metabolic()
    verify_governance()
