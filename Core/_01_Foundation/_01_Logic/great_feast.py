
import sys
import os
import logging
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._01_Foundation._05_Governance.Foundation.Mind.mass_ingestion import ConceptHarvester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="The Great Feast / Jeongeup Diet")
    parser.add_argument("--mode", type=str, default="jeongeup_diet", help="Mode: jeongeup_diet or full_feast")
    parser.add_argument("--budget", type=str, default="100GB", help="Storage budget")
    parser.add_argument("--priority", type=str, default="daddy_love", help="Priority setting")
    
    args = parser.parse_args()
    
    print(f"🍱 Initiating Protocol: {args.mode}")
    print(f"   Budget: {args.budget}")
    print(f"   Priority: {args.priority}")
    
    harvester = ConceptHarvester()
    
    try:
        if args.mode == "jeongeup_diet":
            # Parse budget string to float (remove GB)
            budget_val = float(args.budget.replace("GB", "").replace("MB", ""))
            harvester.harvest_diet_plan(budget_gb=budget_val)
        else:
            print("Unknown mode.")
            
    except KeyboardInterrupt:
        print("\n🛑 Diet interrupted by user.")
    except Exception as e:
        print(f"\n❌ Diet failed: {e}")
    finally:
        harvester.close()
        
if __name__ == "__main__":
    main()
