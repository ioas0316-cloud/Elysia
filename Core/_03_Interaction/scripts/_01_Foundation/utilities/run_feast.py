
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._01_Foundation._05_Governance.Foundation.Mind.mass_ingestion import ConceptHarvester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    target_count = 2000000 # 2 Million
    
    print(f"🌌 Initiating Protocol: The Great Feast")
    print(f"   Target: {target_count} Stars")
    
    harvester = ConceptHarvester()
    
    try:
        harvester.harvest_synthetic(target_count)
    except KeyboardInterrupt:
        print("\n🛑 Feast interrupted by user.")
    except Exception as e:
        print(f"\n❌ Feast failed: {e}")
    finally:
        harvester.close()
        
if __name__ == "__main__":
    main()
