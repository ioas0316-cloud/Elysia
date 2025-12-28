
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._01_Foundation.Foundation.Mind.mass_ingestion import ConceptHarvester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print(f"🧹 Initiating Protocol: Work-Life Balance")
    
    harvester = ConceptHarvester()
    
    try:
        harvester.clean_gallery()
    except KeyboardInterrupt:
        print("\n🛑 Cleanup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
    finally:
        harvester.close()
        
if __name__ == "__main__":
    main()
