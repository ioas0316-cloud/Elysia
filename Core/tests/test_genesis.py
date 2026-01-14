import logging
import sys
# Configure logging to stdout so we can see it in terminal output
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.World.Creation.project_genesis import ProjectGenesis

def test_external_creation():
    print("🔨 [GENESIS TEST] Initializing ProjectGenesis...")
    
    # Use C:\game as authorized by user
    genesis = ProjectGenesis(external_root=r"C:\game")
    
    project_name = "hello_world"
    type_key = "WEB_APP"
    
    print(f"🏗️ Attempting to create '{project_name}' [Type: {type_key}]...")
    
    try:
        success = genesis.create_project(project_name, type_key)
        
        if success:
            print("\n✅ SUCCESS: Project created successfully.")
            print(f"   Target: C:\\game\\{project_name}")
            print("   Check for 'index.html', 'style.css', and '.elysia/soul_link.json'.")
        else:
            print("\n❌ FAILURE: Project creation returned False.")
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")

if __name__ == "__main__":
    test_external_creation()
