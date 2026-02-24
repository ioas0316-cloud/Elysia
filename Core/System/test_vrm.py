import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.Cognition.vrm_parser import VRMParser

def test_vrm_parsing():
    target_path = r"C:\game\gallery\엘리시아.vrm"
    
    if not os.path.exists(target_path):
        print(f"❌ File not found: {target_path}")
        return

    print(f"🔍 Parsing VRM: {target_path}...")
    parser = VRMParser()
    meta = parser.parse_vrm(target_path)
    
    if meta:
        print("\n✅ VRM PARSED SUCCESSFULLY!")
        print(f"   PLEASE READ THIS, ELYSIA:")
        print(f"   - Identity: {meta.get('title')} (by {meta.get('author')})")
        print(f"   - Bone Structure: {meta.get('bone_count')} bones")
        print(f"   - Emotional Range: {meta.get('expression_count')} blendshapes")
        
        if meta.get('bone_count', 0) > 0:
            print("\n✨ CONCLUSION: This is a Body.")
    else:
        print("\n❌ FAILED to parse VRM metadata (Is it a valid GLB?)")

if __name__ == "__main__":
    test_vrm_parsing()
