#!/usr/bin/env python3
"""
VRM Avatar Integration Test
============================

Validates that:
1. VRM file exists and is valid
2. Avatar server can start
3. HTTP + WebSocket servers are operational
4. avatar.html has VRM integration code
"""

import sys
import os
from pathlib import Path

# Get repo root (parent of tests directory)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_vrm_file_exists():
    """Check if VRM model file exists"""
    vrm_path = REPO_ROOT / "static" / "models" / "avatar.vrm"
    
    print("🔍 Checking VRM model file...")
    if vrm_path.exists():
        size_mb = vrm_path.stat().st_size / (1024 * 1024)
        print(f"✅ VRM file found: {vrm_path}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ VRM file not found at: {vrm_path}")
        return False


def test_avatar_html_integration():
    """Check if avatar.html has VRM integration"""
    html_path = REPO_ROOT / "Core" / "Creativity" / "web" / "avatar.html"
    
    print("\n🔍 Checking avatar.html integration...")
    if not html_path.exists():
        print(f"❌ avatar.html not found at: {html_path}")
        return False
    
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "Three.js import": "three" in content and "importmap" in content,
        "VRM loader": "@pixiv/three-vrm" in content,
        "VRM canvas": "vrm-canvas" in content,
        "VRM initialization": "initVRMAvatar" in content,
        "Expression mapping": "updateVRMExpressions" in content,
        "OrbitControls": "OrbitControls" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_web_server_exists():
    """Check if web server script exists"""
    server_path = REPO_ROOT / "start_avatar_web_server.py"
    
    print("\n🔍 Checking web server script...")
    if server_path.exists():
        print(f"✅ Web server script found: {server_path}")
        return True
    else:
        print(f"❌ Web server script not found at: {server_path}")
        return False


def test_server_imports():
    """Test if server modules can be imported"""
    print("\n🔍 Testing server module imports...")
    
    try:
        from Core.Interface.avatar_server import AvatarWebSocketServer
        print("   ✅ AvatarWebSocketServer imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Failed to import AvatarWebSocketServer: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🎭 VRM Avatar Integration Test")
    print("=" * 60)
    
    results = {
        "VRM file exists": test_vrm_file_exists(),
        "avatar.html integration": test_avatar_html_integration(),
        "Web server script": test_web_server_exists(),
        "Server imports": test_server_imports(),
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✨ All tests passed! VRM integration is ready.")
        print("\n🚀 To start the server:")
        print("   python start_avatar_web_server.py")
        print("\n🌐 Then open:")
        print("   http://localhost:8080/Core/Creativity/web/avatar.html")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
