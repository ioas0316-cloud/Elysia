"""
Integration test for Avatar Server WebSocket functionality
Tests actual WebSocket communication between server and client
"""

import asyncio
import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import websockets
except ImportError:
    print("❌ websockets not installed. Run: pip install websockets")
    sys.exit(1)


async def test_server_connection():
    """Test basic server connection"""
    print("🧪 Testing server connection...")
    
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to server")
            
            # Wait for initial state message
            message = await asyncio.wait_for(ws.recv(), timeout=2.0)
            state = json.loads(message)
            
            assert "expression" in state, "State should contain expression"
            assert "spirits" in state, "State should contain spirits"
            print("✅ Received initial state")
            
            return True
    
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for initial state")
        return False
    except ConnectionRefusedError:
        print("❌ Connection refused. Is the server running?")
        print("   Start server with: python start_avatar_server.py")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


async def test_chat_message():
    """Test sending a chat message"""
    print("🧪 Testing chat message...")
    
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as ws:
            # Wait for initial state
            await ws.recv()
            
            # Send chat message
            await ws.send(json.dumps({
                "type": "text",
                "content": "Hello, Elysia!"
            }))
            print("✅ Sent chat message")
            
            # Wait for response
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)
            
            assert data.get("type") == "speech", "Response should be speech type"
            assert "content" in data, "Response should have content"
            assert "spirits" in data, "Response should have spirits"
            
            print(f"✅ Received response: {data['content'][:50]}...")
            return True
    
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for response")
        return False
    except ConnectionRefusedError:
        print("❌ Connection refused")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_emotion_trigger():
    """Test emotion trigger message"""
    print("🧪 Testing emotion trigger...")
    
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as ws:
            # Wait for initial state
            await ws.recv()
            
            # Send emotion trigger
            await ws.send(json.dumps({
                "type": "emotion",
                "emotion": "hopeful",
                "intensity": 0.8
            }))
            print("✅ Sent emotion trigger")
            
            # Wait a bit for processing
            await asyncio.sleep(0.2)
            
            # Get updated state
            state_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            state = json.loads(state_msg)
            
            print("✅ Emotion processed, state updated")
            return True
    
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for state update")
        return False
    except ConnectionRefusedError:
        print("❌ Connection refused")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_state_broadcast():
    """Test continuous state broadcasting"""
    print("🧪 Testing state broadcast (3 frames)...")
    
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as ws:
            # Receive 3 state updates
            for i in range(3):
                message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                state = json.loads(message)
                
                assert "expression" in state, f"Frame {i}: missing expression"
                assert "spirits" in state, f"Frame {i}: missing spirits"
            
            print("✅ Received 3 state broadcasts")
            return True
    
    except asyncio.TimeoutError:
        print("❌ Timeout waiting for state broadcasts")
        return False
    except ConnectionRefusedError:
        print("❌ Connection refused")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("  Avatar Server Integration Tests")
    print("=" * 60)
    print()
    print("⚠️  Make sure the server is running:")
    print("    python start_avatar_server.py")
    print()
    print("Waiting 2 seconds for you to start the server...")
    await asyncio.sleep(2)
    print()
    
    tests = [
        test_server_connection,
        test_state_broadcast,
        test_emotion_trigger,
        test_chat_message,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} exception: {e}")
            failed += 1
        
        print()
        await asyncio.sleep(0.5)  # Small delay between tests
    
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


def main():
    """Main entry point"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
