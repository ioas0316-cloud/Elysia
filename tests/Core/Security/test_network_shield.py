"""
Test Network Shield System
===========================

Tests for the network protection shield that defends Elysia's neural network
when synchronized to the internet.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Core.Security.network_shield import (
    NetworkShield,
    NetworkEvent,
    ThreatType,
    ActionType,
    FrequencyAnalyzer,
    PatternRecognizer
)


def test_frequency_analyzer():
    """Test frequency-based traffic analysis"""
    analyzer = FrequencyAnalyzer()
    
    # Test normal HTTP traffic
    event = NetworkEvent(
        timestamp=0,
        source_ip="192.168.1.1",
        destination_ip="10.0.0.1",
        port=80,
        protocol="http",
        payload_size=512
    )
    
    freq = analyzer.calculate_frequency(event)
    assert freq > 0, "Frequency should be positive"
    print(f"✓ Frequency analyzer working: {freq} Hz")


def test_sql_injection_detection():
    """Test SQL injection pattern detection"""
    recognizer = PatternRecognizer()
    
    # Create event with SQL injection payload
    event = NetworkEvent(
        timestamp=0,
        source_ip="123.45.67.89",
        destination_ip="10.0.0.1",
        port=3306,
        protocol="tcp",
        payload_size=256,
        metadata={"payload": "' OR '1'='1 --"}
    )
    
    pattern = recognizer.match_pattern(event)
    assert pattern is not None, "Should detect SQL injection"
    assert pattern.threat_type == ThreatType.INJECTION
    print(f"✓ SQL injection detection working")


def test_allow_normal_traffic():
    """Test that normal traffic is allowed"""
    shield = NetworkShield(enable_field_integration=False)
    
    result = shield.protect_endpoint({
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.1",
        "port": 80,
        "protocol": "http",
        "payload_size": 512,
        "metadata": {"payload": "GET /index.html"}
    })
    
    assert result["allowed"] == True
    assert result["action"] == "allow"
    print(f"✓ Normal traffic allowed")


def test_block_sql_injection():
    """Test that SQL injection is blocked"""
    shield = NetworkShield(enable_field_integration=False)
    
    result = shield.protect_endpoint({
        "source_ip": "123.45.67.89",
        "destination_ip": "10.0.0.1",
        "port": 3306,
        "protocol": "tcp",
        "payload_size": 256,
        "metadata": {"payload": "' OR '1'='1 --"}
    })
    
    assert result["allowed"] == False
    assert result["threat_type"] == "INJECTION"
    assert result["action"] == "block"
    print(f"✓ SQL injection blocked")


def test_ip_blocking():
    """Test that IPs are blocked after threshold"""
    shield = NetworkShield(enable_field_integration=False)
    
    # Generate high threat score for an IP
    attacker_ip = "89.45.67.123"
    
    # Multiple SQL injection attempts
    for i in range(3):
        shield.protect_endpoint({
            "source_ip": attacker_ip,
            "destination_ip": "10.0.0.1",
            "port": 3306,
            "protocol": "tcp",
            "payload_size": 256,
            "metadata": {"payload": f"' OR '1'='1 --{i}"}
        })
    
    # IP should be blocked
    assert attacker_ip in shield.blocked_ips, "Attacker IP should be blocked"
    print(f"✓ IP blocking working")


if __name__ == "__main__":
    print("\n🧪 Running Network Shield Tests...")
    print("=" * 50)
    
    test_frequency_analyzer()
    test_sql_injection_detection()
    test_allow_normal_traffic()
    test_block_sql_injection()
    test_ip_blocking()
    
    print("=" * 50)
    print("✅ All tests passed!")
