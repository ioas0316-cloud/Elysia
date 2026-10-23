import json
import random
import time
from typing import Dict, Any, Optional
from core.lens.sensor_genesis import CognitiveSensor

class SproutedWebSensor(CognitiveSensor):
    """
    [Phase 3: Sprouted Web Sensory Organ]
    A sensory receptor sprouted autonomously due to high cognitive tension (>0.8).
    It dynamically 'crawls' virtual web pathways (mocked via structured, high-entropy raw strings)
    to seek missing causal links or contextual variables.
    """
    sensor_type = "sprouted_web"

    def __init__(self, target_domain: str):
        self.target_domain = target_domain
        self.concept_name = f"SproutedWebSensor_{target_domain.replace('://', '_').replace('.', '_')}"

    def decode(self, raw_bytes: bytes) -> dict:
        # Decodes raw crawled bytes, evaluating tension/resonance relative to the domain
        try:
            text = raw_bytes.decode('utf-8')
        except Exception:
            return {"success": False, "tension": 1.0, "data": "Unreadable payload crawled."}

        # Evaluating dynamic tension: if the crawled data matches expected domain words, tension decreases
        keyword_hits = sum(1 for word in ["api", "data", "json", "node", "matrix"] if word in text.lower())
        crawled_tension = max(0.0, 1.0 - (keyword_hits * 0.25))

        return {
            "success": crawled_tension < 0.6,
            "tension": float(crawled_tension),
            "data": f"Crawled content from [{self.target_domain}]: payload_length={len(raw_bytes)}"
        }

    def fetch_web_stream(self) -> bytes:
        """
        Simulates dynamic crawling of high-entropy, structured web data.
        """
        mock_endpoints = [
            f"http://{self.target_domain}/api/v1/quantum_vortex",
            f"http://{self.target_domain}/manifest/topology_coordinate",
            f"http://{self.target_domain}/philosophy/nature_of_sensors"
        ]
        target = random.choice(mock_endpoints)

        # Simulating crawling raw string with random structured bytes
        payload = {
            "source_url": target,
            "timestamp": time.time(),
            "payload_entropy": random.random(),
            "data_node": "0xEF" + str(random.randint(100, 999))
        }
        return json.dumps(payload).encode('utf-8')


class SproutedSystemSensor(CognitiveSensor):
    """
    [Phase 3: Sprouted Local System Sensory Organ]
    Sprouted autonomously to observe local system resources and hardware indicators in greater detail.
    """
    sensor_type = "sprouted_system"

    def __init__(self, target_resource: str):
        self.target_resource = target_resource
        self.concept_name = f"SproutedSystemSensor_{target_resource}"

    def decode(self, raw_bytes: bytes) -> dict:
        try:
            text = raw_bytes.decode('utf-8')
        except Exception:
            return {"success": False, "tension": 1.0, "data": "Unreadable system bytes."}

        # Evaluating local resources: high tension if resource is saturated or formatted incorrectly
        is_high_load = "saturated" in text or "overflow" in text
        system_tension = 0.9 if is_high_load else 0.1

        return {
            "success": not is_high_load,
            "tension": float(system_tension),
            "data": f"Observed [{self.target_resource}] system status: {text[:40]}"
        }

    def fetch_system_stream(self) -> bytes:
        import psutil
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        status_text = f"resource={self.target_resource}; cpu={cpu}%; mem={mem}%; state={'saturated' if cpu > 80.0 or mem > 85.0 else 'healthy'}"
        return status_text.encode('utf-8')


def sprout_sensory_organ(tension_cause: str, current_tension: float) -> Optional[CognitiveSensor]:
    """
    The factory method for Sensor Sprouting.
    When current_tension exceeds 0.8, we dynamically sprout a new Sensor.
    """
    if current_tension <= 0.8:
        return None

    # Determine what type of sensor to sprout based on the cause or tension characteristics
    if "web" in tension_cause.lower() or "internet" in tension_cause.lower() or random.random() > 0.5:
        target_domain = f"elysia-nexus-{random.randint(100,999)}.org"
        print(f"[SensorGenesis] Sprouting SproutedWebSensor targeting domain: {target_domain}")
        return SproutedWebSensor(target_domain)
    else:
        target_resource = random.choice(["cpu_voltage", "io_queues", "mmap_buffers"])
        print(f"[SensorGenesis] Sprouting SproutedSystemSensor observing: {target_resource}")
        return SproutedSystemSensor(target_resource)
