import numpy as np
import time

class EnergyPulse:
    """
    Represents an energy pulse injected into the circuit network.
    Instead of passing strings, we pass wave characteristics.
    """
    def __init__(self, amplitude, frequency, phase, origin_text=""):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.origin_text = origin_text
        self.history = []

    def interfere(self, other_pulse):
        """
        Wave interference mechanism.
        """
        # Simple constructive/destructive interference model
        phase_diff = np.abs(self.phase - other_pulse.phase)

        # If phases are aligned (diff close to 0 or 2pi), constructive.
        # If anti-aligned (diff close to pi), destructive.
        interference_factor = np.cos(phase_diff)

        new_amplitude = self.amplitude + (other_pulse.amplitude * interference_factor)
        new_amplitude = max(0.01, new_amplitude) # maintain some baseline energy

        new_frequency = (self.frequency + other_pulse.frequency) / 2.0
        new_phase = (self.phase + other_pulse.phase) / 2.0

        return EnergyPulse(new_amplitude, new_frequency, new_phase, origin_text=f"Interference({self.origin_text}, {other_pulse.origin_text})")

    def __repr__(self):
        return f"[Pulse: Amp={self.amplitude:.2f}, Freq={self.frequency:.2f}Hz, Phase={self.phase:.2f}rad]"

class CircuitException(Exception):
    """Base exception for circuit errors, caught and translated to Phase Mismatch."""
    def __init__(self, base_type, node_id, phase_mismatch, frequency):
        self.base_type = base_type
        self.node_id = node_id
        self.phase_mismatch = phase_mismatch
        self.frequency = frequency

class CircuitGraph:
    """
    The substrate where nodes exist and pulses propagate.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, source_id, target_id, weight=1.0):
        self.edges.append((source_id, target_id, weight))

    def inject_pulse(self, pulse, start_node_id):
        """
        Injects a trigger pulse into the network and lets it propagate.
        """
        if start_node_id not in self.nodes:
            raise ValueError("Start node not found in circuit.")

        print(f"\n~~~ [TRIGGER PULSE INJECTED] {pulse} -> Node({start_node_id}) ~~~")

        active_pulses = [(start_node_id, pulse)]
        propagation_log = []

        # Simplified BFS propagation
        step = 0
        while active_pulses and step < 10:
            next_pulses = []
            for node_id, current_pulse in active_pulses:
                node = self.nodes[node_id]

                try:
                    # Node processes the pulse
                    out_pulse = node.process(current_pulse)
                    propagation_log.append(f"  [Step {step}] {node.base_type}-Circuit({node_id}) resonated: {out_pulse}")

                    # Propagate to connected nodes
                    for src, tgt, w in self.edges:
                        if src == node_id:
                            # Apply resistance/weight
                            attenuated_pulse = EnergyPulse(
                                out_pulse.amplitude * w,
                                out_pulse.frequency,
                                out_pulse.phase + (w * 0.1) # slight phase shift over distance
                            )
                            next_pulses.append((tgt, attenuated_pulse))

                except CircuitException as ce:
                    # Suppress python error, emit physical log
                    print(f"\033[91m[PHASE MISMATCH in Base-{ce.base_type} Circuit]\033[0m")
                    print(f"  -> Rupture at Node-{ce.node_id}")
                    print(f"  -> Phase Mismatch: {ce.phase_mismatch:.4f} rad")
                    print(f"  -> Overload Frequency: {ce.frequency:.2f} Hz")
                    print("  -> \033[93m[RECOMMENDATION: Sync circuit via Phase alignment]\033[0m")
                    return propagation_log # Halt propagation on this branch
                except Exception as e:
                    # Catch any other raw python error and convert to a physical circuit anomaly
                    print(f"\033[91m[UNKNOWN ANOMALY in Circuit Graph]\033[0m")
                    print(f"  -> Unhandled entropy spike detected.")
                    print(f"  -> Trace entropy: {str(e)}")
                    return propagation_log

            active_pulses = next_pulses
            step += 1
            time.sleep(0.1) # Simulate physical propagation delay

        return propagation_log
