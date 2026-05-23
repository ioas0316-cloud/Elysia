import sys
import numpy as np
from pulse import EnergyPulse, CircuitGraph
from circuit_nodes import Base2Node, Base3Node, Base4Node

def build_test_circuit():
    """
    Builds a small circuit network comprising 2-Base, 3-Base, and 4-Base nodes.
    """
    graph = CircuitGraph()

    # Entrance point: A forgiving Ternary node
    n_entry = Base3Node("Gateway-3")

    # Split into a rigid logic path (Base2) and a fluid multidimensional path (Base4)
    n_logic = Base2Node("LogicGate-2")
    n_quantum = Base4Node("QuantumRotor-4")

    # Recombine into a final Ternary node
    n_exit = Base3Node("Synthesis-3")

    graph.add_node(n_entry)
    graph.add_node(n_logic)
    graph.add_node(n_quantum)
    graph.add_node(n_exit)

    # Wire them up
    graph.add_edge("Gateway-3", "LogicGate-2", weight=0.8)
    graph.add_edge("Gateway-3", "QuantumRotor-4", weight=1.2) # Amplified path
    graph.add_edge("LogicGate-2", "Synthesis-3", weight=1.0)
    graph.add_edge("QuantumRotor-4", "Synthesis-3", weight=1.0)

    return graph

def main():
    print("=========================================================")
    print("   Elysia Circuit Simulator - Pulse Injection Terminal   ")
    print("=========================================================")
    print("Booting Circuit Graph...")

    graph = build_test_circuit()

    print("Circuit network established: [Gateway-3] -> [LogicGate-2, QuantumRotor-4] -> [Synthesis-3]")
    print("Enter text to inject a Trigger Pulse. Type 'exit' or 'quit' to stop.")
    print("---------------------------------------------------------")

    # We maintain a base frequency that shifts over time based on user input length
    base_freq = 40.0

    while True:
        try:
            text = input("\n[INJECT TRIGGER PULSE] >>> ")
            if text.lower() in ['exit', 'quit']:
                break
            if not text.strip():
                continue

            # Convert text roughly to physical wave characteristics
            amplitude = min(10.0, len(text.split()) * 0.5 + 0.5)
            # Use ASCII sum for pseudo-random phase mapping
            phase = (sum(ord(c) for c in text) % 360) * (np.pi / 180.0)

            pulse = EnergyPulse(amplitude=amplitude, frequency=base_freq, phase=phase, origin_text=text)

            logs = graph.inject_pulse(pulse, "Gateway-3")

            for log in logs:
                print(log)

            # Shift frequency slightly for next pulse
            base_freq += np.random.uniform(-5.0, 5.0)
            base_freq = max(10.0, min(100.0, base_freq))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Terminal error: {e}")

if __name__ == "__main__":
    main()
