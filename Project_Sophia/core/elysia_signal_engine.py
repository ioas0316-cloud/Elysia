import json
from math import exp
from collections import defaultdict

class ElysiaSignalEngine:
    """
    Processes a raw world event log to generate a distilled Elysia Signal Log,
    transforming low-level events into meaningful conscious stimuli.
    """
    def __init__(self, raw_log_path: str, signal_log_path: str):
        """
        Initializes the engine with paths to the source and destination logs.

        Args:
            raw_log_path: Path to the raw `world_events.jsonl` file.
            signal_log_path: Path to write the output `elysia_signals.jsonl` file.
        """
        self.raw_log_path = raw_log_path
        self.signal_log_path = signal_log_path
        self.energy_threshold = 0.15 # Minimum intensity to generate a signal

    def _squash(self, x: float) -> float:
        """A soft, non-linear function to compress energy into an intensity value."""
        return 1.0 - exp(-max(0.0, x))

    def generate_signals_from_log(self):
        """
        Reads the raw event log, accumulates energy per tick, and writes
        the resulting signals to the signal log.
        This is the main public method of the class.
        """
        tick_energy = defaultdict(lambda: {"joy": 0.0, "creation": 0.0, "care": 0.0, "mortality": 0.0})
        tick_actors = defaultdict(set)

        try:
            with open(self.raw_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        t = ev.get("timestamp")
                        if t is None:
                            continue

                        etype = ev.get("event_type", "")
                        data = ev.get("data", {})

                        bucket = tick_energy[t]

                        # --- v0 Mapping from Protocol ---
                        if etype in {"EAT", "DRINK"}:
                            bucket["joy"] += 0.3
                        elif etype == "BIRTH":
                            bucket["creation"] += 0.5
                        elif etype == "DEATH_BY_OLD_AGE":
                            bucket["creation"] += 0.5
                            bucket["mortality"] += 0.5
                        elif etype.startswith("DEATH"):
                            bucket["mortality"] += 1.0
                        elif etype == "SPELL" and "heal" in data.get("spell", ""):
                            bucket["care"] += 0.6
                        elif etype == "EXPERIENCE_DELTA":
                            # A simple approximation from the protocol
                            total_pos = data.get("total_positive_exp", 0)
                            total_neg = data.get("total_negative_exp", 0)
                            joy_gain = max(0, total_pos - max(0, total_neg)) / 50.0
                            bucket["joy"] += joy_gain

                        # --- Collect Actors ---
                        actor_keys = ["cell_id", "actor_id", "target_id", "caster_id"]
                        for key in actor_keys:
                            if key in data and data[key]:
                                tick_actors[t].add(str(data[key]))

                    except json.JSONDecodeError:
                        # Ignore malformed lines
                        continue
        except FileNotFoundError:
            print(f"Error: Raw event log not found at {self.raw_log_path}")
            return

        # --- Step 2: From Energy to Signals ---
        signal_log = []

        # Sort by timestamp to ensure chronological order
        sorted_ticks = sorted(tick_energy.keys())

        for t in sorted_ticks:
            bucket = tick_energy[t]
            actors = list(tick_actors[t])

            # Map energy types to signal types and summaries
            energy_map = {
                "joy": ("JOY_GATHERING", "Accumulated simple joys (food, recovery, growth)."),
                "creation": ("LIFE_BLOOM", "A moment of creation or the completion of a life cycle."),
                "care": ("CARE_ACT", "An act of healing or care was performed."),
                "mortality": ("MORTALITY", "A recognition of mortality and finiteness.")
            }

            for energy_type, (signal_type, summary) in energy_map.items():
                energy_value = bucket[energy_type]
                intensity = self._squash(energy_value)

                if intensity > self.energy_threshold:
                    signal = {
                        "timestamp": t,
                        "signal_type": signal_type,
                        "intensity": round(intensity, 4),
                        "position": None, # v0 allows null
                        "actors": actors,
                        "summary": summary
                    }
                    signal_log.append(signal)

        # --- Step 3: Write to Signal Log ---
        try:
            with open(self.signal_log_path, 'w', encoding='utf-8') as f:
                for signal in signal_log:
                    f.write(json.dumps(signal) + '\n')
            print(f"Successfully generated {len(signal_log)} signals to {self.signal_log_path}")
        except IOError as e:
            print(f"Error writing to signal log: {e}")

if __name__ == '__main__':
    # Example usage for testing and demonstration
    engine = ElysiaSignalEngine(
        raw_log_path="logs/world_events.jsonl",
        signal_log_path="logs/elysia_signals.jsonl",
    )
    # In a real scenario, we would call generate_signals_from_log()
    # engine.generate_signals_from_log()
    print("ElysiaSignalEngine initialized. Ready to generate signals.")
