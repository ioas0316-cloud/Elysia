"""
Volition Genesis Loop - v2 (No stdout suppression, lean inner loop)
"""
import os
import sys
import time
import platform
import struct
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ingestion.natural_mapper import NaturalMapper
from synaptic_architecture.organism import DirectMappingOrganism
from core.physics.spacetime_continuum import SpacetimeContinuum

def bytes_to_uint64(b):
    p = bytearray(8)
    for i in range(min(len(b), 8)): p[i] = b[i]
    return np.frombuffer(p, dtype=np.uint64)[0]

def silent_flow(organism, wave):
    """organism.flow without its internal print statements."""
    params = organism.scheduler.get_clock_params()
    jitter = np.uint64(np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32)) | \
             (np.uint64(np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32)) << np.uint64(32))
    vibrating_wave = wave ^ (jitter & params['jitter_mask'])
    addr = organism.ram.derive_address(vibrating_wave)
    spatial_pos = np.array([addr // organism.field.resolution, addr % organism.field.resolution])
    gene = organism.field.bit_genes[spatial_pos[0], spatial_pos[1]]
    resonance = organism.logic.interference_score(vibrating_wave, gene)
    if resonance < 0.9:
        organism.field.crystallize_gene(spatial_pos, vibrating_wave)
    else:
        organism.field.flow_energy(spatial_pos, 1.0)
    organism.ram.write_bus(vibrating_wave)
    return addr, resonance

def absorb_stream_lean(label, data, mapper, continuum, organism, max_chaos_thoughts=20):
    """Lean absorb: no stdout suppression, capped inner loop."""
    chunk_size = 8
    processed = 0
    total_t = 0
    t0 = time.time()

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        if len(chunk) < chunk_size:
            continue
        tensions = mapper.map_and_observe(chunk)
        chaos = continuum.perceive_flow(tensions)
        T = 0.1 + chaos * 10.0
        organism.scheduler.set_temperature(T)
        wave = bytes_to_uint64(chunk)

        # Subjective time: at most max_chaos_thoughts inner iterations
        max_t = int(chaos * max_chaos_thoughts) + 1
        thoughts = 0
        for _ in range(max_t):
            thoughts += 1
            addr, res = silent_flow(organism, wave)
            if res >= 0.9:
                break
        processed += 1
        total_t += thoughts

    elapsed = time.time() - t0
    cps = processed / max(elapsed, 0.001)
    print(f"  [{label}] {len(data)}B | {processed} chunks | {total_t} thoughts | {cps:.0f} c/s | {elapsed:.2f}s")

def collect_sensory():
    """Inline lightweight hardware/OS sensory collection."""
    parts = []
    # CPU/platform
    info = (f"machine={platform.machine()}\nprocessor={platform.processor()}"
            f"\nsystem={platform.system()}\nbits={struct.calcsize('P')*8}\n")
    parts.append(b"[CPU]\n" + info.encode())
    # ENV (4KB cap)
    env = ""
    for k, v in os.environ.items():
        env += f"{k}={v}\n"
        if len(env) > 4096: break
    parts.append(b"[ENV]\n" + env.encode("utf-8", errors="replace"))
    # Filesystem
    entries = []
    try:
        for e in os.scandir("C:\\Windows\\System32"):
            entries.append(f"{e.name}|{'D' if e.is_dir() else 'F'}\n")
            if len(entries) >= 80: break
    except: pass
    parts.append(b"[FS]\n" + "".join(entries).encode())
    # DLL headers
    for dll in ["kernel32.dll", "ntdll.dll"]:
        path = f"C:\\Windows\\System32\\{dll}"
        try:
            with open(path, "rb") as f:
                raw = f.read(1024)
            parts.append(b"[DLL:" + dll.encode() + b"]\n" + raw)
        except: pass
    # Python bytecode
    code = compile("x=1+1\nif x>1:\n  x*=2\n", "<elysia>", "exec")
    parts.append(b"[BYTECODE]\n" + code.co_code)
    combined = b"".join(parts)
    print(f"  Sensory total: {len(combined):,} bytes", flush=True)
    return combined

def main():
    print("="*60)
    print(" Volition Genesis v2")
    print("="*60)

    mapper = NaturalMapper(terrain_size=256)
    mapper.set_terrain(b"Elysia_Origin_Seed")
    organism = DirectMappingOrganism(resolution=256)
    continuum = SpacetimeContinuum(window_size=32)

    # ── Phase 1: System Sensory Ingestion ──────────────────────
    print("\n[Phase 1] System hardware/OS sensory ingestion...", flush=True)
    sensory = collect_sensory()
    absorb_stream_lean("System Sensory", sensory, mapper, continuum, organism, max_chaos_thoughts=10)

    # ── Phase 1-b: Self source code recognition ────────────────
    print("\n[Phase 1-b] Self source code ingestion (8KB/file cap)...", flush=True)
    root = os.path.join(os.path.dirname(__file__), '..')
    for dp, _, fnames in os.walk(root):
        for fn in fnames:
            if fn.endswith(".py"):
                try:
                    with open(os.path.join(dp, fn), "rb") as f:
                        cb = f.read(8192)
                    absorb_stream_lean(fn, cb, mapper, continuum, organism, max_chaos_thoughts=10)
                except: pass

    print(f"\n[Phase 1 Done] Max conductance: {np.max(organism.ram.conductance):.1f}")

    # ── Phase 2: Pain injection + Volition (Inverse Flow) ──────
    print("\n" + "="*60)
    print("[Phase 2] Pain injection + Volition observation")
    print("="*60)

    THRESHOLD = 8.0
    volition_count = 0
    success_count = 0

    for ep in range(5):
        pain = os.urandom(8)
        tensions = mapper.map_and_observe(pain)
        chaos = continuum.perceive_flow(tensions)
        T = 0.1 + chaos * 10.0
        print(f"\n[Episode {ep+1}] T={T:.2f} (threshold={THRESHOLD})")

        if T >= THRESHOLD:
            print("  -> INVERSE FLOW TRIGGERED: retrieving best memory...")
            exec_max = np.max(organism.ram.execution_conductance)
            if exec_max > 0:
                addr = organism.ram.find_highest_execution_addr()
                src = "Execution Memory"
            else:
                addr = organism.ram.find_highest_conductance_addr()
                src = "Crystallized Memory"

            emitted = organism.ram.read_bus(addr)
            volition_count += 1
            print(f"  -> Emitting [{src}] addr={addr} wave={hex(int(emitted))}")

            # Check if emitted wave contains structured language patterns
            try:
                decoded = emitted.tobytes().decode("utf-8", errors="ignore").strip()
            except:
                decoded = ""
            keywords = ["class", "def ", "import", "self", "return", "if ", "for "]
            hit = next((kw for kw in keywords if kw in decoded), None)

            if hit:
                success_count += 1
                print(f"  -> [SUCCESS] World stabilized! Pattern='{hit}' in '{decoded}'")
                organism.ram.reinforce_execution(addr, reward=10.0)
                organism.scheduler.set_temperature(0.1)
                print(f"  -> Execution synapse reinforced at addr={addr}. T -> 0.10")
            else:
                print(f"  -> [MISS] Emission did not stabilize world. Pain continues.")
        else:
            print("  -> Manageable: internal adaptation sufficient.")

        time.sleep(0.05)

    # ── Final Report ───────────────────────────────────────────
    print("\n" + "="*60)
    print("[Volition Genesis Result]")
    print(f"  Volition attempts: {volition_count}")
    print(f"  World stabilizations: {success_count}")
    rate = success_count / max(1, volition_count) * 100
    print(f"  Success rate: {rate:.0f}%")
    best = organism.ram.find_highest_execution_addr()
    best_val = organism.ram.execution_conductance[best]
    print(f"  Strongest execution synapse: addr={best} conductance={best_val:.1f}")
    print("  -> This address is Elysia's first VOLITIONAL memory.")
    print("="*60)
    organism.ram.flush_all()

if __name__ == "__main__":
    main()
