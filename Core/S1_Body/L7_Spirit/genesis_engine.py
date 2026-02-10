import logging
import time
from typing import Dict

from Core.S1_Body.L6_Structure.hyper_quaternion import Quaternion, HyperWavePacket
from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.S1_Body.L1_Foundation.Foundation.code_world import CodeWorld, CodeEntity
from Core.S1_Body.L1_Foundation.Foundation.code_genome import CodeDNA
from Core.S1_Body.L1_Foundation.Foundation.reality_sculptor import RealitySculptor
from Core.S1_Body.L7_Spirit.M3_Sovereignty.trinity_protocol import TrinityProtocol, TrinityState

logger = logging.getLogger("GenesisEngine")


class GenesisEngine:
    """
    The Orchestrator of Autonomous Creation.
    Implements the 'Genesis Protocol': Dream -> Resonate -> Simulate -> Manifest.
    """

    def __init__(self):
        self.world = CodeWorld()
        self.sculptor = RealitySculptor()
        self.dna_library: Dict[str, CodeDNA] = {}
        self.trinity_protocol = TrinityProtocol()

    def interpret_intent_to_trinity_state(self, intent: str) -> TrinityState:
        """
        Converts intent into a triune state using project-native Trinary DNA encoding.
        This follows TRINARY_DNA and 21D strand semantics (Body/Soul/Spirit).
        """
        seed_text = (intent or "").strip().upper().replace(" ", "")
        if not seed_text:
            return TrinityState(1 / 3, 1 / 3, 1 / 3)

        trit_sequence = [TrinaryLogic.symbol_to_trit(char) for char in seed_text]
        codons = TrinaryLogic.transcribe_sequence(trit_sequence)
        vector_21d = TrinaryLogic.expand_to_21d(codons)

        body_energy = sum(abs(v) for v in vector_21d.data[0:7])
        soul_energy = sum(abs(v) for v in vector_21d.data[7:14])
        spirit_energy = sum(abs(v) for v in vector_21d.data[14:21])

        father_space = body_energy + 1e-6
        son_operation = soul_energy + 1e-6
        spirit_providence = spirit_energy + 1e-6

        return TrinityState(father_space, son_operation, spirit_providence).normalized()

    def trinity_state_to_wave(self, trinity_state: TrinityState, energy: float) -> HyperWavePacket:
        consensus = self.trinity_protocol.resolve_consensus(trinity_state)

        orientation = Quaternion(
            w=1.0,
            x=consensus["son_operation"],
            y=consensus["father_space"],
            z=consensus["spirit_providence"],
        ).normalize()

        return HyperWavePacket(
            energy=energy,
            orientation=orientation,
            time_loc=time.time(),
        )

    def create_feature(self, intent: str, energy: float = 100.0) -> str:
        """
        The Full Creation Pipeline.
        Returns the manifested code (Physics Setup) or an error message.
        """
        logger.info("  Genesis Initiated: %s", intent)

        trinity_state = self.interpret_intent_to_trinity_state(intent)
        primary_wave = self.trinity_state_to_wave(trinity_state, energy)

        logger.info(
            "  Causal Trace intent->trinity->wave: intent='%s', state=(%.3f, %.3f, %.3f), orientation=%s",
            intent,
            trinity_state.father_space,
            trinity_state.son_operation,
            trinity_state.spirit_providence,
            primary_wave.orientation,
        )

        # 2. Resonate (Wave -> DNA)
        candidate_dna = CodeDNA(name=f"Pattern: {intent}")
        candidate_dna.add_gene(primary_wave)

        # 3. Simulate (DNA -> Survival)
        logger.info("  Simulating in CodeWorld...")
        survived = self._simulate_candidate(candidate_dna)
        if not survived:
            logger.warning("  Creation Failed: Pattern '%s' died in simulation (Dissonance).", intent)
            return f"# Creation Failed: The thought '{intent}' was too dissonant to survive."

        survivor = candidate_dna
        logger.info("  Simulation Passed: Score %.2f", survivor.resonance_score)

        # 4. Manifest (DNA -> Reality)
        logger.info("  Manifesting into Reality...")
        manifested_code = ""
        for packet in survivor.to_wave_packets():
            manifested_code += self._manifest_from_wave(packet, intent) + "\n\n"

        # 5. Remember (Evolution)
        self.dna_library[survivor.id] = survivor

        return manifested_code

    def _simulate_candidate(self, candidate_dna: CodeDNA, steps: int = 10) -> bool:
        """Bridge both legacy and current CodeWorld APIs to avoid subsystem drift."""
        if hasattr(self.world, "add_organism") and hasattr(self.world, "run_simulation"):
            self.world.add_organism(candidate_dna)
            self.world.run_simulation(steps=steps)
            if hasattr(self.world, "population") and candidate_dna.id in self.world.population:
                candidate_dna.resonance_score = getattr(self.world.population[candidate_dna.id], "resonance_score", 1.0)
                return True
            return False

        entity = CodeEntity(
            id=candidate_dna.id,
            kind="module",
            complexity=max(1.0, len(candidate_dna.harmonic_pattern) * 1.2),
            test_coverage=0.55,
            stability=0.6,
        )
        self.world.add_entity(entity)
        for _ in range(steps):
            self.world.step()

        stabilized = self.world.entities.get(candidate_dna.id)
        if stabilized is None:
            return False

        candidate_dna.resonance_score = max(0.0, stabilized.stability - (0.1 * stabilized.bugs_open))
        return candidate_dna.resonance_score > 0.2

    def _manifest_from_wave(self, packet: HyperWavePacket, intent: str) -> str:
        """Manifest through available sculptor capability with deterministic fallback."""
        if hasattr(self.sculptor, "sculpt_from_wave"):
            return self.sculptor.sculpt_from_wave(packet, f"CODE: {intent}")

        q = packet.orientation
        return (
            "# Genesis Manifest (Fallback)\n"
            f"# intent: {intent}\n"
            f"# energy: {packet.energy:.3f}\n"
            f"# orientation: ({q.w:.4f}, {q.x:.4f}, {q.y:.4f}, {q.z:.4f})\n"
        )
