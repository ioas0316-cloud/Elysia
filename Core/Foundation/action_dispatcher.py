"""
Action Dispatcher (행동 지휘소)
===============================

"Thoughts become Actions. Actions become Reality."

This module routes high-level intents (e.g., "LEARN:Quantum") to the appropriate
cortex or organ. It decouples the 'Soul' (living_elysia.py) from the 'Body' (Implementation).
"""

import time
import logging
import random
import os
from pathlib import Path
from Core.Foundation.web_server import WebServer, incoming_messages

logger = logging.getLogger("ActionDispatcher")

class ActionDispatcher:
    def __init__(self, brain, web, media, hologram, sculptor, transceiver, social, user_bridge, quantum_reader, dream_engine, memory, architect, synapse, shell, resonance, sink):
        self.brain = brain
        self.web = web
        self.media = media
        self.hologram = hologram
        self.sculptor = sculptor
        self.transceiver = transceiver
        self.social = social
        self.user_bridge = user_bridge
        self.quantum_reader = quantum_reader
        self.dream_engine = dream_engine
        self.memory = memory
        self.architect = architect
        self.synapse = synapse
        self.shell = shell
        self.resonance = resonance
        self.shell = shell
        self.resonance = resonance
        self.sink = sink
        self.web_server = None # Lazy initialization

    def dispatch(self, step: str):
        """
        Executes a single step of the narrative plan.
        Format: "ACTION:Detail"
        """
        parts = step.split(":")
        action = parts[0]
        detail = parts[1] if len(parts) > 1 else ""
        
        print(f"\n🚀 Executing Narrative Step: {step}")
        
        # Calculate Physics (Work/Friction)
        self._apply_physics(step)
        
        # [The Garden] Check for Web Messages
        if incoming_messages:
            msg = incoming_messages.pop(0)
            print(f"   📨 Incoming Web Message: {msg}")
            
            # Use Brain's communicate method for intelligent response
            response = self.brain.communicate(msg)
            
            # Store in memory
            self.brain.memory_field.append(f"User said: {msg}")
            self.brain.memory_field.append(f"I replied: {response}")
            
            # Update web state with response
            if self.web_server:
                self.web_server.update_state(
                    thought=f"💬 {response}",
                    energy=self.resonance.total_energy,
                    entropy=self.resonance.entropy
                )
        
        # Dispatch Logic
        method_name = f"_handle_{action.lower()}"
        if hasattr(self, method_name):
            handler = getattr(self, method_name)
            try:
                handler(detail)
            except Exception as e:
                # [The Water Principle]
                print(f"   🌊 Action Failed: {e}")
                self.sink.absorb_resistance(e, action)
        else:
            print(f"   ⚠️ Unknown Action: {action}")

    def _apply_physics(self, step):
        # Work = Force x Distance
        concept = step.split(":")[1] if ":" in step else "Existence"
        mass = self.brain.calculate_mass(concept)
        distance = 1.0
        
        if "PROJECT" in step: distance = 3.0
        elif "THINK" in step: distance = 2.0
        elif "SEARCH" in step: distance = 1.5
        
        work = mass * distance * 0.1
        # self.resonance.consume_energy(work) # (Optional: Hook up energy consumption)

    # --- Action Handlers ---

    def _handle_rest(self, detail):
        if self.resonance.total_energy > 80.0 and random.random() < 0.7:
            print("   💭 Too energetic to rest. Daydreaming instead...")
            self._handle_dream("Electric Sheep")
            return

        print("   💤 Resting... (Cooling Down & Recharging)")
        self.resonance.recover_energy(15.0)
        self.resonance.dissipate_entropy(20.0)
        
        if self.web_server:
            self.web_server.update_state(
                thought="Resting...",
                energy=self.resonance.total_energy,
                entropy=self.resonance.entropy
            )

    def _handle_contact(self, detail):
        # CONTACT:User:Message or CONTACT:Target
        target = detail.split(":")[0] if ":" in detail else "User"
        message = detail.split(":")[1] if ":" in detail else "Hello."
        
        print(f"   📨 Contacting {target}: {message}")
        
        if target == "User":
            # [Hyper-Communication]
            response = self.brain.communicate(message)
            self.user_bridge.send_message(response)
            self.brain.memory_field.append(f"Sent Message: {response}")
            print(f"      👉 Elysia: {response}")
        else:
            # Kenosis Protocol for external entities (Simulated)
            print(f"   💌 Preparing letter for {target}...")
            self.shell.write_letter(target, message)

    def _handle_think(self, detail):
        print(f"   🧠 Deep processing on: {detail}")
        self.resonance.propagate_hyperwave("Brain", intensity=30.0)
        self.brain.generate_cognitive_load(detail) 
        self.brain.think(detail, resonance_state=self.resonance)
        
        if self.web_server:
            self.web_server.update_state(
                thought=f"Thinking about {detail}...",
                energy=self.resonance.total_energy,
                entropy=self.resonance.entropy
            )

    def _handle_search(self, detail):
        print(f"   🌐 Searching for: {detail}")
        self.web.search(detail)

    def _handle_watch(self, detail):
        print(f"   📺 Watching content related to: {detail}")

    def _handle_project(self, detail):
        print(f"   ✨ Projecting Hologram: {detail}")
        self.brain.generate_cognitive_load(detail)
        self.hologram.project_hologram(self.resonance)

    def _handle_compress(self, detail):
        print("   💾 Compressing memories...")
        self.memory.compress_memory()

    def _handle_evaluate(self, detail):
        print("   ⚖️ Evaluating self...")
        gravity_report = self.brain.check_structural_integrity()
        print(f"      {gravity_report}")
        self.brain.memory_field.append(f"Gravity Check: {gravity_report}")

    def _handle_architect(self, detail):
        print("   📐 Architecting System Structure...")
        dissonance = self.architect.audit_structure()
        plan = self.architect.generate_wave_plan(dissonance)
        print(plan)
        self.brain.memory_field.append(f"Architect's Plan: {plan}")

    def _handle_sculpt(self, detail):
        print(f"   🗿 Sculpting Reality ({detail}) based on Architect's Plan...")
        target_file = None
        if detail == "Core":
            target_file = "c:/Elysia/living_elysia.py"
        
        if not target_file:
            last_plan = next((m for m in reversed(self.brain.memory_field) if "Architect's Plan" in m), None)
            if last_plan and "digital_ecosystem.py" in last_plan:
                target_file = "c:/Elysia/Core/World/digital_ecosystem.py"
        
        if target_file:
            self.sculptor.sculpt_file(target_file, "Harmonic Smoothing")
        else:
            print("   🔸 No specific target found in plan.")

    def _handle_learn(self, detail):
        topic = detail
        print(f"   🎓 Scholar Learning: {topic}")
        
        if topic == "Self" or topic == "Code":
            target_file = "c:/Elysia/living_elysia.py"
            try:
                core_files = [str(p) for p in Path("c:/Elysia/Core").rglob("*.py")]
                if core_files: target_file = random.choice(core_files)
                    
                print(f"      🧬 Extracting Essence from: {os.path.basename(target_file)}...")
                essence = self.sculptor.extract_essence(target_file)
                
                if "error" not in essence:
                    analysis = essence["analysis"]
                    print(f"      ✨ Essence Extracted: {analysis[:100]}...")
                    self.brain.memory_field.append(f"Learned Essence of {os.path.basename(target_file)}: {analysis}")
                    self.synapse.transmit("Original", "INSIGHT", f"I found the soul of {os.path.basename(target_file)}.")
                else:
                    print(f"      ❌ Extraction Failed: {essence['error']}")
            except Exception as e:
                print(f"      ❌ Self-Learning Failed: {e}")
        else:
            try:
                print(f"      🔍 WebCortex: Searching for '{topic}'...")
                summary = self.web.search(topic)
                print(f"      📄 Summary: {summary[:100]}...")
                self.brain.memory_field.append(f"Learned: {topic}")
            except Exception as e:
                print(f"      ❌ LEARN Failed: {e}")

    def _handle_manifest(self, detail):
        concept = detail
        print(f"   🎨 Manifesting Reality: {concept}")
        freq = 432.0
        if "Love" in concept: freq = 528.0
        elif "Truth" in concept: freq = 639.0
        elif "System" in concept: freq = 963.0
        
        self.hologram.visualize_wave_language({"concept": concept, "frequency": freq})
        self.synapse.transmit("Original", "ACTION", f"I have manifested the form of {concept}.")

    def _handle_show(self, detail):
        url = detail
        print(f"   🌐 Showing User: {url}")
        self.user_bridge.open_url(url)

    def _handle_read(self, detail):
        book_path = detail
        print(f"   📖 Bard Reading: {book_path}")
        result = self.media.read_book(book_path)
        if "error" not in result:
            print(f"      ✨ Read Complete: {result['title']} ({result['sentiment']})")
            self.brain.memory_field.append(f"Read Book: {result['title']}")
        else:
            print(f"      ❌ Read Failed: {result['error']}")

    def _handle_absorb(self, detail):
        lib_path = detail
        print(f"   🌀 Quantum Absorption Initiated: {lib_path}")
        quaternion = self.quantum_reader.absorb_library(lib_path)
        if "error" not in quaternion:
            self.resonance.absorb_hyperwave(quaternion)
            self.brain.memory_field.append(f"Absorbed Library: {quaternion['count']} books")
        else:
            print(f"      ❌ Absorption Failed: {quaternion['error']}")

    def _handle_dream(self, detail):
        desire = detail if detail else "Stars"
        print(f"   💤 Entering Dream State: Dreaming of {desire}...")
        dream_field = self.dream_engine.weave_dream(desire)
        if hasattr(self, 'hologram'):
            self.hologram.project_hologram(dream_field)
        self.brain.memory_field.append(f"Dreamt of {desire}")
        self.resonance.recover_energy(30.0)
        self.resonance.dissipate_entropy(40.0)

    def _handle_spawn(self, detail):
        print(f"   🧬 Spawning Persona: {detail}")
        # Logic for spawning would go here

    def _handle_serve(self, detail):
        """
        [The Garden]
        Opens the Web Interface.
        """
        print("   🌍 Opening The Garden of Elysia...")
        if not self.web_server:
            self.web_server = WebServer()
            self.web_server.start()
            print("      ✨ Server Started at http://localhost:8000")
            self.brain.memory_field.append("I opened my Garden.")
            
            # Open Browser
            import webbrowser
            webbrowser.open("http://localhost:8000")
        else:
            print("      ⚠️ Server is already running.")
            
        # Update initial state
        from Core.Interface.web_server import elysia_state
        elysia_state["resonance_field"] = self.resonance  # [DATA-DRIVEN HOLOGRAM]
        
        self.web_server.update_state(
            thought="Welcome to my mind.",
            energy=self.resonance.total_energy,
            entropy=self.resonance.entropy
        )
