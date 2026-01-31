"""
Core/Engine/Genesis/cognitive_cycle.py
======================================
The Mechanism of Self-Realization.

This module implements the feedback loop where Elysia:
1. Experiences (Runs Sim)
2. Suffers (Detects Bad Metrics like Starvation)
3. Realizes (Generates Insight)
4. Evolves (Changes the Law)
"""

import time
from typing import Dict, Any, Callable
from Core.S1_Body.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab

class CognitiveCycle:
    def __init__(self, lab: GenesisLab):
        self.lab = lab
        self.history = [] # Memory of what happened
        self.insights = [] # "I realized X"
        self.current_paradigm = "Naive" 
        
    def run_cycle(self, ticks: int, goal_metric: str = "Fairness"):
        """
        Runs the simulation and reflects on it.
        """
        print(f"\n  [Cognitive Cycle] Operating under Paradigm: {self.current_paradigm}")
        
        # 1. Experience (Run Sim)
        self.lab.run_simulation(ticks)
        
        # 2. Reflect (Analyze)
        analysis = self._analyze_metrics()
        
        # 3. Realize & Evolve
        if goal_metric == "Fairness":
            if analysis["max_wait_time"] > 5.0: # Threshold for suffering
                print(f"      [Suffering] Max Wait Time is {analysis['max_wait_time']} ticks! Someone is starving.")
                
                if self.current_paradigm == "Naive":
                    self._shift_paradigm("RoundRobin")
                else:
                    print("     I am already trying my best, but suffering persists.")
            else:
                print("     [Status] Fairness is acceptable.")
                
        elif goal_metric == "Throughput":
            # Deadlock Check: Throughput should increase. If 0 after many ticks, it's a deadlock.
            throughput = analysis["total_throughput"]
            print(f"     [Metric] Global Throughput: {throughput} meals.")
            
            if throughput == 0:
                print("      [Suffering] ZERO Throughput! The System is Frozen (Deadlock).")
                if self.current_paradigm == "Naive":
                    self._shift_paradigm("Hierarchy")
            else:
        elif goal_metric == "Reliability":
            # OOM Check: Did we crash?
            if "system_status" in self.lab.rotors[0].context and self.lab.rotors[0].context["system_status"] == "CRASHED_OOM":
                print("      [Suffering] SYSTEM CRASHED (OOM)! Memory full.")
                if self.current_paradigm == "Naive":
                    self._shift_paradigm("Paging")
                    # Recover from crash?
                    self.lab.rotors[0].context["system_status"] = "RECOVERING"
            else:
                 # Check usage percentage just for logs
                 pass
                 # print("     [Status] System Stable.")

    def _analyze_metrics(self) -> Dict[str, Any]:
        """
        Look at monads and calculate stats.
        """
        procs = [m for m in self.lab.monads if m.domain == "Process"]
        threads = [m for m in self.lab.monads if m.domain == "Thread"]
        
        stats = {"max_wait_time": 0, "total_throughput": 0}
        
        # Fairness Stats
        if procs:
            max_wait = 0
            for p in procs:
                w = p.props.get("wait_time", 0)
                if w > max_wait: max_wait = w
            stats["max_wait_time"] = max_wait
            
        # Throughput Stats
        if threads:
            total_meals = sum(t.props.get("meals_eaten", 0) for t in threads)
            stats["total_throughput"] = total_meals
            
        return stats
        
    def _shift_paradigm(self, new_paradigm: str):
        """
        The Evolution Step. Swaps the Law.
        """
        print(f"\n  [Realization] 'My current way ({self.current_paradigm}) causes suffering. I must evolve to {new_paradigm}.'")
        self.insights.append(f"Shifted from {self.current_paradigm} to {new_paradigm} due to suffering.")
        
        # Hot-swap Logic
        if new_paradigm == "RoundRobin":
            from Core.S1_Body.L6_Structure.Engine.Genesis.silicon_scholar_laws import law_round_robin_scheduling
            self.lab.rotors = [r for r in self.lab.rotors if "Scheduler" not in r.name]
            self.lab.decree_law("Round Robin Scheduler", law_round_robin_scheduling, rpm=60.0)
            for m in self.lab.monads: m.props["wait_time"] = 0
            
        elif new_paradigm == "Hierarchy":
            from Core.S1_Body.L6_Structure.Engine.Genesis.silicon_evolution_laws import law_resource_hierarchy
            self.lab.rotors = [r for r in self.lab.rotors if "Dining" not in r.name]
            self.lab.decree_law("Hierarchy Dining Law", law_resource_hierarchy, rpm=60.0)
            # Reset forks? Yes.
            for m in self.lab.monads:
                if m.domain == "Resource": m.val = 0
                if m.domain == "Thread": 
                    m.props["state"] = "THINKING"
                    m.props["meals_eaten"] = 0
                    
        elif new_paradigm == "Paging":
            from Core.S1_Body.L6_Structure.Engine.Genesis.silicon_evolution_laws import law_lru_paging
            self.lab.rotors = [r for r in self.lab.rotors if "OOM" in r.name]
            self.lab.decree_law("LRU Paging Law", law_lru_paging, rpm=60.0)
            # Restore Monads? In a test, we might re-populate to prove survival.
            # But the 'crash' law wiped them. So let's re-seed in the test script.
                
        self.current_paradigm = new_paradigm
