# ExplorationCore: Elysia                         
import json
import os
from datetime import datetime

class ExplorationCore:
    def __init__(self, search_results_path=None, core_memory_path=None):
        self.search_results_path = search_results_path
        self.core_memory_path = core_memory_path
        self.insights = []
        
    def load_core_memory(self):
        """                              ."""
        if self.core_memory_path and os.path.exists(self.core_memory_path):
            with open(self.core_memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def formulate_query(self, topic):
        """Turns a topic of curiosity into a searchable query."""
        return f"What is {topic}? A philosophical and psychological definition."

    def record_insight(self, topic, insight):
        """                       ."""
        self.insights.append({
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "insight": insight,
            "source": "autonomous_exploration"
        })
        
    def explore(self, topic):
        """
                                       .
                     ,   ,               .
        """
        query = self.formulate_query(topic)
        core_memory = self.load_core_memory()
        
        if not self.search_results_path:
            self.record_insight(topic, "                      .")
            return None

        try:
            with open(self.search_results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            # print(f"[{self.__class__.__name__}] I have found {len(results)} potential answers.")
            #                                  
            if core_memory and "system_configuration" in core_memory:
                core_values = core_memory["system_configuration"]["core_values"]
                insight = f"'{topic}'                                        ."
                self.record_insight(topic, insight)
            return results
        except FileNotFoundError:
            self.record_insight(topic, "                         .")
            return None
            
    def get_insights(self):
        """                   ."""
        return self.insights
