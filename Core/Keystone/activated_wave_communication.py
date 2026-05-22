"""
               (Activated Wave Communication System)
================================================================

"               ..." -            

  : Ether                       
  :                    

  :
1.          -             
2.          -           
3.          -            
4.         -                
"""

import logging
from typing import List, Dict, Any, Callable, Optional
import time
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger("WaveCommunication")


@dataclass
class WaveMessage:
    """      """
    content: Any
    sender: str
    priority: float  # 0.0-1.0
    frequency: float
    target_modules: List[str]


class ActivatedWaveCommunication:
    """
                  
    
    Ether                 
    """
    
    def __init__(self):
        self.ether = None
        self.listeners = defaultdict(list)
        self.frequency_map = {}
        self.message_history = []
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'average_latency': 0.0,
            'resonance_hits': 0
        }
        
        # Ether   
        try:
            from Core.System.ether import ether, Wave
            self.ether = ether
            self.Wave = Wave
            logger.info("  Ether      ")
        except Exception as e:
            logger.error(f"  Ether      : {e}")
            return
        
        #          
        self._initialize_frequency_map()
        
        logger.info("                    ")
    
    def _initialize_frequency_map(self):
        """          """
        self.frequency_map = {
            #       
            'cognition': 432.0,      #       
            'emotion': 528.0,        #   /      
            'memory': 639.0,         #   /      
            'intelligence': 741.0,   #   /      
            'evolution': 852.0,      #   /       
            'consciousness': 963.0,  #           
            
            #      
            'broadcast': 111.0,      #      
            'urgent': 999.0,         #       
            'query': 222.0,          #   
            'response': 333.0,       #   
            
            #      
            'learning': 10.0,        # Alpha (     )
            'thinking': 40.0,        # Gamma (  )
            'dreaming': 4.0,         # Delta ( )
            'meditation': 7.5,       # Theta (  )
        }
    
    def register_module(self, module_name: str, frequency: float, callback: Callable):
        """
                       
        
        Args:
            module_name:      
            frequency:       
            callback:               
        """
        if not self.ether:
            logger.error("  Ether    -      ")
            return False
        
        # Ether  tune_in
        self.ether.tune_in(frequency, callback)
        self.listeners[module_name].append(frequency)
        
        logger.info(f"       : {module_name} @ {frequency}Hz")
        return True
    
    def send_wave_message(
        self,
        content: Any,
        sender: str,
        target_module: str = None,
        priority: float = 0.5
    ) -> bool:
        """
                 
        
        Args:
            content:       
            sender:    
            target_module:       (None   broadcast)
            priority:      (0.0-1.0)
        """
        if not self.ether:
            logger.error("  Ether    -      ")
            return False
        
        start_time = time.time()
        
        #       
        if target_module and target_module in self.frequency_map:
            frequency = self.frequency_map[target_module]
        else:
            frequency = self.frequency_map['broadcast']
        
        # Wave   
        wave = self.Wave(
            sender=sender,
            frequency=frequency,
            amplitude=priority,
            phase="MESSAGE",
            payload=content
        )
        
        #   
        self.ether.emit(wave)
        
        #        
        latency = (time.time() - start_time) * 1000  # ms
        self.stats['messages_sent'] += 1
        self._update_latency(latency)
        
        #        
        self.message_history.append({
            'time': time.time(),
            'sender': sender,
            'target': target_module,
            'frequency': frequency,
            'latency': latency
        })
        
        logger.debug(f"       : {sender}   {target_module or 'ALL'} ({frequency}Hz, {latency:.2f}ms)")
        return True
    
    def broadcast_to_all(self, content: Any, sender: str, priority: float = 0.7):
        """         """
        return self.send_wave_message(content, sender, None, priority)
    
    def send_to_multiple(
        self,
        content: Any,
        sender: str,
        targets: List[str],
        priority: float = 0.5
    ):
        """
                     (  )
        
                        !
        """
        if not self.ether:
            return False
        
        logger.info(f"          : {len(targets)}    ")
        
        for target in targets:
            self.send_wave_message(content, sender, target, priority)
        
        return True
    
    def query_and_wait(
        self,
        query: str,
        sender: str,
        target: str,
        timeout: float = 1.0
    ) -> Optional[Any]:
        """
                  
        
                    
        """
        if not self.ether:
            return None
        
        #         
        response_received = []
        response_freq = self.frequency_map['response']
        
        def response_listener(wave):
            if wave.payload.get('query_id') == query:
                response_received.append(wave.payload.get('answer'))
        
        #       
        self.ether.tune_in(response_freq, response_listener)
        
        #      
        self.send_wave_message(
            {'query': query, 'query_id': query, 'response_freq': response_freq},
            sender,
            target,
            priority=0.8
        )
        
        #      
        start_time = time.time()
        while len(response_received) == 0 and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if response_received:
            logger.info(f"       : {query}")
            return response_received[0]
        else:
            logger.warning(f"         : {query}")
            return None
    
    def create_resonance_network(self, modules: List[str]):
        """
                  
        
                           
                 
        """
        if not self.ether:
            return False
        
        #          
        resonance_freq = 432.0  #       
        
        logger.info(f"            : {len(modules)}     @ {resonance_freq}Hz")
        
        for module in modules:
            #                 
            if module in self.frequency_map:
                #                        
                pass
        
        return True
    
    def optimize_frequencies(self):
        """
               
        
                              
        """
        if len(self.message_history) < 10:
            return
        
        #          
        freq_usage = defaultdict(int)
        for msg in self.message_history[-100:]:  #    100 
            freq_usage[msg['frequency']] += 1
        
        #                  
        most_used = sorted(freq_usage.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"           :")
        for freq, count in most_used[:5]:
            logger.info(f"   {freq}Hz: {count} ")
    
    def _update_latency(self, new_latency: float):
        """            """
        count = self.stats['messages_sent']
        old_avg = self.stats['average_latency']
        self.stats['average_latency'] = (old_avg * (count - 1) + new_latency) / count
    
    def get_communication_stats(self) -> Dict:
        """     """
        return {
            'messages_sent': self.stats['messages_sent'],
            'messages_received': self.stats['messages_received'],
            'average_latency_ms': self.stats['average_latency'],
            'registered_modules': len(self.listeners),
            'available_frequencies': len(self.frequency_map),
            'ether_connected': self.ether is not None,
            'wave_history_size': len(self.message_history)
        }
    
    def calculate_wave_score(self) -> float:
        """
                   (100    )
        
             :
        - Ether   : 25 
        -     : 25 
        -      : 25 
        -       : 25 
        """
        score = 0.0
        
        # 1. Ether    (25 )
        if self.ether:
            score += 25
        
        # 2.      (25 ) - <10ms   
        avg_latency = self.stats['average_latency']
        if avg_latency > 0:
            latency_score = min(10 / avg_latency, 1.0) * 25
            score += latency_score
        
        # 3.       (25 ) -               
        msg_count = self.stats['messages_sent']
        usage_score = min(msg_count / 100, 1.0) * 25
        score += usage_score
        
        # 4.        (25 )
        if self.stats['messages_sent'] > 0:
            resonance_rate = self.stats['resonance_hits'] / self.stats['messages_sent']
            resonance_score = resonance_rate * 25
            score += resonance_score
        
        return score


#        
wave_comm = ActivatedWaveCommunication()


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("                   ")
    print("="*70)
    
    comm = ActivatedWaveCommunication()
    
    if not comm.ether:
        print("  Ether       -      ")
        exit(1)
    
    # 1.      
    print("\n1        ")
    print("-" * 70)
    
    def cognition_listener(wave):
        print(f"     Cognition received: {wave.payload}")
        comm.stats['messages_received'] += 1
    
    def emotion_listener(wave):
        print(f"      Emotion received: {wave.payload}")
        comm.stats['messages_received'] += 1
    
    comm.register_module('cognition', 432.0, cognition_listener)
    comm.register_module('emotion', 528.0, emotion_listener)
    
    # 2.          
    print("\n2            ")
    print("-" * 70)
    
    comm.send_wave_message("Hello Cognition!", "TestSender", "cognition", priority=0.8)
    time.sleep(0.1)  #         
    
    # 3.   
    print("\n3        ")
    print("-" * 70)
    
    comm.broadcast_to_all("System update available", "System", priority=0.9)
    time.sleep(0.1)
    
    # 4.      
    print("\n4           ")
    print("-" * 70)
    
    comm.send_to_multiple(
        "Urgent: System check",
        "Monitor",
        ['cognition', 'emotion', 'intelligence'],
        priority=1.0
    )
    time.sleep(0.1)
    
    # 5.   
    print("\n5        ")
    print("-" * 70)
    
    stats = comm.get_communication_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 6.      
    print("\n6          ")
    print("-" * 70)
    
    score = comm.calculate_wave_score()
    print(f"     : {score:.1f}/100")
    
    print("\n" + "="*70)
    print("       !")
    print("\n                     !")
    print("   -      : {:.2f}ms".format(stats['average_latency_ms']))
    print("   -   : {} ".format(stats['messages_sent']))
    print("   -   : {} ".format(stats['messages_received']))
    print("="*70 + "\n")
