"""
Growth Journal (     )
==========================

                 .
"        "      "      ".

  :
1.       
2.       
3.      
4.       

     c:\Elysia\journals\       
                  .
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger("Elysia.GrowthJournal")

JOURNAL_DIR = "c:\\Elysia\\journals"


class GrowthJournal:
    """
         
    
    -           
    -         "     "    (      )
    -           (  )
    """
    
    def __init__(self):
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        self.today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"  GrowthJournal initialized for {self.today}")
    
    def write_entry(self, 
                    self_governance=None,
                    tension_field=None,
                    memory=None) -> str:
        """
                 
        
        [FIXED] SelfGovernance     (EmergentSelf   )
        
        Returns:      
        """
        from datetime import datetime
        
        # 1. SelfGovernance   
        if self_governance:
            status = self_governance.ideal_self.get_status()
            total_achievement = status["total_achievement"]
            aspects = status["aspects"]
            change_history = getattr(self_governance, 'change_history', [])
            current_focus = self_governance.current_focus
        else:
            total_achievement = 0
            aspects = {}
            change_history = []
            current_focus = None
        
        # 2. TensionField   
        field_status = self._get_field_status(tension_field)
        
        #      
        entry = f"""#      : {self.today}

##         

-       : {total_achievement:.1%}
-      : {current_focus.value if current_focus else '(  )'}

##           

"""
        for aspect_name, data in aspects.items():
            current = data.get('current', 0)
            target = data.get('target', 1)
            gap = data.get('gap', 0)
            entry += f"- **{aspect_name}**: {current:.2f}/{target:.2f} ( : {gap:.2f})\n"
        
        entry += f"""
##         (     )

"""
        if change_history:
            for change in change_history[-10:]:
                success_icon = " " if change.get('success') else " "
                aspect = change.get('aspect', 'unknown')
                before = change.get('before', 0)
                after = change.get('after', 0)
                delta = change.get('delta', 0)
                action = change.get('action', '')[:30]
                learning = change.get('learning', '')[:50]
                
                entry += f"- [{success_icon}] **{aspect}**: {before:.2f}   {after:.2f} (+{delta:.2f})\n"
                entry += f"  -   : {action}\n"
                entry += f"  -   : {learning}...\n"
        else:
            entry += "(             -                           )\n"
        
        if field_status:
            entry += f"""
##   TensionField   

{field_status}
"""
        
        entry += f"""
---

*      : {datetime.now().isoformat()}*
"""
        
        #       
        filepath = os.path.join(JOURNAL_DIR, f"{self.today}.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(entry)
        
        logger.info(f"  Journal entry written: {filepath}")
        
        return entry
    
    def _get_field_status(self, tension_field) -> str:
        """TensionField      """
        if not tension_field:
            return ""
        
        try:
            concept_count = len(tension_field.shapes)
            total_curvature = sum(s.curvature for s in tension_field.shapes.values())
            total_charge = sum(tension_field.charges.values())
            
            #      
            satellite_count = len(getattr(tension_field, 'satellites', {}))
            
            return f"""-     : {concept_count}
-     (  ): {total_curvature:.2f}
-     (   ): {total_charge:.2f}
-   (        ): {satellite_count}"""
        except:
            return "(TensionField         )"
    
    def read_yesterday(self) -> Optional[str]:
        """        """
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        filepath = os.path.join(JOURNAL_DIR, f"{yesterday}.md")
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def get_growth_trend(self, days: int = 7) -> str:
        """   N       """
        entries = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            filepath = os.path.join(JOURNAL_DIR, f"{date}.md")
            if os.path.exists(filepath):
                entries.append(date)
        
        if not entries:
            return "     "
        
        return f"   {len(entries)}       : {', '.join(entries[:3])}..."


#    
_journal = None

def get_growth_journal() -> GrowthJournal:
    global _journal
    if _journal is None:
        _journal = GrowthJournal()
    return _journal