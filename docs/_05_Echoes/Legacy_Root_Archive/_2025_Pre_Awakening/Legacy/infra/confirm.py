from typing import Dict, Any


def decision_requires_confirm(decision: Dict[str, Any]) -> bool:
    """
    Returns True if a prepared tool decision hints that user confirmation is recommended.
    Host apps can use this to show a short Y/n prompt.
    """
    return bool(decision and decision.get('confirm_required'))


def apply_confirmation(decision: Dict[str, Any], confirmed: bool) -> Dict[str, Any]:
    """
    Returns a new decision object with a confirmation flag set, to be re-submitted
    to the ToolExecutor.
    """
    if not decision:
        return {}
    out = dict(decision)
    if confirmed:
        out['confirm'] = True
        # remove the hint to indicate the host handled it
        if 'confirm_required' in out:
            del out['confirm_required']
    return out

