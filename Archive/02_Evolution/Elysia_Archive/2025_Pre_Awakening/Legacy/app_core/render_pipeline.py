from __future__ import annotations

from typing import Iterable, List, Set


def filter_drawables(
    entities: Iterable[object],
    visible_layers: dict[str, bool],
    selected_ids: Set[int] | None,
    show_only_selected: bool,
) -> List[object]:
    """
    Entities may expose either attribute 'layer' or key ['layer'], and 'id'.
    """
    out: List[object] = []
    for e in entities:
        # layer
        layer = None
        if hasattr(e, "layer"):
            layer = getattr(e, "layer")
        elif isinstance(e, dict) and "layer" in e:
            layer = e["layer"]

        # id
        ent_id = None
        if hasattr(e, "id"):
            ent_id = getattr(e, "id")
        elif isinstance(e, dict) and "id" in e:
            ent_id = e["id"]

        # layer filter
        if not show_only_selected:
            if layer is not None and not visible_layers.get(str(layer), True):
                continue

        # selection filter
        if show_only_selected:
            if selected_ids and ent_id in selected_ids:
                out.append(e)
            continue

        out.append(e)
    return out

