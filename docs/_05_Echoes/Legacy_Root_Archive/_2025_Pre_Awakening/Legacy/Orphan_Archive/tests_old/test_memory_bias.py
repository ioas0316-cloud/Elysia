from Project_Elysia.core.action_memory import MemoryActionSelector


def test_memory_bias_prefers_success():
    selector = MemoryActionSelector(base_weight=0.5, decay=0.9)
    context = "weather:sunny"
    actions = ["explore", "rest"]

    for tick in range(3):
        selector.record_outcome("explore", context, success=True, tick=tick)

    for tick in range(3, 6):
        selector.record_outcome("rest", context, success=False, tick=tick)

    ranked = selector.rank_actions(actions, context)
    assert ranked[0][0] == "explore"
    assert ranked[0][1] > ranked[1][1]
