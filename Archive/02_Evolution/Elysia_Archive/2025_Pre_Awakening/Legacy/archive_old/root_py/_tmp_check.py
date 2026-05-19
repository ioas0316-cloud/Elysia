import tools.godot_bridge_server as m
b = m.GodotBridge(m.BridgeConfig(sim_rate=1.0, frame_every=1))
print('WORLD_OK', getattr(b.world,'width',None))
init = b._build_init()
print('INIT_LENSES', init.get('lenses'))
fr = b._build_frame()
print('OVERLAY_KEYS', sorted(list(fr['overlays'].keys()))[:9])
