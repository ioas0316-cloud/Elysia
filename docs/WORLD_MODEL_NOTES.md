Planet Overlay — Events & Tools (View Layer)

- Telemetry event `route.arc`: payload {from_mod, to_mod, latency_ms, outcome}
- Report `tools/traffic_report.py`: summarizes top-N congested routes from telemetry
- Updater `tools/update_planet_routes.py`: updates `data/world/planet.json` routes from telemetry (thickness=throughput, latency_heat=avg latency)
- Truth remains unchanged: KG/geometry/causality/data are immutable; this is a view-only overlay.
