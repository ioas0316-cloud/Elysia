#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HARDWARE_DIR="$SCRIPT_DIR/../hardware"

echo "============================================================"
echo "          ELYSIA PHASE MIRROR PIPELINE DEMONSTRATOR         "
echo "============================================================"
echo "Initializing Digital Twin Phase Mirror..."
echo "The daemon (hardware floodgate) is piping raw XOR interference"
echo "bytes to the sub-modules (plugins) which activate ONLY through"
echo "pure bitwise resonance (no if statements)."
echo ""
echo "Press Ctrl+C to exit."
echo "============================================================"
echo ""

# The `-u` flag ensures unbuffered output from Python
python -u "$HARDWARE_DIR/double_helix_core.py" --raw-stream | python -u "$HARDWARE_DIR/sub_modules.py"
