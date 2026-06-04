#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HARDWARE_DIR="$SCRIPT_DIR/../hardware"

# Ensure C-Rotor is compiled
echo "Compiling C-Rotor Core..."
gcc -shared -o "$HARDWARE_DIR/c_rotor.so" -fPIC "$HARDWARE_DIR/c_rotor.c" -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful. Launching Hybrid Daemon Pipeline..."
echo "---------------------------------------------------------"

# Run pipeline with unbuffered Python execution
# hybrid_daemon.py now launches gpu_synapse.py itself for bidirectional communication.
python -u "$HARDWARE_DIR/hybrid_daemon.py"
