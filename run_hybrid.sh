#!/bin/bash

# Ensure C-Rotor is compiled
echo "Compiling C-Rotor Core..."
gcc -shared -o c_rotor.so -fPIC c_rotor.c -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful. Launching Hybrid Daemon Pipeline..."
echo "---------------------------------------------------------"

# Run pipeline with unbuffered Python execution
# hybrid_daemon.py now launches gpu_synapse.py itself for bidirectional communication.
python -u hybrid_daemon.py
