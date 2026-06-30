#!/bin/bash
# Auto-generated NEB Script for Sim_P120_s19_K100_N1_6407
# Config: Nodes=node-4-16*40, Mem=-S 1.9 -sizepar 1.5

# Uncomment if you need to load modules explicitly here
# module load ansys/2024R2

cd "C:\Users\IHarbi\OneDrive - Politecnico di Milano\Desktop\NonEqBaro_CFX\NEB_CFX_Pipeline\Water_Pipeline\Sim_P120_s19_K100_N1_6407"

echo "[INFO] Starting Solver..."
# Run Solver
/software/ansys/v242/CFX/bin\cfx5solve -def ../base_setup.def -ccl Material_K100_N1_6407.ccl -double \
  -start-method 'Intel MPI Local Parallel' -par-dist 'node-4-16*40' \
  -S 1.9 -sizepar 1.5 -name Result_K100_N1_6407

if [ $? -ne 0 ]; then echo "[ERROR] Solver Failed"; exit 1; fi

echo "[INFO] Starting Post-Process..."

# --- SMART RES FINDER ---
# CFX appends _001.res, _002.res etc. We need the latest one.
# ls -t sorts by time (newest first). head -n 1 takes the top one.
RES_FILE=$(ls -t Result_K100_N1_6407*.res 2>/dev/null | head -n 1)

if [ -f "$RES_FILE" ]; then
  echo "Found Result: $RES_FILE"
  /software/ansys/v242/CFX/bin\cfx5post -batch PostProcess_K100_N1_6407.cse -res "$RES_FILE"
  echo "[SUCCESS] Pipeline Complete"
else
  echo "[ERROR] No .res file found matching Result_K100_N1_6407*"
  exit 1
fi
